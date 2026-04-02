#!/usr/bin/env python3
"""
PD 分离：配置驱动的 vLLM Prefill / Decode / 负载均衡代理 启停控制。

两种用法：
  1) 命令行：
       python pd_service_ctl.py start --config configs/xxx.yaml --log_dir logs/ [--dry_run]
       python pd_service_ctl.py stop  [--config configs/xxx.yaml]
  2) 代码：
       from pd_service_ctl import load_config, PdServiceCtl
       cfg = load_config(Path("configs/xxx.yaml"))
       ctl = PdServiceCtl(cfg)
       ctl.start_stack(Path("logs"))
       ctl.stop()
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# 常量 & 工具
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent

_FALLBACK_NIC_NAME = "eth0"
_FALLBACK_LOCAL_IP = "172.17.0.4"

LogFn = Callable[[str], None]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_default(msg: str) -> None:
    print(f"{ts()} [pd] {msg}")


# ---------------------------------------------------------------------------
# 网卡 / IP 自动探测（保留原有逻辑）
# ---------------------------------------------------------------------------


def _ipv4_on_dev(dev: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["ip", "-json", "addr", "show", "dev", dev],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        data = json.loads(p.stdout)
        if not data:
            return None
        for info in data[0].get("addr_info", []):
            if info.get("family") != "inet":
                continue
            loc = info.get("local")
            if loc and not loc.startswith("127."):
                return loc
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_json() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "-json", "route", "show", "default"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        routes = json.loads(p.stdout)
        if not routes:
            return None
        r0 = routes[0]
        dev = r0.get("dev")
        if not dev:
            return None
        prefsrc = r0.get("prefsrc")
        if prefsrc:
            return dev, prefsrc
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_text() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        line = (p.stdout or "").strip().splitlines()[0]
        m = re.search(r"\bdefault\s+(?:via\s+\S+\s+)?dev\s+(\S+)", line)
        if not m:
            return None
        dev = m.group(1)
        m2 = re.search(r"\bsrc\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m2:
            return dev, m2.group(1)
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ifconfig() -> Optional[Tuple[str, str]]:
    for cmd in (["ifconfig"], ["ifconfig", "-a"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
            if p.returncode != 0:
                continue
            text = p.stdout or ""
        except OSError:
            continue
        for block in text.split("\n\n"):
            lines = [ln for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            m0 = re.match(r"^(\S+):", lines[0].strip())
            if not m0:
                continue
            iface = m0.group(1)
            if iface == "lo":
                continue
            rest = "\n".join(lines[1:])
            m = re.search(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", rest)
            if not m:
                continue
            ip = m.group(1)
            if ip.startswith("127."):
                continue
            return iface, ip
    return None


def _detect_default_nic_ip() -> Tuple[str, str]:
    for fn in (_detect_nic_ip_via_ip_json, _detect_nic_ip_via_ip_text, _detect_nic_ip_via_ifconfig):
        t = fn()
        if t:
            return t
    print(
        f"{ts()} [pd] NIC_NAME/LOCAL_IP 探测失败，"
        f"使用回退值: NIC_NAME={_FALLBACK_NIC_NAME!r} LOCAL_IP={_FALLBACK_LOCAL_IP!r}",
        file=sys.stderr,
    )
    return _FALLBACK_NIC_NAME, _FALLBACK_LOCAL_IP


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------


@dataclass
class InstanceConfig:
    """一个 vLLM 引擎实例（prefill 或 decode）。"""
    name: str                       # "P0", "D0", "D1"
    role: str                       # "prefill" | "decode"
    port: int
    devices: str                    # ASCEND_RT_VISIBLE_DEVICES, 如 "0,1"
    tensor_parallel_size: int
    dp_size: int                    # 同角色实例总数（自动计算）
    dp_rank: int                    # 在同角色列表中的 index（自动计算）
    dp_port: int
    kv_port: int
    engine_id: int
    hccl_buffsize: int
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterConfig:
    """从 YAML 解析出的完整集群配置。"""
    cluster_name: str
    model_path: str
    served_model_name: str
    venvs: Dict[str, Path]          # 所有可用 venv，key = 名称，value = 路径
    venv_activate: str               # 当前激活的 venv 名称（必须是 venvs 的 key）
    # 日志
    log_level: str              # DEBUG / INFO / WARNING / ERROR
    # 路径
    transfer_engine_lib: str
    python_lib: str
    # 网络
    nic_name: str
    local_ip: str
    # vLLM 参数：defaults → role defaults → instance overrides（优先级递增）
    vllm_defaults: Dict[str, Any]
    prefill_defaults: Dict[str, Any]
    decode_defaults: Dict[str, Any]
    # 实例
    prefill_instances: List[InstanceConfig]
    decode_instances: List[InstanceConfig]
    # 代理
    proxy_port: Optional[int]       # None = 不启代理
    proxy_prefill_only: bool = False  # True = decode 阶段打桩，专用于压测 prefill
    config_path: Optional[Path] = None  # 原始配置文件路径（供 pd_proxy.py start --config 使用）
    # 就绪等待
    ready_timeout_s: int = 300
    proxy_sleep_s: int = 10

    @property
    def active_venv(self) -> Path:
        """返回当前激活的 venv 路径。"""
        return self.venvs[self.venv_activate]


# ---------------------------------------------------------------------------
# YAML 加载 & 校验
# ---------------------------------------------------------------------------


def load_config(path: Path) -> ClusterConfig:
    """读取 YAML 配置文件并返回 ``ClusterConfig``。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 网络：null → 自动探测
    net = raw.get("network") or {}
    nic_name = net.get("nic_name")
    local_ip = net.get("local_ip")
    if not nic_name or not local_ip:
        detected_nic, detected_ip = _detect_default_nic_ip()
        nic_name = nic_name or detected_nic
        local_ip = local_ip or detected_ip

    # 解析实例
    prefill_raw = raw.get("prefill") or []
    decode_raw = raw.get("decode") or []
    prefill_count = len(prefill_raw)
    decode_count = len(decode_raw)

    def _parse_instances(items: list, role: str, count: int) -> List[InstanceConfig]:
        result: List[InstanceConfig] = []
        for idx, item in enumerate(items):
            result.append(InstanceConfig(
                name=item["name"],
                role=role,
                port=item["port"],
                devices=str(item["devices"]),
                tensor_parallel_size=item["tensor_parallel_size"],
                dp_size=count,
                dp_rank=idx,
                dp_port=item["dp_port"],
                kv_port=item.get("kv_port"),
                engine_id=item.get("engine_id"),
                hccl_buffsize=item.get("hccl_buffsize", 256 if role == "prefill" else 1024),
                overrides=item.get("overrides") or {},
            ))
        return result

    prefill_instances = _parse_instances(prefill_raw, "prefill", prefill_count)
    decode_instances = _parse_instances(decode_raw, "decode", decode_count)

    # 代理
    proxy_raw = raw.get("proxy")
    proxy_port = proxy_raw["port"] if proxy_raw else None
    proxy_prefill_only = bool(proxy_raw.get("prefill_only", False)) if proxy_raw else False

    # venv：支持多个 venv，通过 venv_activate 指定激活哪一个
    venv_raw = raw.get("venv") or {}
    venvs = {k: Path(v) for k, v in venv_raw.items()}
    venv_activate = raw.get("venv_activate")
    if not venvs:
        raise ValueError("配置缺少 venv 段，至少需要定义一个 venv")
    if venv_activate is None:
        # 未指定时：若只有一个 venv 则自动选中，否则报错
        if len(venvs) == 1:
            venv_activate = next(iter(venvs))
        else:
            raise ValueError(
                f"定义了多个 venv ({', '.join(venvs)}), 必须通过 venv_activate 指定激活哪一个"
            )
    if venv_activate not in venvs:
        raise ValueError(f"venv_activate={venv_activate!r} 未在 venv 中定义，可选: {', '.join(venvs)}")

    paths = raw.get("paths") or {}

    cfg = ClusterConfig(
        cluster_name=raw.get("cluster_name", "unnamed"),
        model_path=raw["model"]["path"],
        served_model_name=raw["model"].get("served_name", "default"),
        venvs=venvs,
        venv_activate=venv_activate,
        log_level=raw.get("log_level", "INFO").upper(),
        transfer_engine_lib=paths.get("transfer_engine_lib", "/usr/local/lib"),
        python_lib=paths.get("python_lib", ""),
        nic_name=nic_name,
        local_ip=local_ip,
        vllm_defaults=raw.get("vllm_defaults") or {},
        prefill_defaults=raw.get("prefill_defaults") or {},
        decode_defaults=raw.get("decode_defaults") or {},
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
        proxy_port=proxy_port,
        proxy_prefill_only=proxy_prefill_only,
        config_path=Path(path).resolve(),
    )

    # 将 kv_connector_extra_config 中的相对路径解析为绝对路径（基于配置文件所在目录）
    cfg_dir = Path(path).resolve().parent
    for role_defaults in (cfg.prefill_defaults, cfg.decode_defaults):
        kv_cfg = role_defaults.get("kv_transfer_config")
        if not kv_cfg:
            continue
        extra = kv_cfg.get("kv_connector_extra_config")
        if not extra:
            continue
        for key in ("UCM_CONFIG_FILE",):
            val = extra.get(key)
            if val and not Path(val).is_absolute():
                extra[key] = str((cfg_dir / val).resolve())

    # 校验
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ClusterConfig) -> None:
    """基本校验：端口唯一、设备不重叠。"""
    all_inst = cfg.prefill_instances + cfg.decode_instances
    ports = [inst.port for inst in all_inst]
    if len(ports) != len(set(ports)):
        raise ValueError(f"实例端口有重复: {ports}")
    devices_sets = [(inst.name, set(inst.devices.split(","))) for inst in all_inst]
    for i in range(len(devices_sets)):
        for j in range(i + 1, len(devices_sets)):
            overlap = devices_sets[i][1] & devices_sets[j][1]
            if overlap:
                raise ValueError(
                    f"实例 {devices_sets[i][0]} 和 {devices_sets[j][0]} 设备重叠: {overlap}"
                )


# ---------------------------------------------------------------------------
# PID 文件管理
# ---------------------------------------------------------------------------

PID_DIR = PKG_DIR / ".pid"


def _pid_file(cluster_name: str, name: str) -> Path:
    """返回实例对应的 PID 文件路径（存放在脚本同级 .pid/ 目录下）。
    文件名格式为 {cluster_name}-{name}.pid，避免不同集群实例 PID 混淆。
    """
    PID_DIR.mkdir(exist_ok=True)
    return PID_DIR / f"{cluster_name}-{name}.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _direct_child_pids(pid: int) -> List[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return []
    kids: List[int] = []
    for line in (out.stdout or "").strip().splitlines():
        line = line.strip()
        if line:
            try:
                kids.append(int(line))
            except ValueError:
                continue
    return kids


def _collect_pid_tree(root: int) -> set:
    seen: set = set()
    stack = [root]
    while stack:
        p = stack.pop()
        if p <= 0 or p in seen:
            continue
        seen.add(p)
        for c in _direct_child_pids(p):
            if c not in seen:
                stack.append(c)
    return seen


def _kill_pid_tree(pid: int, log: LogFn, label: str, wait_cap: int = 30) -> None:
    if not _pid_alive(pid):
        return
    tree = _collect_pid_tree(pid)
    others = sorted((tree - {pid}), reverse=True)
    for p in others:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    t0 = time.time()
    while time.time() - t0 < wait_cap:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)

    if _pid_alive(pid):
        log(f"[stop {label}] pid {pid} 在 {wait_cap}s 内仍存在，放弃。")


def _get_npu_hbm_usage(log: LogFn) -> Optional[Dict[int, int]]:
    """运行 ``npu-smi info`` 并返回各 NPU 位置索引到 HBM 已用显存（MB）的映射。

    npu-smi 按顺序输出 NPU 0、1、2…，返回字典 key 即为该顺序位置（0-based），
    与 ``ASCEND_RT_VISIBLE_DEVICES`` / ``devices`` 配置中的卡号一致。

    解析形如::

        | 0     910B2               | OK            | ...                                                |
        | 0                         | 0000:C1:00.0  | 0           0    / 0          3435 / 65536         |

    的两行一组结构：首行（NPU 行）提取 NPU 编号，次行（Chip 行）提取 HBM 已用值。
    """
    try:
        p = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"npu-smi info 执行失败: {e}")
        return None

    if p.returncode != 0:
        log(f"npu-smi info 返回码 {p.returncode}，stderr: {p.stderr.strip()[:200]}")
        return None

    # NPU 首行：以 "| <数字>  <型号名>" 开头，例如 "| 0     910B2"
    npu_head_re = re.compile(r'^\|\s*(\d+)\s+\S')
    # Chip 行：含 PCI 总线地址（格式: XXXX:XX:XX.X）
    pci_re = re.compile(r'\b[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]\b')
    usage_re = re.compile(r'(\d+)\s*/\s*\d+')

    hbm_map: Dict[int, int] = {}
    pending_npu_idx: Optional[int] = None

    for line in (p.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            pending_npu_idx = None
            continue

        if pending_npu_idx is None:
            m = npu_head_re.match(stripped)
            if m:
                pending_npu_idx = int(m.group(1))
        else:
            # 下一个 | 开头行：若是 Chip 行则提取 HBM
            if pci_re.search(stripped):
                matches = usage_re.findall(stripped)
                # 每行含两组 used/total：Memory-Usage 和 HBM-Usage，取最后一组
                if len(matches) >= 2:
                    hbm_map[pending_npu_idx] = int(matches[-1])
            pending_npu_idx = None

    return hbm_map if hbm_map else None


def _collect_device_indices(cfg: ClusterConfig) -> List[int]:
    """收集 config 中所有 prefill/decode 实例使用的卡号（即 npu-smi 输出中的位置索引）。"""
    indices: set = set()
    for inst in cfg.prefill_instances + cfg.decode_instances:
        for d in inst.devices.split(","):
            d = d.strip()
            if d:
                indices.add(int(d))
    return sorted(indices)



def _stop_by_pid_file(pid_file: Path, label: str, log: LogFn) -> bool:
    """读取 PID 文件并停止对应进程树。返回是否成功停止了进程。"""
    if not pid_file.is_file():
        log(f"[stop {label}] PID 文件 {pid_file} 不存在，跳过。")
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return False
    if not _pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        log(f"[stop {label}] PID {pid} 已不存在，清理 PID 文件。")
        return False
    log(f"[stop {label}] 结束 PID {pid}（含子进程）...")
    _kill_pid_tree(pid, log, label)
    pid_file.unlink(missing_ok=True)
    log(f"[stop {label}] 已停止。")
    return True


def _find_pids_by_pattern(pattern: str) -> List[int]:
    """用 pgrep -f 按命令行模式查找进程 PID。"""
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    pids: List[int] = []
    for line in (out.stdout or "").strip().splitlines():
        line = line.strip()
        if line:
            try:
                pids.append(int(line))
            except ValueError:
                continue
    return pids


def _stop_proxy_fallback(log: LogFn) -> None:
    """PID 文件失效时，按命令行模式兜底查杀 pd_proxy.py 进程。"""
    pids = _find_pids_by_pattern("pd_proxy.py")
    # 排除自身
    my_pid = os.getpid()
    pids = [p for p in pids if p != my_pid]
    if not pids:
        return
    log(f"[stop proxy] 兜底：通过命令行匹配找到 pd_proxy.py 进程 {pids}")
    for pid in pids:
        _kill_pid_tree(pid, log, "proxy-fallback")
    log("[stop proxy] 兜底清理完成。")


# ---------------------------------------------------------------------------
# LD_PRELOAD 探测
# ---------------------------------------------------------------------------

_LD_PRELOAD_CACHE: Optional[str] = None


def _resolve_ld_preload() -> str:
    global _LD_PRELOAD_CACHE
    if _LD_PRELOAD_CACHE is not None:
        return _LD_PRELOAD_CACHE

    stdcxx = ""
    for candidate in (
        "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    ):
        if Path(candidate).is_file():
            stdcxx = candidate
            break

    jemalloc = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
    parts = [p for p in [jemalloc, stdcxx] if p]
    _LD_PRELOAD_CACHE = ":".join(parts)
    return _LD_PRELOAD_CACHE


def _resolve_ld_library_path(cfg: ClusterConfig) -> str:
    parts = []
    if cfg.python_lib:
        parts.append(cfg.python_lib)
    if cfg.transfer_engine_lib:
        parts.append(cfg.transfer_engine_lib)
    parts.extend(["/usr/lib64", "/usr/lib/aarch64-linux-gnu", "/usr/lib"])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


# ---------------------------------------------------------------------------
# 命令构建
# ---------------------------------------------------------------------------


def _build_env(cfg: ClusterConfig, inst: InstanceConfig) -> Dict[str, str]:
    """构建一个 vLLM 实例的完整环境变量字典。"""
    omp_threads = str(cfg.vllm_defaults.get("omp_num_threads", 10))

    env = os.environ.copy()
    env.update({
        "ASCEND_RT_VISIBLE_DEVICES": inst.devices,
        "LD_PRELOAD": _resolve_ld_preload(),
        "LD_LIBRARY_PATH": _resolve_ld_library_path(cfg),
        "HCCL_IF_IP": cfg.local_ip,
        "GLOO_SOCKET_IFNAME": cfg.nic_name,
        "TP_SOCKET_IFNAME": cfg.nic_name,
        "HCCL_SOCKET_IFNAME": cfg.nic_name,
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": omp_threads,
        "HCCL_BUFFSIZE": str(inst.hccl_buffsize),
        "VLLM_DP_SIZE": str(inst.dp_size),
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(inst.dp_port),
        "VLLM_DP_RANK_LOCAL": "0",
        "VLLM_DP_RANK": str(inst.dp_rank),
        "VLLM_DP_SIZE_LOCAL": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
        "VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED": "1",
        "VLLM_LOGGING_LEVEL": cfg.log_level,
    })
    return env


def _get_kv_transfer_config(cfg: ClusterConfig, inst: InstanceConfig) -> str:
    """读取 prefill/decode_defaults 中的 kv_transfer_config 并注入实例级字段。

    合并优先级：role_defaults.kv_transfer_config → inst.overrides.kv_transfer_config
    kv_port / engine_id 若实例定义了且 dict 中尚无，则自动注入。
    """
    role_defaults = cfg.prefill_defaults if inst.role == "prefill" else cfg.decode_defaults
    kv_cfg: Dict[str, Any] = dict(role_defaults.get("kv_transfer_config") or {})
    kv_cfg.update(inst.overrides.get("kv_transfer_config") or {})
    if inst.kv_port is not None and "kv_port" not in kv_cfg:
        kv_cfg["kv_port"] = str(inst.kv_port)
    if inst.engine_id is not None and "engine_id" not in kv_cfg:
        kv_cfg["engine_id"] = str(inst.engine_id)
    return json.dumps(kv_cfg)


def _yaml_key_to_flag(key: str) -> str:
    """将 YAML key 转为 CLI flag: ``max_model_len`` → ``--max-model-len``。"""
    return "--" + key.replace("_", "-")


# 不映射到 CLI flag 的内部 key（仅用于环境变量或特殊处理）
_SKIP_KEYS = {"omp_num_threads", "kv_transfer_config"}


def _build_vllm_args(cfg: ClusterConfig, inst: InstanceConfig) -> List[str]:
    """构建 ``vllm serve ...`` 的完整参数列表。

    ``vllm_defaults`` 与实例 ``overrides`` 合并后，自动将每个 key
    转为 ``--key-name value`` 形式的 CLI 参数：
    - 值为 ``true`` 的布尔型 → 仅追加 flag（如 ``--enforce-eager``）
    - 值为 ``false`` 的布尔型 → 跳过
    - 其他 → ``--flag value``
    """
    # 合并优先级：vllm_defaults → prefill/decode_defaults → instance overrides
    role_defaults = cfg.prefill_defaults if inst.role == "prefill" else cfg.decode_defaults
    merged = {**cfg.vllm_defaults, **role_defaults, **inst.overrides}

    args = [
        "vllm", "serve", cfg.model_path,
        "--host", "0.0.0.0",
        "--port", str(inst.port),
        "--tensor-parallel-size", str(inst.tensor_parallel_size),
        "--served-model-name", cfg.served_model_name,
    ]

    # 通用参数映射
    for key, value in merged.items():
        if key in _SKIP_KEYS:
            continue
        flag = _yaml_key_to_flag(key)
        if isinstance(value, bool):
            if value:
                args.append(flag)
            # False → 不加 flag
        else:
            args.extend([flag, str(value)])

    # decode 多实例时加 data-parallel 参数
    if inst.role == "decode" and inst.dp_size > 1:
        args.extend([
            "--nnodes", "1",
            "--data-parallel-size", str(inst.dp_size),
            "--data-parallel-rank", str(inst.dp_rank),
            "--data-parallel-address", "127.0.0.1",
            "--data-parallel-rpc-port", str(inst.dp_port),
            "--data-parallel-size-local", "1",
        ])

    # kv-transfer-config（prefill-only 模式下 prefill 实例不挂 KV connector）
    role_defaults = cfg.prefill_defaults if inst.role == "prefill" else cfg.decode_defaults
    has_kv_cfg = bool(role_defaults.get("kv_transfer_config") or inst.overrides.get("kv_transfer_config"))
    if has_kv_cfg:
        args.extend(["--kv-transfer-config", _get_kv_transfer_config(cfg, inst)])

    return args


def _build_proxy_args(cfg: ClusterConfig) -> List[str]:
    """构建内置代理 pd_proxy.py 的启动参数列表。"""
    proxy_script = PKG_DIR / "pd_proxy.py"
    venv_python = cfg.active_venv / "bin" / "python3"
    config_path = cfg.config_path or proxy_script.parent  # 兜底：不应发生
    return [str(venv_python), str(proxy_script), "start", "--config", str(config_path)]


# ---------------------------------------------------------------------------
# Dry-run 输出
# ---------------------------------------------------------------------------


def dry_run(cfg: ClusterConfig, log_dir: Path) -> None:
    """打印所有实例的启动命令和环境变量（不实际执行）。"""
    log_dir = log_dir.resolve()
    all_inst = cfg.prefill_instances + cfg.decode_instances

    print(f"{'=' * 72}")
    print(f"  Cluster: {cfg.cluster_name}")
    print(f"  Model:   {cfg.model_path}")
    print(f"  Network: NIC={cfg.nic_name} IP={cfg.local_ip}")
    print(f"  Log dir: {log_dir}")
    print(f"{'=' * 72}\n")

    for inst in all_inst:
        env = _build_env(cfg, inst)
        args = _build_vllm_args(cfg, inst)

        # 只打印与父进程不同的环境变量
        diff_env = {k: v for k, v in env.items() if os.environ.get(k) != v}

        role_tag = inst.role.upper()
        log_file = _instance_log_file(inst, log_dir)

        print(f"--- [{role_tag}] {inst.name} (port={inst.port}, devices={inst.devices}) ---")
        print(f"  PID file: {_pid_file(cfg.cluster_name, inst.name)}")
        print(f"  Log file: {log_file}")
        print(f"  Key env vars:")
        for k in sorted(diff_env):
            if k in ("PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
                      "LOGNAME", "HOSTNAME", "SHLVL", "_"):
                continue
            print(f"    {k}={diff_env[k]}")
        print(f"  Command:")
        print(f"    {' '.join(shlex.quote(a) for a in args)}")
        print()

    if cfg.proxy_port is not None:
        proxy_args = _build_proxy_args(cfg)
        print(f"--- [PROXY] (port={cfg.proxy_port}) ---")
        print(f"  PID file: {_pid_file(cfg.cluster_name, 'proxy')}")
        print(f"  Log file: {log_dir / 'proxy.log'}")
        print(f"  Command:")
        print(f"    {' '.join(shlex.quote(a) for a in proxy_args)}")
        print()


# ---------------------------------------------------------------------------
# 服务编排
# ---------------------------------------------------------------------------


def _instance_log_file(inst: InstanceConfig, log_dir: Path) -> Path:
    """返回实例日志文件路径。"""
    return log_dir / f"{inst.role}_{inst.name}.log"


class PdServiceCtl:
    """
    PD 服务编排：基于 ClusterConfig 启动 / 停止所有 vLLM 实例和代理。

    ``log`` 为行级回调；外部调用者可传入自定义日志函数。
    """

    def __init__(self, cfg: ClusterConfig, log: LogFn = log_default) -> None:
        self._cfg = cfg
        self._log = log

    @property
    def cfg(self) -> ClusterConfig:
        return self._cfg

    @property
    def has_proxy(self) -> bool:
        return self._cfg.proxy_port is not None

    # ---------- 启动单个实例 ----------

    def start_instance(self, inst: InstanceConfig, log_dir: Path) -> None:
        """启动一个 vLLM 实例（prefill 或 decode）。"""
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        env = _build_env(self._cfg, inst)
        args = _build_vllm_args(self._cfg, inst)
        log_file = _instance_log_file(inst, log_dir)

        venv_activate = self._cfg.active_venv / "bin" / "activate"
        if not venv_activate.is_file():
            raise FileNotFoundError(f"venv activate 不存在: {venv_activate}")

        cmd_str = " ".join(shlex.quote(a) for a in args)
        inner = f"""
set -euo pipefail
source "{venv_activate}"
set -m
nohup {cmd_str} >> "{log_file}" 2>&1 &
echo $! > "{_pid_file(self._cfg.cluster_name, inst.name)}"
"""
        self._log(f"启动 {inst.role} [{inst.name}] (port={inst.port}, devices={inst.devices})...")
        _SKIP_ENV = {"PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
                     "LOGNAME", "HOSTNAME", "SHLVL", "_"}
        diff_env = {k: v for k, v in env.items()
                    if os.environ.get(k) != v and k not in _SKIP_ENV}
        self._log("  env:")
        for k in sorted(diff_env):
            self._log(f"    {k}={diff_env[k]}")
        self._log(f"  command: {' '.join(shlex.quote(a) for a in args)}")
        subprocess.run(
            ["/bin/bash", "-c", inner],
            check=True,
            cwd=str(log_dir),
            env=env,
        )

    # ---------- 启动代理 ----------

    def start_proxy(self, log_dir: Path) -> None:
        """启动内置负载均衡代理 pd_proxy.py。"""
        if self._cfg.proxy_port is None:
            self._log("配置中未定义 proxy，跳过。")
            return

        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        proxy_args = _build_proxy_args(self._cfg)
        cmd_str = " ".join(f'"{a}"' if " " in a else a for a in proxy_args)
        log_file = log_dir / "proxy.log"

        inner = f"""
set -euo pipefail
nohup {cmd_str} >> "{log_file}" 2>&1 &
echo $! > "{_pid_file(self._cfg.cluster_name, 'proxy')}"
"""
        env = os.environ.copy()
        env["LOCAL_IP"] = self._cfg.local_ip

        self._log(f"启动代理 (port={self._cfg.proxy_port})...")
        self._log(f"  command: {' '.join(shlex.quote(a) for a in proxy_args)}")
        subprocess.run(
            ["/bin/bash", "-c", inner],
            check=True,
            cwd=str(log_dir),
            env=env,
        )

    # ---------- 端口就绪等待 ----------

    def wait_for_port(
        self,
        port: int,
        name: str,
        timeout_s: Optional[int] = None,
    ) -> bool:
        t = self._cfg.ready_timeout_s if timeout_s is None else timeout_s
        self._log(f"等待 {name} 端口 {port} 就绪（超时 {t}s）...")
        elapsed = 0
        url = f"http://localhost:{port}/health"
        step = 5
        while elapsed < t:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        self._log(f"{name} 已就绪（已等待 {elapsed}s）。")
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(step)
            elapsed += step
        self._log(f"ERROR: {name} 在 {t}s 内未就绪。")
        return False

    # ---------- 一键启动 ----------

    def start_stack(
        self,
        log_dir: Path,
        *,
        with_proxy: Optional[bool] = None,
        wait_ready: bool = True,
    ) -> int:
        """
        顺序启动 prefill → decode →（可选）代理。

        ``with_proxy`` 为 ``None`` 时：配置中有 proxy 则启。
        返回 0 成功。
        """
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        wp = self.has_proxy if with_proxy is None else with_proxy
        if wp and not self.has_proxy:
            self._log("ERROR: 指定了代理但配置中未定义 proxy")
            return 1

        # 启动 prefill
        for inst in self._cfg.prefill_instances:
            try:
                self.start_instance(inst, log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: {inst.name} 启动失败: {e}")
                self.stop()
                return 1

        if wait_ready:
            for inst in self._cfg.prefill_instances:
                if not self.wait_for_port(inst.port, f"prefill[{inst.name}]"):
                    self.stop()
                    return 1

        # 启动 decode（prefill-only 模式下跳过，prefill 不挂 KV connector，无需 decode 节点）
        if self._cfg.proxy_prefill_only:
            self._log("prefill-only 模式：跳过 decode 实例启动")
        else:
            for inst in self._cfg.decode_instances:
                try:
                    self.start_instance(inst, log_dir)
                except (FileNotFoundError, subprocess.CalledProcessError) as e:
                    self._log(f"ERROR: {inst.name} 启动失败: {e}")
                    self.stop()
                    return 1

            if wait_ready:
                for inst in self._cfg.decode_instances:
                    if not self.wait_for_port(inst.port, f"decode[{inst.name}]"):
                        self.stop()
                        return 1

        # 启动代理
        if wp:
            try:
                self.start_proxy(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: 代理启动失败: {e}")
                self.stop()
                return 1
            time.sleep(self._cfg.proxy_sleep_s)

        self._log(f"PD 服务已拉起，日志目录: {log_dir}")
        return 0

    # ---------- 停止 ----------

    def stop(self) -> None:
        """停止所有实例：代理 → decode → prefill。"""
        # 代理：先尝试 PID 文件，失败则兜底按进程名查杀
        stopped = _stop_by_pid_file(_pid_file(self._cfg.cluster_name, "proxy"), "proxy", self._log)
        if not stopped:
            _stop_proxy_fallback(self._log)

        # decode（逆序）
        for inst in reversed(self._cfg.decode_instances):
            _stop_by_pid_file(_pid_file(self._cfg.cluster_name, inst.name), inst.name, self._log)

        # prefill（逆序）
        for inst in reversed(self._cfg.prefill_instances):
            _stop_by_pid_file(_pid_file(self._cfg.cluster_name, inst.name), inst.name, self._log)

    def stop_proxy(self) -> None:
        """仅停止代理。"""
        stopped = _stop_by_pid_file(_pid_file(self._cfg.cluster_name, "proxy"), "proxy", self._log)
        if not stopped:
            _stop_proxy_fallback(self._log)

    def restart_proxy(self, log_dir: Path) -> int:
        """重启代理：先停后启。"""
        self._log("重启代理...")
        self.stop_proxy()
        time.sleep(1)
        if not self.has_proxy:
            self._log("ERROR: 配置中未定义 proxy，无法重启。")
            return 1
        try:
            self.start_proxy(log_dir)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            self._log(f"ERROR: 代理启动失败: {e}")
            return 1
        time.sleep(self._cfg.proxy_sleep_s)
        self._log("代理已重启。")
        return 0

    # ---------- NPU 显存检查 ----------

    def wait_npu_memory_release(
        self,
        device_indices: List[int],
        threshold_mb: int = 5000,
        timeout_s: int = 300,
        poll_interval_s: int = 3,
    ) -> bool:
        """
        轮询 ``npu-smi info``，直到 ``device_indices`` 指定卡的 HBM 已用显存均低于
        ``threshold_mb``。

        ``device_indices`` 为 npu-smi 输出中的位置索引（与 devices 配置中的卡号对应），
        例如 ``devices: "2,3"`` 对应 ``[2, 3]``。

        返回 True 表示显存已释放，False 表示超时。
        """
        self._log(
            f"等待 NPU 卡 {device_indices} 显存释放"
            f"（阈值 {threshold_mb} MB，超时 {timeout_s}s）..."
        )
        t0 = time.time()
        while True:
            hbm_map = _get_npu_hbm_usage(self._log)
            if hbm_map is not None:
                # 只取 config 中配置的卡
                target = {idx: hbm_map[idx] for idx in device_indices if idx in hbm_map}
                missing = [idx for idx in device_indices if idx not in hbm_map]
                if missing:
                    self._log(f"WARNING: npu-smi 输出中未找到卡 {missing}，已跳过。")
                self._log(
                    f"当前 HBM 用量: "
                    + ", ".join(f"NPU{k}={v}MB" for k, v in sorted(target.items()))
                )
                if target and all(v < threshold_mb for v in target.values()):
                    self._log(
                        f"NPU 卡 {device_indices} 显存已释放"
                        f"（最大 {max(target.values())} MB < {threshold_mb} MB）。"
                    )
                    return True
            elapsed = time.time() - t0
            if elapsed >= timeout_s:
                self._log(
                    f"ERROR: NPU 卡 {device_indices} 显存在 {timeout_s}s 内未释放"
                    f"（当前: {hbm_map}）。"
                )
                return False
            time.sleep(poll_interval_s)

    # ---------- 整栈重启 ----------

    def restart(
        self,
        log_dir: Path,
        *,
        mem_threshold_mb: int = 5000,
        mem_timeout_s: int = 300,
        wait_ready: bool = True,
    ) -> int:
        """
        重启全部服务：stop → 等待 NPU 显存 < ``mem_threshold_mb`` → start_stack。

        返回 0 成功，非 0 失败。
        """
        self._log("=== restart: 开始停止所有服务 ===")
        self.stop()
        device_indices = _collect_device_indices(self._cfg)
        self._log(f"=== restart: 服务已停止，等待 NPU 卡 {device_indices} 显存释放 ===")
        if not self.wait_npu_memory_release(
            device_indices=device_indices,
            threshold_mb=mem_threshold_mb,
            timeout_s=mem_timeout_s,
        ):
            return 1
        self._log("=== restart: 显存已释放，重新启动服务 ===")
        return self.start_stack(log_dir, wait_ready=wait_ready)

    @staticmethod
    def stop_all(log: LogFn = log_default) -> None:
        """扫描 .pid/ 目录停止所有残留实例（无需配置文件）。"""
        pid_files = sorted(PID_DIR.glob("*.pid")) if PID_DIR.is_dir() else []
        if not pid_files:
            log(f"未找到 {PID_DIR}/*.pid。")
        else:
            for pf in pid_files:
                label = pf.stem
                _stop_by_pid_file(pf, label, log)
        # 兜底查杀 proxy
        _stop_proxy_fallback(log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PD Prefill / Decode / 代理 启停（配置驱动）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="启动 prefill → decode →（可选）代理")
    p_start.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML 配置文件路径",
    )
    p_start.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="日志目录（必填）",
    )
    p_start.add_argument("--no_wait", action="store_true", help="不等待 /health")
    p_start.add_argument("--dry_run", action="store_true", help="只打印命令，不实际启动")

    p_stop = sub.add_parser("stop", help="停止所有实例")
    p_stop.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="YAML 配置文件（不指定则扫描 .pid/ 目录）",
    )

    p_restart = sub.add_parser("restart", help="重启全部服务：stop → 等待 NPU 显存释放 → start")
    p_restart.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML 配置文件路径",
    )
    p_restart.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="日志目录（必填）",
    )
    p_restart.add_argument(
        "--mem_threshold_mb",
        type=int,
        default=5000,
        help="HBM 显存释放阈值（MB，默认 5000）",
    )
    p_restart.add_argument(
        "--mem_timeout_s",
        type=int,
        default=300,
        help="等待显存释放的超时时间（秒，默认 300）",
    )
    p_restart.add_argument("--no_wait", action="store_true", help="不等待 /health 就绪")

    p_rp = sub.add_parser("restart-proxy", help="仅重启代理（不影响 P/D 实例）")
    p_rp.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML 配置文件路径",
    )
    p_rp.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="日志目录（必填）",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_cli_parser().parse_args(argv)

    if args.cmd == "stop":
        if args.config:
            cfg = load_config(args.config)
            PdServiceCtl(cfg).stop()
        else:
            PdServiceCtl.stop_all()
        return 0

    if args.cmd == "restart":
        cfg = load_config(args.config)
        log_dir = args.log_dir.resolve()
        return PdServiceCtl(cfg).restart(
            log_dir,
            mem_threshold_mb=args.mem_threshold_mb,
            mem_timeout_s=args.mem_timeout_s,
            wait_ready=not args.no_wait,
        )

    if args.cmd == "restart-proxy":
        cfg = load_config(args.config)
        log_dir = args.log_dir.resolve()
        return PdServiceCtl(cfg).restart_proxy(log_dir)

    # start
    cfg = load_config(args.config)

    log_dir = args.log_dir.resolve()

    if args.dry_run:
        dry_run(cfg, log_dir)
        return 0

    ctl = PdServiceCtl(cfg)
    return ctl.start_stack(log_dir, with_proxy=None, wait_ready=not args.no_wait)


if __name__ == "__main__":
    raise SystemExit(main())
