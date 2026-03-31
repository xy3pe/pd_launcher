# PD 分离（Prefill / Decode）编排与结果分析

本目录包含 **PD 分离推理** 的启停编排以及压测日志的离线分析脚本。

所有部署配置集中在 **`configs/`** 目录下的 YAML 文件中，每个文件描述一个完整的 PD 集群（模型路径、实例列表、端口、卡号等），无需手写 shell 脚本。

---

## 快速开始

### 方式一：Web UI（推荐）

```bash
# 启动 Web 启动器（默认端口 8080）
python pd_launcher.py

# 指定端口或配置目录
python pd_launcher.py --port 8080 --configs configs/
```

浏览器访问 `http://<host>:8080`，即可通过页面：
- **配置管理**：新建 / 编辑 / 另存为 / 删除 YAML 配置文件
- **服务控制**：选择配置、启停控制服务器（`pd_service_server`）、启停 PD 集群、实时查看任务日志与 NPU HBM 用量

### 方式二：命令行

```bash
# 一键拉起 PD 集群（prefill → decode → 代理）
python pd_service_ctl.py start --config configs/qwen3_32b_1p2_1d2.yaml --log_dir logs/

# 预览将要执行的命令（不实际启动）
python pd_service_ctl.py start --config configs/qwen3_32b_1p2_1d2.yaml --log_dir logs/ --dry_run

# 停止所有实例
python pd_service_ctl.py stop --config configs/qwen3_32b_1p2_1d2.yaml
# 或扫描 .pid/ 目录全部停止（无需指定配置）
python pd_service_ctl.py stop

# 重启全部服务（stop → 等 NPU 显存释放 → start）
python pd_service_ctl.py restart --config configs/qwen3_32b_1p2_1d2.yaml --log_dir logs/

# 单独重启代理（不影响 P/D 实例）
python pd_service_ctl.py restart-proxy --config configs/qwen3_32b_1p2_1d2.yaml --log_dir logs/

# 直接启动 HTTP 控制服务器（支持远程 API 调用）
python pd_service_server.py --config configs/qwen3_32b_1p2_1d2.yaml --port 8088
```

---

## 配置文件（`configs/`）

每个 YAML 文件定义一个 PD 集群，核心结构：

```yaml
cluster_name: "Qwen3-32B_1P2_2D2"

model:
  path: "/root/autodl-tmp/models/Qwen3-32B"
  served_name: "qwen3_32b"

venv:
  vllm: "/root/autodl-tmp/py_venv/vllm2"

network:
  nic_name: null    # null = 自动探测
  local_ip: null

vllm_defaults:
  dtype: "bfloat16"
  max_model_len: 32768
  # ... 所有实例共享的 vLLM serve 参数

prefill_defaults:      # 覆盖 vllm_defaults，仅对 prefill 生效
  kv_transfer_config:
    kv_connector: "MooncakeConnectorV1"
    kv_role: "kv_producer"
    # ... 其他 connector 参数

decode_defaults:       # 覆盖 vllm_defaults，仅对 decode 生效
  kv_transfer_config:
    kv_connector: "MooncakeConnectorV1"
    kv_role: "kv_consumer"
    # ... 其他 connector 参数

prefill:
  - name: "P0"
    port: 9000
    devices: "0,1"
    tensor_parallel_size: 2
    dp_port: 13395      # 同角色实例共享
    kv_port: 20001      # 可选；存在时自动注入到 kv_transfer_config
    engine_id: 0        # 可选；存在时自动注入到 kv_transfer_config
    hccl_buffsize: 256
    overrides: {}       # 实例级别参数覆盖（可含 kv_transfer_config 覆盖）

decode:
  - name: "D0"
    port: 9010
    devices: "2,3"
    tensor_parallel_size: 2
    dp_port: 13495      # 与 prefill 不同
    kv_port: 20002      # 可选；存在时自动注入到 kv_transfer_config
    engine_id: 1        # 可选；存在时自动注入到 kv_transfer_config
    hccl_buffsize: 1024

proxy:
  port: 8000            # 删除此段或设为 null 则不启代理
  prefill_only: false   # true = 打桩 decode，专用于压测 prefill 性能
```

**参数优先级**：`vllm_defaults` → `prefill/decode_defaults` → 实例 `overrides`（优先级递增）。

**`kv_transfer_config`**：在 `prefill_defaults` / `decode_defaults` 中分别配置，直接序列化为 `--kv-transfer-config` JSON。实例字段 `kv_port` / `engine_id` 若存在且 dict 中尚无，会自动注入。支持在实例 `overrides.kv_transfer_config` 中覆盖部分字段。

**现有配置**：

| 文件 | 说明 |
|------|------|
| `qwen3_32b_1p2_1d2.yaml` | Qwen3-32B，1P(TP=2, 卡 2-3) + 1D(TP=2, 卡 4-5)，CPUOffloadingConnector（num_cpu_blocks=5000），proxy prefill-only 模式（port 9050） |
| `qwen3_32b_1p2_1d2_cpuoffload.yaml` | Qwen3-32B，1P(TP=2, 卡 0-1) + 1D(TP=2, 卡 2-3)，CPUOffloadingConnector（offload_strategy=eviction），proxy prefill-only 模式（port 8050） |
| `qwen3_32b_prefill_ucm.yaml` | Qwen3-32B，1P(TP=2, 卡 0-1) + 1D(TP=2, 卡 4-5)，UCMConnector（内存缓存，引用 `ucm_config/32b_ucm_memory_only.yaml`），proxy prefill-only 模式（port 9050） |
| `ucm_config/32b_ucm_memory_only.yaml` | UCM connector 配置，UcmPipelineStore 使用 Cache\|Empty 流水线，256 GB 内存缓存 |

---

## `pd_launcher.py`

**作用**：轻量 Web 启动器，提供一体化 UI，无需手写命令行即可完成全部操作。

**启动**

```bash
python pd_launcher.py [--host 0.0.0.0] [--port 8080] [--configs configs/]
```

**API**（供前端调用，也可直接使用）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/api/configs` | 列出所有配置文件 |
| `GET` | `/api/configs/<name>` | 读取配置内容 |
| `POST` | `/api/configs` | 保存配置（body: `{name, content}`） |
| `DELETE` | `/api/configs/<name>` | 删除配置 |
| `GET` | `/api/template` | 获取默认配置模板 |
| `GET` | `/api/server` | 控制服务器运行状态 |
| `POST` | `/api/server/start` | 启动控制服务器（body: `{config, port}`） |
| `POST` | `/api/server/stop` | 停止控制服务器 |
| `ANY` | `/pd/*` | 透明代理到 pd_service_server |

**说明**

- 纯标准库实现，无额外依赖
- `/pd/*` 路由将请求透明代理到正在运行的 `pd_service_server`（`/pd/status` → `/status`，`/pd/start` → `/start` 等）
- 控制服务器以子进程方式启动，Launcher 退出时自动终止

---

## `pd_service_ctl.py`

**作用**：统一拉起 / 停止 / 重启 **Prefill → Decode →（可选）代理**。

**命令行**

```bash
# 启动（默认等待 /health 就绪）
python pd_service_ctl.py start --config <yaml> --log_dir <dir> [--no_wait] [--dry_run]

# 停止（指定配置或扫描 .pid/ 全部停止）
python pd_service_ctl.py stop [--config <yaml>]

# 重启全部服务：stop → 等待 NPU HBM 显存释放 → start
python pd_service_ctl.py restart --config <yaml> --log_dir <dir> \
    [--mem_threshold_mb 5000] [--mem_timeout_s 300] [--no_wait]

# 仅重启代理（不影响 P/D 实例）
python pd_service_ctl.py restart-proxy --config <yaml> --log_dir <dir>
```

**`restart` 流程**

1. 停止所有实例（代理 → decode → prefill）
2. 轮询 `npu-smi info`，检查配置中所用各卡的 HBM 已用显存（`devices: "2,3"` 对应 npu-smi 输出中第 2、3 张卡），每 3 秒一次，直到全部低于阈值（默认 5000 MB）
3. 按正常顺序重新启动

**代码调用**

```python
from pd_service_ctl import load_config, PdServiceCtl
cfg = load_config(Path("configs/xxx.yaml"))
ctl = PdServiceCtl(cfg)
ctl.start_stack(Path("logs"))
ctl.stop()
ctl.restart(Path("logs"), mem_threshold_mb=5000, mem_timeout_s=300)
```

**日志文件**

| 文件 | 内容 |
|------|------|
| `{role}_{name}.log` | vLLM 实例 stdout/stderr（nohup 重定向，不限大小） |
| `proxy.log` | 代理进程 stdout/stderr |

**说明**

- `NIC_NAME` / `LOCAL_IP` 在配置中设为 `null` 时自动从 `ip` / `ifconfig` 探测。
- PID 文件存放在 `.pid/` 目录，`stop` 时通过 PID 文件终止进程树；文件不存在时按进程名兜底查杀。
- 启动顺序：prefill 实例全部就绪后，再启动 decode 实例，最后启动代理。

---

## `pd_service_server.py`

**作用**：HTTP 控制服务器，提供 REST API 远程控制 PD 集群，适用于多机协同或自动化运维场景。

**启动**

```bash
python pd_service_server.py --config configs/xxx.yaml [--host 0.0.0.0] [--port 8088]
```

**API**

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/start` | 后台启动全部服务，立即返回 `task_id` |
| `POST` | `/stop` | 后台停止全部服务 |
| `GET` | `/status` | 各实例存活状态 + NPU HBM 用量 |
| `GET` | `/task` | 最新任务进度和日志 |
| `GET` | `/task/<id>` | 指定任务进度和完整日志 |

**示例**

```bash
# 启动服务（必须传入 log_dir）
curl -X POST http://host:8088/start \
  -H 'Content-Type: application/json' \
  -d '{"log_dir": "/path/to/logs"}'

# 停止服务
curl -X POST http://host:8088/stop

# 查询服务状态
curl http://host:8088/status

# 轮询任务进度（start 是长时操作）
curl http://host:8088/task
curl http://host:8088/task/a1b2c3d4
```

**`POST /start` 请求体参数**

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `log_dir` | **是** | 日志目录 | — |
| `no_wait` | 否 | 不等待 `/health` 就绪 | `false` |

**`GET /status` 响应示例**

```json
{
  "busy": false,
  "alive_instances": 3,
  "total_instances": 3,
  "current_task": null,
  "instances": [
    {"name": "P0", "role": "prefill", "port": 9000, "devices": "0,1", "pid": 12345, "alive": true},
    {"name": "D0", "role": "decode",  "port": 9010, "devices": "2,3", "pid": 12346, "alive": true},
    {"name": "proxy", "role": "proxy", "port": 8000, "devices": null,  "pid": 12347, "alive": true}
  ],
  "npu_hbm_mb": {"npu0": 3435, "npu1": 3431, "npu2": 12000, "npu3": 11980}
}
```

**设计说明**

- start 等长时操作（分钟级）在后台线程执行，202 立即返回，通过 `GET /task` 轮询进度
- 同一时间只允许一个操作运行；并发请求返回 `409 Conflict` + 当前任务快照
- 任务日志最多保留 2000 行，历史任务保留最近 10 条
- 纯标准库实现，无额外依赖

---

## `pd_proxy.py`

内置负载均衡代理（基于 vLLM `disagg_proxy_demo.py` 精简），支持多 P/D 实例轮询调度。

当配置文件中定义了 `proxy` 段时，`pd_service_ctl.py` 会在启动时自动管理代理的生命周期。也可通过以下命令独立管理：

```bash
# 启动代理（前台运行，自动写入 .pid/proxy.pid）
python pd_proxy.py start --config configs/qwen3_32b_1p2_1d2.yaml

# 停止代理（读取 .pid/proxy.pid 终止进程）
python pd_proxy.py stop
```

**Prefill-only 模式（压测 prefill 性能）**

有两种方式开启，命令行标志优先级高于配置文件：

```bash
# 方式一：命令行标志（无需修改配置文件）
python pd_proxy.py start --config configs/qwen3_32b_1p2_1d2.yaml --prefill-only
python pd_proxy.py start --config configs/qwen3_32b_1p2_1d2.yaml --no-prefill-only  # 强制关闭

# 方式二：配置文件
```
```yaml
proxy:
  port: 8000
  prefill_only: true
```

效果：
- Prefill 节点**正常执行**（`max_tokens=1`，生成 KV cache）
- Decode 阶段**不转发任何请求**，立即返回空响应（`finish_reason: stop`，content 为空）
- 支持 streaming / non-streaming 两种请求格式
- `GET /status` 返回 `"prefill_only": true` 可供确认

**API 端点**

| 端点 | 说明 |
|------|------|
| `POST /v1/completions` | 文本补全（两阶段转发） |
| `POST /v1/chat/completions` | 对话补全（两阶段转发） |
| `GET /v1/models` | 返回代理配置的模型名 |
| `POST /v1/release_kv_cache` | 广播到所有 prefill + decode 实例 |
| `GET /status` | 集群状态（实例列表、prefill_only 标志） |
| `GET /health` | 健康检查 |

---

## `analysis/` 分析脚本

均为 **`analysis/`** 下的 Jupyter Notebook。

| 文件 | 功能概要 |
|------|----------|
| **`gen_hit_rate_v5.ipynb`** | KV cache 命中率分析（v5） |
| **`gen_hit_rate_v6.ipynb`** | KV cache 命中率分析（v6） |
