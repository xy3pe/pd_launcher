#!/usr/bin/env python3
"""
pd_launcher.py — PD 推理服务 Web 启动器

提供 Web UI，支持：配置文件管理（新建/编辑/删除）、启停 pd_service_server、
通过代理控制 PD 集群启停并查看日志。

用法:
    python pd_launcher.py [--host 0.0.0.0] [--port 8080] [--configs configs/]

然后浏览器访问 http://<host>:<port>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

_configs_dir: Path = Path("configs")
_server_pid: Optional[int] = None      # pd_service_server 进程 PID
_server_port: Optional[int] = None
_server_config: Optional[str] = None
_lock = threading.Lock()

_http_server: Optional[HTTPServer] = None   # 当前 HTTPServer 实例（供热更新关闭）
_reload_requested = False                   # 热更新标志

_STATE_FILE = Path(__file__).resolve().parent / ".launcher_state.json"

# ---------------------------------------------------------------------------
# 默认配置模板
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = """\
cluster_name: "MyCluster"

model:
  path: "/root/autodl-tmp/models/ModelName"
  served_name: "model_name"

venv:
  vllm: "/root/autodl-tmp/py_venv/vllm2"

venv_activate: "vllm"  # 激活哪个 venv（对应 venv 段中的 key）

log_level: "INFO"

paths:
  transfer_engine_lib: "/usr/local/lib"
  python_lib: "/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"

network:
  nic_name: null
  local_ip: null

vllm_defaults:
  dtype: "bfloat16"
  max_model_len: 32768
  max_num_batched_tokens: 32768
  max_num_seqs: 64
  gpu_memory_utilization: 0.9
  enforce_eager: true
  trust_remote_code: true
  enable_auto_tool_choice: true
  tool_call_parser: "hermes"
  seed: 1024
  omp_num_threads: 10

prefill_defaults:
  enable_chunked_prefill: true
  max_num_batched_tokens: 32768
  long-prefill-token-threshold: 1024
  kv_transfer_config:
    kv_connector: "CPUOffloadingConnector"
    kv_role: "kv_producer"
    kv_connector_extra_config:
      num_cpu_blocks: 5000

decode_defaults:

prefill:
  - name: "P0"
    port: 7000
    devices: "0,1"
    tensor_parallel_size: 2
    dp_port: 13395
    hccl_buffsize: 256

decode:
  - name: "D0"
    port: 7010
    devices: "2,3"
    tensor_parallel_size: 2
    dp_port: 13495
    hccl_buffsize: 512

proxy:
  port: 9050
  prefill_only: true
"""

# ---------------------------------------------------------------------------
# 服务器管理（PID 模式，支持热更新后恢复）
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """检查 PID 对应的进程是否存活。"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _save_server_state() -> None:
    """将 pd_service_server 状态写入文件，供热更新后恢复。"""
    with _lock:
        state: Dict[str, Any] = {}
        if _server_pid is not None and _pid_alive(_server_pid):
            state = {"pid": _server_pid, "port": _server_port, "config": _server_config}
    try:
        _STATE_FILE.write_text(json.dumps(state))
    except OSError:
        pass


def _recover_server_state() -> None:
    """热更新启动时，从状态文件恢复 pd_service_server 引用。"""
    global _server_pid, _server_port, _server_config
    if not _STATE_FILE.is_file():
        return
    try:
        state = json.loads(_STATE_FILE.read_text())
        pid = state.get("pid")
        if pid and _pid_alive(pid):
            _server_pid = pid
            _server_port = state.get("port")
            _server_config = state.get("config")
            print(f"[热更新] 恢复控制服务器连接 (PID {pid}，端口 {_server_port})")
    except (json.JSONDecodeError, OSError, KeyError):
        pass


def _get_server_status() -> Dict[str, Any]:
    global _server_pid, _server_port, _server_config
    with _lock:
        if _server_pid is None:
            return {"running": False, "pid": None, "port": None, "config": None}
        if not _pid_alive(_server_pid):
            _server_pid = None
            _server_port = None
            _server_config = None
            return {"running": False, "pid": None, "port": None, "config": None}
        return {
            "running": True,
            "pid": _server_pid,
            "port": _server_port,
            "config": _server_config,
        }


def _start_pd_server(config_path: str, port: int) -> Tuple[bool, str]:
    global _server_pid, _server_port, _server_config
    with _lock:
        if _server_pid is not None and _pid_alive(_server_pid):
            return False, "控制服务器已在运行"
        script = Path(__file__).parent / "pd_service_server.py"
        if not script.is_file():
            return False, f"找不到 pd_service_server.py: {script}"
        cmd = [sys.executable, str(script), "--config", config_path, "--port", str(port)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
            _server_pid = proc.pid
            _server_port = port
            _server_config = config_path
            return True, f"控制服务器已启动 (PID {proc.pid}，端口 {port})"
        except Exception as e:
            return False, f"启动失败: {e}"


def _stop_pd_server() -> Tuple[bool, str]:
    global _server_pid, _server_port, _server_config
    with _lock:
        if _server_pid is None or not _pid_alive(_server_pid):
            _server_pid = None
            _server_port = None
            _server_config = None
            return False, "控制服务器未运行"
        pid = _server_pid
        try:
            os.kill(pid, signal.SIGTERM)
            # 等待退出
            for _ in range(50):
                time.sleep(0.1)
                if not _pid_alive(pid):
                    break
            else:
                os.kill(pid, signal.SIGKILL)
            _server_pid = None
            _server_port = None
            _server_config = None
            return True, f"控制服务器已停止 (PID {pid})"
        except Exception as e:
            return False, f"停止失败: {e}"


# ---------------------------------------------------------------------------
# 代理到 pd_service_server
# ---------------------------------------------------------------------------


def _proxy(method: str, pd_path: str, body: bytes, content_type: str) -> Tuple[int, bytes, str]:
    with _lock:
        port = _server_port
        alive = _server_pid is not None and _pid_alive(_server_pid)
    if not alive or port is None:
        return 503, json.dumps({"error": "控制服务器未运行"}).encode(), "application/json"
    url = f"http://127.0.0.1:{port}{pd_path}"
    req = urllib.request.Request(url, data=body or None, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ct = resp.headers.get("Content-Type", "application/json")
            return resp.status, resp.read(), ct
    except urllib.error.HTTPError as e:
        ct = e.headers.get("Content-Type", "application/json")
        return e.code, e.read(), ct
    except Exception as e:
        return 502, json.dumps({"error": str(e)}).encode(), "application/json"


# ---------------------------------------------------------------------------
# HTTP 处理器工具函数
# ---------------------------------------------------------------------------


def _json_resp(handler: BaseHTTPRequestHandler, code: int, body: Any) -> None:
    data = json.dumps(body, ensure_ascii=False).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def _read_body(handler: BaseHTTPRequestHandler) -> Tuple[bytes, str]:
    length = int(handler.headers.get("Content-Length", 0))
    ct = handler.headers.get("Content-Type", "")
    return (handler.rfile.read(length) if length > 0 else b""), ct


def _read_json(handler: BaseHTTPRequestHandler) -> Dict:
    body, _ = _read_body(handler)
    try:
        return json.loads(body) if body else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTTP 请求处理器
# ---------------------------------------------------------------------------


class LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/") or "/"
        if p == "/":
            self._html()
        elif p == "/api/configs":
            self._list_configs()
        elif p.startswith("/api/configs/"):
            self._get_config(p[len("/api/configs/"):])
        elif p == "/api/ucm_configs":
            self._list_ucm_configs()
        elif p.startswith("/api/ucm_configs/"):
            self._get_ucm_config(p[len("/api/ucm_configs/"):])
        elif p == "/api/template":
            _json_resp(self, 200, {"content": _DEFAULT_TEMPLATE})
        elif p == "/api/server":
            _json_resp(self, 200, _get_server_status())
        elif p.startswith("/pd"):
            self._proxy_req("GET", p[3:] or "/", b"", "")
        else:
            _json_resp(self, 404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")
        if p == "/api/configs":
            self._save_config()
        elif p == "/api/ucm_configs":
            self._save_ucm_config()
        elif p == "/api/server/start":
            self._server_start()
        elif p == "/api/server/stop":
            self._server_stop()
        elif p.startswith("/pd"):
            body, ct = _read_body(self)
            self._proxy_req("POST", p[3:] or "/", body, ct)
        else:
            _json_resp(self, 404, {"error": "not found"})

    def do_DELETE(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.startswith("/api/configs/"):
            self._delete_config(p[len("/api/configs/"):])
        elif p.startswith("/api/ucm_configs/"):
            self._delete_ucm_config(p[len("/api/ucm_configs/"):])
        else:
            _json_resp(self, 404, {"error": "not found"})

    # ── 前端页面 ──

    def _html(self):
        data = _FRONTEND_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── 配置文件 CRUD ──

    def _list_configs(self):
        configs = []
        if _configs_dir.is_dir():
            for p in sorted(_configs_dir.glob("*.yaml")):
                configs.append({"name": p.stem, "path": str(p)})
        _json_resp(self, 200, {"configs": configs})

    def _get_config(self, name: str):
        name = urllib.parse.unquote(name)
        stem = Path(name).stem
        p = _configs_dir / f"{stem}.yaml"
        if not p.is_file():
            _json_resp(self, 404, {"error": "config not found"}); return
        _json_resp(self, 200, {"name": stem, "path": str(p), "content": p.read_text()})

    def _save_config(self):
        d = _read_json(self)
        name = str(d.get("name", "")).strip()
        content = str(d.get("content", ""))
        if not name:
            _json_resp(self, 400, {"error": "name required"}); return
        stem = Path(name).stem or name.replace("/", "_")
        _configs_dir.mkdir(parents=True, exist_ok=True)
        p = _configs_dir / f"{stem}.yaml"
        p.write_text(content)
        _json_resp(self, 200, {"name": stem, "path": str(p)})

    def _delete_config(self, name: str):
        name = urllib.parse.unquote(name)
        stem = Path(name).stem
        p = _configs_dir / f"{stem}.yaml"
        if not p.is_file():
            _json_resp(self, 404, {"error": "config not found"}); return
        p.unlink()
        _json_resp(self, 200, {"deleted": stem})

    # ── UCM 配置文件 CRUD ──

    @staticmethod
    def _ucm_dir() -> Path:
        return _configs_dir / "ucm_config"

    def _list_ucm_configs(self):
        ucm_dir = self._ucm_dir()
        configs = []
        if ucm_dir.is_dir():
            for p in sorted(ucm_dir.glob("*.yaml")):
                configs.append({"name": p.stem, "path": str(p)})
        _json_resp(self, 200, {"configs": configs})

    def _get_ucm_config(self, name: str):
        name = urllib.parse.unquote(name)
        stem = Path(name).stem
        p = self._ucm_dir() / f"{stem}.yaml"
        if not p.is_file():
            _json_resp(self, 404, {"error": "ucm config not found"}); return
        _json_resp(self, 200, {"name": stem, "path": str(p), "content": p.read_text()})

    def _save_ucm_config(self):
        d = _read_json(self)
        name = str(d.get("name", "")).strip()
        content = str(d.get("content", ""))
        if not name:
            _json_resp(self, 400, {"error": "name required"}); return
        stem = Path(name).stem or name.replace("/", "_")
        ucm_dir = self._ucm_dir()
        ucm_dir.mkdir(parents=True, exist_ok=True)
        p = ucm_dir / f"{stem}.yaml"
        p.write_text(content)
        _json_resp(self, 200, {"name": stem, "path": str(p)})

    def _delete_ucm_config(self, name: str):
        name = urllib.parse.unquote(name)
        stem = Path(name).stem
        p = self._ucm_dir() / f"{stem}.yaml"
        if not p.is_file():
            _json_resp(self, 404, {"error": "ucm config not found"}); return
        p.unlink()
        _json_resp(self, 200, {"deleted": stem})

    # ── 控制服务器 ──

    def _server_start(self):
        d = _read_json(self)
        cfg = str(d.get("config", "")).strip()
        port = int(d.get("port", 8088))
        if not cfg:
            _json_resp(self, 400, {"error": "config required"}); return
        ok, msg = _start_pd_server(cfg, port)
        _json_resp(self, 200 if ok else 409, {"ok": ok, "message": msg})

    def _server_stop(self):
        ok, msg = _stop_pd_server()
        _json_resp(self, 200, {"ok": ok, "message": msg})

    # ── 代理 ──

    def _proxy_req(self, method: str, pd_path: str, body: bytes, ct: str):
        if not pd_path.startswith("/"):
            pd_path = "/" + pd_path
        code, resp_body, resp_ct = _proxy(method, pd_path, body, ct)
        self.send_response(code)
        self.send_header("Content-Type", resp_ct)
        self.send_header("Content-Length", str(len(resp_body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp_body)


# ---------------------------------------------------------------------------
# 嵌入式前端 HTML
# ---------------------------------------------------------------------------

_FRONTEND_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PD Launcher</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
/* theme variables */
:root{
--bg:#0f1117;--surface:#1a1d2e;--surface-dark:#151821;--surface-sel:#1e2235;
--border:#2d3748;--border-light:#4a5568;--border-inst:#1f2535;
--text-primary:#e2e8f0;--text-body:#a0aec0;--text-muted:#718096;--text-dim:#4a5568;
--accent:#90cdf4;--accent-hover:#63b3ed;--purple:#b794f4;--warning:#fbd38d;
--dot-off:#fc8181;--dot-on:#68d391;
--btn-p-bg:#2b4c7e;--btn-p-text:#90cdf4;--btn-p-hover:#2c5282;
--btn-d-bg:#742a2a;--btn-d-text:#fed7d7;--btn-d-hover:#9b2c2c;
--btn-g-bg:#276749;--btn-g-text:#c6f6d5;--btn-g-hover:#2f855a;
--btn-n-bg:#2d3748;--btn-n-text:#a0aec0;--btn-n-hover:#4a5568;
--pill-run-bg:#2b4c7e;--pill-run-text:#90cdf4;
--pill-done-bg:#22543d;--pill-done-text:#c6f6d5;
--pill-fail-bg:#742a2a;--pill-fail-text:#fed7d7;
--toast-ok-bg:#22543d;--toast-ok-text:#c6f6d5;
--toast-err-bg:#742a2a;--toast-err-text:#fed7d7;
}
[data-theme="light"]{
--bg:#f7f8fa;--surface:#ffffff;--surface-dark:#f0f1f4;--surface-sel:#e8edf5;
--border:#d8dde6;--border-light:#c4cbd6;--border-inst:#e8edf5;
--text-primary:#1a202c;--text-body:#4a5568;--text-muted:#718096;--text-dim:#a0aec0;
--accent:#2b6cb0;--accent-hover:#2c5282;--purple:#6b46c1;--warning:#b7791f;
--dot-off:#e53e3e;--dot-on:#38a169;
--btn-p-bg:#bee3f8;--btn-p-text:#2b6cb0;--btn-p-hover:#90cdf4;
--btn-d-bg:#fed7d7;--btn-d-text:#c53030;--btn-d-hover:#feb2b2;
--btn-g-bg:#c6f6d5;--btn-g-text:#276749;--btn-g-hover:#9ae6b4;
--btn-n-bg:#e2e8f0;--btn-n-text:#4a5568;--btn-n-hover:#cbd5e0;
--pill-run-bg:#bee3f8;--pill-run-text:#2b6cb0;
--pill-done-bg:#c6f6d5;--pill-done-text:#276749;
--pill-fail-bg:#fed7d7;--pill-fail-text:#c53030;
--toast-ok-bg:#c6f6d5;--toast-ok-text:#276749;
--toast-err-bg:#fed7d7;--toast-err-text:#c53030;
}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text-primary);height:100vh;display:flex;flex-direction:column;overflow:hidden}
/* header */
.hdr{background:var(--surface);border-bottom:1px solid var(--border);padding:0 20px;display:flex;align-items:center;height:46px;gap:12px;flex-shrink:0}
.hdr-title{font-size:15px;font-weight:700;color:var(--accent);letter-spacing:-.01em}
.tabs{display:flex;gap:2px;margin-left:8px}
.tab{padding:5px 14px;border-radius:4px;font-size:13px;cursor:pointer;color:var(--text-muted);border:none;background:transparent}
.tab.active{background:var(--btn-n-bg);color:var(--text-primary)}
.srv-badge{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dot-off);flex-shrink:0}
.dot.on{background:var(--dot-on);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* theme toggle */
.theme-toggle{background:var(--btn-n-bg);color:var(--text-muted);border:none;border-radius:4px;padding:4px 8px;font-size:14px;cursor:pointer;line-height:1;display:flex;align-items:center}
.theme-toggle:hover{background:var(--btn-n-hover);color:var(--text-primary)}
/* pages */
.page{flex:1;overflow:hidden;display:none}
.page.active{display:flex}
/* config tab */
#tab-config{gap:0}
.sidebar{width:220px;flex-shrink:0;background:var(--surface-dark);border-right:1px solid var(--border);display:flex;flex-direction:column}
.sidebar-hdr{padding:9px 12px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--text-dim);border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.cfg-list{flex:1;overflow-y:auto;padding:4px 0}
.cfg-item{padding:7px 12px;cursor:pointer;font-size:13px;color:var(--text-body);border-left:3px solid transparent;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cfg-item:hover{background:var(--surface)}
.cfg-item.sel{background:var(--surface-sel);color:var(--accent);border-left-color:var(--accent)}
.cfg-main{flex:1;display:flex;flex-direction:column;padding:14px;gap:9px;overflow:hidden}
.name-row{display:flex;align-items:center;gap:8px}
.name-lbl{font-size:12px;color:var(--text-muted);white-space:nowrap}
input.name-inp{flex:1;padding:6px 8px;background:var(--bg);border:1px solid var(--border-light);border-radius:4px;color:var(--text-primary);font-size:13px;outline:none}
textarea.editor{flex:1;width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:12px;color:var(--text-body);font-family:'Cascadia Code','Fira Code',monospace;font-size:12px;line-height:1.6;resize:none;outline:none}
textarea.editor:focus{border-color:var(--border-light)}
.btn-row{display:flex;gap:7px}
/* service tab */
#tab-service{gap:14px;padding:14px;overflow:hidden}
.svc-left{width:250px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;overflow-y:auto}
.svc-right{flex:1;display:flex;flex-direction:column;overflow:hidden}
.card{background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:12px}
.card-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--text-dim);margin-bottom:9px}
label{display:block;font-size:11px;color:var(--text-muted);margin-bottom:2px;margin-top:7px}
label:first-of-type{margin-top:0}
select,input[type=text],input[type=number]{width:100%;padding:5px 8px;background:var(--bg);border:1px solid var(--border-light);border-radius:4px;color:var(--text-primary);font-size:12px;outline:none}
.inst-item{display:flex;align-items:center;gap:7px;padding:5px 0;font-size:12px;border-bottom:1px solid var(--border-inst)}
.inst-item:last-child{border:none}
.bdot{width:6px;height:6px;border-radius:50%;background:var(--dot-off);flex-shrink:0}
.bdot.on{background:var(--dot-on)}
.iname{font-weight:600;width:38px}
.irole{color:var(--text-muted);width:46px;font-size:10px}
.iport{color:var(--accent);width:44px}
.idev{color:var(--purple);font-size:10px}
.npu-grid{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-top:6px}
.npu-cell{background:var(--bg);border-radius:3px;padding:5px 7px}
.nk{font-size:10px;color:var(--text-muted)}
.nv{font-size:11px;color:var(--warning);font-family:monospace}
.log-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:12px}
.log-hdr{display:flex;align-items:center;gap:7px;margin-bottom:8px;flex-shrink:0}
.log-hdr-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--text-dim)}
.log-box{flex:1;overflow-y:auto;font-family:'Cascadia Code','Fira Code',monospace;font-size:11.5px;line-height:1.55;color:var(--text-body);white-space:pre-wrap;word-break:break-all}
.log-ts{color:var(--text-dim)}
.pill{padding:1px 6px;border-radius:3px;font-size:10px;font-weight:700}
.pill-run{background:var(--pill-run-bg);color:var(--pill-run-text)}
.pill-done{background:var(--pill-done-bg);color:var(--pill-done-text)}
.pill-fail{background:var(--pill-fail-bg);color:var(--pill-fail-text)}
/* buttons */
button{padding:5px 11px;border:none;border-radius:4px;font-size:12px;font-weight:600;cursor:pointer;transition:opacity .15s}
button:disabled{opacity:.38;cursor:not-allowed}
.btn-p{background:var(--btn-p-bg);color:var(--btn-p-text)}
.btn-p:hover:not(:disabled){background:var(--btn-p-hover)}
.btn-d{background:var(--btn-d-bg);color:var(--btn-d-text)}
.btn-d:hover:not(:disabled){background:var(--btn-d-hover)}
.btn-g{background:var(--btn-g-bg);color:var(--btn-g-text)}
.btn-g:hover:not(:disabled){background:var(--btn-g-hover)}
.btn-n{background:var(--btn-n-bg);color:var(--btn-n-text)}
.btn-n:hover:not(:disabled){background:var(--btn-n-hover)}
.stext{font-size:11px;color:var(--text-muted);margin-top:6px}
/* logs tab */
#tab-logs{flex-direction:column;padding:14px;gap:0;overflow:hidden}
.log-tabs{display:flex;gap:2px;padding-bottom:10px;flex-shrink:0;overflow-x:auto;border-bottom:1px solid var(--border)}
.log-tab{padding:5px 12px;border-radius:4px 4px 0 0;font-size:12px;cursor:pointer;color:var(--text-muted);border:none;background:transparent;white-space:nowrap}
.log-tab.active{background:var(--surface);color:var(--accent);border:1px solid var(--border);border-bottom-color:var(--surface)}
.log-tab .ltdot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px;vertical-align:middle}
.log-tab .ltdot.on{background:var(--dot-on)}
.log-tab .ltdot.off{background:var(--dot-off)}
.log-toolbar{display:flex;align-items:center;gap:8px;padding:8px 0;flex-shrink:0}
.log-toolbar select{width:auto;padding:3px 6px;font-size:11px}
.log-content{flex:1;overflow-y:auto;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;font-family:'Cascadia Code','Fira Code',monospace;font-size:11.5px;line-height:1.55;color:var(--text-body);white-space:pre-wrap;word-break:break-all}
.log-content .log-ts{color:var(--text-dim)}
.toast{position:fixed;bottom:18px;right:18px;padding:9px 14px;border-radius:5px;font-size:12px;z-index:999;opacity:0;transition:opacity .25s;pointer-events:none}
.toast.show{opacity:1}
.toast.ok{background:var(--toast-ok-bg);color:var(--toast-ok-text)}
.toast.err{background:var(--toast-err-bg);color:var(--toast-err-text)}
</style>
</head>
<body>
<div class="hdr">
  <span class="hdr-title">PD Launcher</span>
  <div class="tabs">
    <button class="tab active" onclick="gotoTab('config')">配置管理</button>
    <button class="tab" onclick="gotoTab('service')">服务控制</button>
    <button class="tab" onclick="gotoTab('logs')">服务日志</button>
  </div>
  <div class="srv-badge">
    <div class="dot" id="srv-dot"></div>
    <span id="srv-hdr-lbl">控制服务器: 未运行</span>
  </div>
  <button class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" title="切换主题">&#127769;</button>
</div>

<!-- ══ Tab: 配置管理 ══ -->
<div class="page active" id="tab-config">
  <div class="sidebar">
    <div class="sidebar-hdr">
      集群配置
      <button class="btn-n" style="padding:2px 7px;font-size:10px" onclick="newCfg()">+ 新建</button>
    </div>
    <div class="cfg-list" id="cfg-list"></div>
    <div class="sidebar-hdr" style="border-top:1px solid var(--border)">
      UCM 配置
      <button class="btn-n" style="padding:2px 7px;font-size:10px" onclick="newUcmCfg()">+ 新建</button>
    </div>
    <div class="cfg-list" id="ucm-cfg-list"></div>
  </div>
  <div class="cfg-main">
    <div class="name-row">
      <span class="name-lbl">配置名称</span>
      <input class="name-inp" id="cfg-name" type="text" placeholder="my_cluster">
    </div>
    <textarea class="editor" id="editor" placeholder="选择左侧配置，或点击「+ 新建」创建…"></textarea>
    <div class="btn-row">
      <button class="btn-g" onclick="saveCfg()">保存</button>
      <button class="btn-n" onclick="saveAsCfg()">另存为</button>
      <button class="btn-d" id="btn-del" onclick="delCfg()" disabled>删除</button>
    </div>
  </div>
</div>

<!-- ══ Tab: 服务控制 ══ -->
<div class="page" id="tab-service">
  <div class="svc-left">
    <div class="card">
      <div class="card-title">控制服务器</div>
      <label>配置文件</label>
      <select id="svc-cfg"></select>
      <label>控制端口</label>
      <input type="number" id="svc-port" value="8088">
      <label>PD 日志目录</label>
      <input type="text" id="svc-logdir" value="logs/">
      <div class="btn-row" style="margin-top:9px">
        <button class="btn-g" id="btn-srv-start" onclick="srvStart()">启动服务器</button>
        <button class="btn-d" id="btn-srv-stop"  onclick="srvStop()" disabled>停止</button>
      </div>
      <div class="stext" id="srv-status-txt">未运行</div>
    </div>
    <div class="card">
      <div class="card-title">PD 实例 <span id="inst-sum" style="font-weight:400;color:#718096;font-size:10px;text-transform:none;letter-spacing:0"></span></div>
      <div id="inst-list"><span style="color:#4a5568;font-size:11px">—</span></div>
      <div class="btn-row" style="margin-top:9px">
        <button class="btn-g" id="btn-pd-start" onclick="pdStart()" disabled>启动 PD</button>
        <button class="btn-d" id="btn-pd-stop"  onclick="pdStop()"  disabled>停止 PD</button>
      </div>
    </div>
    <div class="card">
      <div class="card-title">NPU HBM</div>
      <div class="npu-grid" id="npu-grid"><span style="color:#4a5568;font-size:11px">—</span></div>
    </div>
  </div>
  <div class="svc-right">
    <div class="log-wrap">
      <div class="log-hdr">
        <span class="log-hdr-title">任务日志</span>
        <span id="task-op" style="color:#718096;font-size:10px"></span>
        <span id="task-pill" class="pill" style="display:none"></span>
        <label style="margin-left:auto;display:flex;align-items:center;gap:4px;font-size:10px;color:#718096;cursor:pointer;margin-top:0">
          <input type="checkbox" id="auto-scroll" checked style="width:auto">自动滚动
        </label>
      </div>
      <div class="log-box" id="log-box"></div>
    </div>
  </div>
</div>

<!-- ══ Tab: 服务日志 ══ -->
<div class="page" id="tab-logs">
  <div class="log-tabs" id="log-tabs">
    <span style="color:var(--text-dim);font-size:11px;padding:5px 8px">启动服务后查看日志</span>
  </div>
  <div class="log-toolbar">
    <label style="margin:0;font-size:11px;color:var(--text-muted)">显示行数</label>
    <select id="log-tail-n">
      <option value="100">100</option>
      <option value="200" selected>200</option>
      <option value="500">500</option>
      <option value="1000">1000</option>
      <option value="2000">2000</option>
    </select>
    <button class="btn-n" style="padding:3px 8px;font-size:11px" onclick="fetchActiveLog()">刷新</button>
    <label style="margin:0;display:flex;align-items:center;gap:4px;font-size:11px;color:var(--text-muted);cursor:pointer;margin-left:auto">
      <input type="checkbox" id="log-auto-refresh" checked style="width:auto">自动刷新
    </label>
    <label style="margin:0;display:flex;align-items:center;gap:4px;font-size:11px;color:var(--text-muted);cursor:pointer">
      <input type="checkbox" id="log-auto-scroll2" checked style="width:auto">自动滚动
    </label>
  </div>
  <div class="log-content" id="log-content"></div>
</div>

<div class="toast" id="toast"></div>
<script>
// ── theme ──
function applyTheme(theme){
  document.documentElement.setAttribute('data-theme',theme);
  document.getElementById('theme-toggle').innerHTML=theme==='light'?'&#9728;&#65039;':'&#127769;';
  localStorage.setItem('pd-theme',theme);
}
function toggleTheme(){
  const cur=document.documentElement.getAttribute('data-theme')||'dark';
  applyTheme(cur==='dark'?'light':'dark');
}
applyTheme(localStorage.getItem('pd-theme')||'dark');

// ── tabs ──
function gotoTab(name) {
  ['config','service','logs'].forEach((n,i) => {
    document.getElementById('tab-'+n).classList.toggle('active', n===name);
    document.querySelectorAll('.tab')[i].classList.toggle('active', n===name);
  });
  if (name==='service') syncSvcCfg();
  if (name==='logs') refreshLogTabs();
}

// ── toast ──
function toast(msg,type){
  const t=document.getElementById('toast');
  t.textContent=msg; t.className='toast '+type+' show';
  clearTimeout(t._t); t._t=setTimeout(()=>t.className='toast',3000);
}

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ════════════════════════════════════
// 配置管理
// ════════════════════════════════════
let _cfgs=[], _selCfg=null, _editMode='cluster';

async function loadCfgList(){
  try{
    const r=await fetch('/api/configs'); const d=await r.json();
    _cfgs=d.configs||[]; renderCfgList(); syncSvcCfg();
  }catch(e){}
}

function renderCfgList(){
  const el=document.getElementById('cfg-list');
  el.innerHTML=_cfgs.map(c=>`<div class="cfg-item${_selCfg===c.name?' sel':''}"
    onclick="loadCfg('${c.name}')" title="${c.path}">${c.name}</div>`).join('')
    ||'<div style="padding:10px 12px;color:#4a5568;font-size:11px">暂无配置</div>';
}

async function loadCfg(name){
  try{
    const r=await fetch('/api/configs/'+encodeURIComponent(name));
    const d=await r.json();
    _switchEditMode('cluster');
    _selCfg=name;
    document.getElementById('cfg-name').value=name;
    document.getElementById('editor').value=d.content;
    document.getElementById('btn-del').disabled=false;
    renderCfgList();
  }catch(e){toast('加载失败: '+e,'err');}
}

async function newCfg(){
  _switchEditMode('cluster');
  _selCfg=null;
  document.getElementById('cfg-name').value='';
  document.getElementById('editor').value='';
  document.getElementById('btn-del').disabled=true;
  renderCfgList();
  try{
    const r=await fetch('/api/template'); const d=await r.json();
    document.getElementById('editor').value=d.content;
  }catch(e){}
  document.getElementById('cfg-name').focus();
}

async function saveCfg(){
  const name=document.getElementById('cfg-name').value.trim();
  const content=document.getElementById('editor').value;
  if(!name){toast('请输入配置名称','err');return;}
  const api=_editMode==='ucm'?'/api/ucm_configs':'/api/configs';
  try{
    const r=await fetch(api,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,content})});
    const d=await r.json();
    if(r.ok){
      if(_editMode==='ucm'){_selUcmCfg=name;await loadUcmCfgList();}
      else{_selCfg=name;await loadCfgList();}
      toast('已保存: '+name,'ok');document.getElementById('btn-del').disabled=false;
    }
    else toast('保存失败: '+(d.error||r.status),'err');
  }catch(e){toast('保存失败: '+e,'err');}
}

async function saveAsCfg(){
  const orig=document.getElementById('cfg-name').value.trim();
  const n=prompt('另存为（新配置名）:',orig?(orig+'_copy'):'');
  if(!n)return;
  document.getElementById('cfg-name').value=n.replace(/\\.yaml$/,'');
  if(_editMode==='ucm')_selUcmCfg=null; else _selCfg=null;
  await saveCfg();
}

async function delCfg(){
  const sel=_editMode==='ucm'?_selUcmCfg:_selCfg;
  if(!sel)return;
  if(!confirm('确认删除配置 '+sel+' ?'))return;
  const api=_editMode==='ucm'?'/api/ucm_configs/':'/api/configs/';
  try{
    const r=await fetch(api+encodeURIComponent(sel),{method:'DELETE'});
    const d=await r.json();
    if(r.ok){toast('已删除','ok');
      if(_editMode==='ucm')_selUcmCfg=null; else _selCfg=null;
      document.getElementById('cfg-name').value='';
      document.getElementById('editor').value='';document.getElementById('btn-del').disabled=true;
      if(_editMode==='ucm')await loadUcmCfgList(); else await loadCfgList();}
    else toast('删除失败: '+(d.error||r.status),'err');
  }catch(e){toast('删除失败: '+e,'err');}
}

function syncSvcCfg(){
  const sel=document.getElementById('svc-cfg');
  const prev=sel.value;
  sel.innerHTML=_cfgs.map(c=>`<option value="${c.path}"${c.path===prev?' selected':''}>${c.name}</option>`).join('');
}

// ════════════════════════════════════
// UCM 配置管理
// ════════════════════════════════════
let _ucmCfgs=[], _selUcmCfg=null;

async function loadUcmCfgList(){
  try{
    const r=await fetch('/api/ucm_configs'); const d=await r.json();
    _ucmCfgs=d.configs||[]; renderUcmCfgList();
  }catch(e){}
}

function renderUcmCfgList(){
  const el=document.getElementById('ucm-cfg-list');
  el.innerHTML=_ucmCfgs.map(c=>`<div class="cfg-item${_selUcmCfg===c.name&&_editMode==='ucm'?' sel':''}"
    onclick="loadUcmCfg('${c.name}')" title="${c.path}">${c.name}</div>`).join('')
    ||'<div style="padding:10px 12px;color:var(--text-dim);font-size:11px">暂无 UCM 配置</div>';
}

function _switchEditMode(mode){
  _editMode=mode;
  if(mode==='ucm'){_selCfg=null;renderCfgList();}
  else{_selUcmCfg=null;renderUcmCfgList();}
}

async function loadUcmCfg(name){
  try{
    const r=await fetch('/api/ucm_configs/'+encodeURIComponent(name));
    const d=await r.json();
    _switchEditMode('ucm');
    _selUcmCfg=name;
    document.getElementById('cfg-name').value=name;
    document.getElementById('editor').value=d.content;
    document.getElementById('btn-del').disabled=false;
    renderUcmCfgList();
  }catch(e){toast('加载失败: '+e,'err');}
}

async function newUcmCfg(){
  _switchEditMode('ucm');
  _selUcmCfg=null;
  document.getElementById('cfg-name').value='';
  document.getElementById('editor').value='';
  document.getElementById('btn-del').disabled=true;
  renderUcmCfgList();
  document.getElementById('cfg-name').focus();
}

// ════════════════════════════════════
// 服务控制
// ════════════════════════════════════
let _srvRunning=false, _taskId=null;

async function pollServer(){
  try{
    const r=await fetch('/api/server'); const d=await r.json();
    _srvRunning=d.running;
    document.getElementById('srv-dot').className='dot'+(d.running?' on':'');
    document.getElementById('srv-hdr-lbl').textContent=d.running?('控制服务器: :'+d.port):'控制服务器: 未运行';
    document.getElementById('srv-status-txt').textContent=d.running?('运行中 PID '+d.pid+' — '+(d.config||'')):'未运行';
    document.getElementById('btn-srv-start').disabled=d.running;
    document.getElementById('btn-srv-stop').disabled=!d.running;
    document.getElementById('btn-pd-start').disabled=!d.running;
    document.getElementById('btn-pd-stop').disabled=!d.running;
    if(d.running)pollPD();
  }catch(e){document.getElementById('srv-dot').className='dot';}
}

async function pollPD(){
  if(!_srvRunning)return;
  try{
    const r=await fetch('/pd/status'); if(!r.ok)return;
    const d=await r.json();
    renderInsts(d.instances||[]);
    renderNpu(d.npu_hbm_mb);
    if(d.busy&&d.current_task)_taskId=d.current_task.task_id;
    const busy=d.busy;
    document.getElementById('btn-pd-start').disabled=busy;
    document.getElementById('btn-pd-stop').disabled=busy;
  }catch(e){}
}

async function pollTask(){
  if(!_srvRunning)return;
  try{
    const url=_taskId?('/pd/task/'+_taskId):'/pd/task';
    const r=await fetch(url); if(!r.ok)return;
    const d=await r.json(); renderTask(d);
  }catch(e){}
}

function renderInsts(insts){
  const el=document.getElementById('inst-list');
  const alive=insts.filter(i=>i.alive).length;
  document.getElementById('inst-sum').textContent=insts.length?(alive+'/'+insts.length+' 在线'):'';
  el.innerHTML=insts.map(i=>`<div class="inst-item">
    <div class="bdot${i.alive?' on':''}"></div>
    <div class="iname">${i.name}</div>
    <div class="irole">${i.role}</div>
    <div class="iport">:${i.port}</div>
    <div class="idev">${i.devices||''}</div>
  </div>`).join('')||'<span style="color:#4a5568;font-size:11px">—</span>';
}

function renderNpu(npu){
  const el=document.getElementById('npu-grid');
  if(!npu){el.innerHTML='<span style="color:#4a5568;font-size:11px">—</span>';return;}
  el.innerHTML=Object.entries(npu).map(([k,v])=>`<div class="npu-cell">
    <div class="nk">${k}</div><div class="nv">${v.toLocaleString()} MB</div></div>`).join('');
}

function renderTask(task){
  const pill=document.getElementById('task-pill');
  document.getElementById('task-op').textContent=task.op+' #'+task.task_id+(task.elapsed_s!=null?' ('+task.elapsed_s+'s)':'');
  pill.style.display='inline';
  pill.className='pill pill-'+({running:'run',done:'done',failed:'fail'}[task.state]||'run');
  pill.textContent={running:'运行中',done:'完成',failed:'失败'}[task.state]||task.state;
  const box=document.getElementById('log-box');
  box.innerHTML=(task.logs||[]).map(l=>{
    const i=l.indexOf(' ',11); // after "YYYY-MM-DD HH:MM:SS"
    if(i>0)return '<span class="log-ts">'+esc(l.substring(0,i))+'</span> '+esc(l.substring(i+1));
    return esc(l);
  }).join('\\n');
  if(document.getElementById('auto-scroll').checked)box.scrollTop=box.scrollHeight;
}

async function srvStart(){
  const cfg=document.getElementById('svc-cfg').value;
  const port=parseInt(document.getElementById('svc-port').value)||8088;
  if(!cfg){toast('请选择配置','err');return;}
  try{
    const r=await fetch('/api/server/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({config:cfg,port})});
    const d=await r.json();
    toast(d.message||(r.ok?'已启动':'失败'),r.ok?'ok':'err');
    await pollServer();
  }catch(e){toast('请求失败: '+e,'err');}
}

async function srvStop(){
  try{
    const r=await fetch('/api/server/stop',{method:'POST'});
    const d=await r.json(); toast(d.message||'已停止','ok');
    await pollServer();
  }catch(e){toast('请求失败: '+e,'err');}
}

async function pdStart(){
  const logDir=document.getElementById('svc-logdir').value.trim()||'logs/';
  try{
    const r=await fetch('/pd/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({log_dir:logDir})});
    const d=await r.json();
    if(r.status===202){_taskId=d.task_id;toast('启动任务已提交 #'+d.task_id,'ok');}
    else toast(d.error||'操作失败','err');
  }catch(e){toast('请求失败: '+e,'err');}
}

async function pdStop(){
  try{
    const r=await fetch('/pd/stop',{method:'POST'});
    const d=await r.json();
    if(r.status===202){_taskId=d.task_id;toast('停止任务已提交','ok');}
    else toast(d.error||'操作失败','err');
  }catch(e){toast('请求失败: '+e,'err');}
}

// ════════════════════════════════════
// 服务日志
// ════════════════════════════════════
let _logFiles=[], _activeLog=null, _logInstStatus={};

async function refreshLogTabs(){
  // 获取日志文件列表和实例状态
  try{
    const [lr, sr] = await Promise.all([
      fetch('/pd/logs').then(r=>r.ok?r.json():null).catch(()=>null),
      fetch('/pd/status').then(r=>r.ok?r.json():null).catch(()=>null)
    ]);
    if(lr && lr.files) _logFiles=lr.files;
    else _logFiles=[];
    // 构建 name→alive 映射
    _logInstStatus={};
    if(sr && sr.instances){
      sr.instances.forEach(i=>{
        const logName = i.role==='proxy'?'proxy':(i.role+'_'+i.name);
        _logInstStatus[logName]=i.alive;
      });
    }
  }catch(e){_logFiles=[];}
  renderLogTabs();
  if(_logFiles.length>0 && (!_activeLog || !_logFiles.find(f=>f.name===_activeLog))){
    _activeLog=_logFiles[0].name;
  }
  renderLogTabs();
  if(_activeLog) await fetchActiveLog();
}

function renderLogTabs(){
  const el=document.getElementById('log-tabs');
  if(_logFiles.length===0){
    el.innerHTML='<span style="color:var(--text-dim);font-size:11px;padding:5px 8px">启动服务后查看日志</span>';
    return;
  }
  el.innerHTML=_logFiles.map(f=>{
    const alive=_logInstStatus[f.name];
    const dotClass=alive===true?'on':(alive===false?'off':'');
    const sel=_activeLog===f.name?' active':'';
    return `<button class="log-tab${sel}" onclick="switchLog('${f.name}')"><span class="ltdot ${dotClass}"></span>${f.name}</button>`;
  }).join('');
}

function switchLog(name){
  _activeLog=name;
  renderLogTabs();
  fetchActiveLog();
}

async function fetchActiveLog(){
  if(!_activeLog)return;
  const tail=document.getElementById('log-tail-n').value;
  try{
    const r=await fetch('/pd/logs/'+encodeURIComponent(_activeLog)+'?tail='+tail);
    if(!r.ok){document.getElementById('log-content').textContent='加载失败';return;}
    const d=await r.json();
    const box=document.getElementById('log-content');
    box.innerHTML=(d.lines||[]).map(l=>{
      const i=l.indexOf(' ',11);
      if(i>0)return '<span class="log-ts">'+esc(l.substring(0,i))+'</span> '+esc(l.substring(i+1));
      return esc(l);
    }).join('\\n');
    if(document.getElementById('log-auto-scroll2').checked) box.scrollTop=box.scrollHeight;
  }catch(e){document.getElementById('log-content').textContent='请求失败: '+e;}
}

// 自动刷新日志
setInterval(()=>{
  if(!document.getElementById('tab-logs').classList.contains('active'))return;
  if(!document.getElementById('log-auto-refresh').checked)return;
  fetchActiveLog();
  // 同步实例状态（较低频率在 refreshLogTabs 中已做）
},3000);

// 每 10s 刷新日志文件列表和实例状态
setInterval(()=>{
  if(!document.getElementById('tab-logs').classList.contains('active'))return;
  refreshLogTabs();
},10000);

// ── 初始化 ──
async function init(){
  await loadCfgList();
  await loadUcmCfgList();
  await pollServer();
  setInterval(pollServer,3000);
  setInterval(()=>{pollPD();pollTask();},2000);
}
init();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 热更新
# ---------------------------------------------------------------------------


def _file_watcher(watch_dir: Path, interval: float = 1.0) -> None:
    """
    监控 watch_dir 下所有 .py 文件的修改时间。
    检测到变更后设置 _reload_requested 并关闭 HTTP 服务器，触发主线程重启。
    """
    global _reload_requested

    mtimes: Dict[str, float] = {}
    for f in watch_dir.glob("*.py"):
        try:
            mtimes[str(f)] = f.stat().st_mtime
        except OSError:
            pass

    while True:
        time.sleep(interval)
        changed: list[str] = []
        for f in watch_dir.glob("*.py"):
            key = str(f)
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if key not in mtimes or mt != mtimes[key]:
                changed.append(f.name)
                mtimes[key] = mt

        if changed:
            names = ", ".join(changed)
            print(f"\n[热更新] 检测到文件变更: {names}，重启中...")
            _reload_requested = True
            _save_server_state()
            if _http_server is not None:
                _http_server.shutdown()     # 使 serve_forever() 返回
            return


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    global _configs_dir, _http_server

    parser = argparse.ArgumentParser(
        description="PD 推理服务 Web 启动器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--configs", type=Path, default=Path("configs"),
                        dest="configs_dir", help="配置文件目录")
    parser.add_argument("--reload", action="store_true",
                        help="启用热更新：检测 .py 文件变更后自动重启")
    args = parser.parse_args()

    _configs_dir = args.configs_dir.resolve()

    # 热更新启动时尝试恢复 pd_service_server 连接
    _recover_server_state()

    reload_tag = " [热更新模式]" if args.reload else ""
    print(f"PD Launcher 启动  http://{args.host}:{args.port}{reload_tag}")
    print(f"配置目录: {_configs_dir}")
    print("在浏览器打开上述地址即可使用")

    HTTPServer.allow_reuse_address = True
    server = HTTPServer((args.host, args.port), LauncherHandler)
    _http_server = server

    # 启动文件监控线程
    if args.reload:
        watcher = threading.Thread(
            target=_file_watcher,
            args=(Path(__file__).resolve().parent,),
            daemon=True,
        )
        watcher.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n启动器停止。")
        _stop_pd_server()
        return

    # serve_forever 正常返回 → 由热更新触发
    if _reload_requested:
        server.server_close()
        os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
