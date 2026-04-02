#!/usr/bin/env python3
"""
pd_service_server.py — PD 服务 HTTP 控制服务器

通过 REST API 远程控制 pd_service_ctl.py 的启动、停止操作，
并提供服务状态和 NPU 显存查询。

用法:
    python pd_service_server.py --config configs/xxx.yaml [--host 0.0.0.0] [--port 8088]

API:
    POST /start              启动全部服务（body 中必须传入 log_dir）
    POST /stop               停止全部服务
    GET  /status             查询实例运行状态 + NPU HBM 用量
    GET  /task               查询最新任务进度和日志
    GET  /task/<id>          查询指定任务进度和日志

所有 POST 均返回 202 Accepted + {"task_id": "...", "op": "...", "state": "running"}。
长时任务在后台线程执行；若当前已有任务运行，返回 409 Conflict。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# 嵌入式前端 HTML
# ---------------------------------------------------------------------------

_FRONTEND_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PD 服务控制台</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 18px; font-weight: 600; color: #90cdf4; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #68d391; animation: pulse 2s infinite; }
  .dot.offline { background: #fc8181; animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .container { display: grid; grid-template-columns: 340px 1fr; gap: 16px; padding: 16px 24px; max-width: 1400px; }
  .card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 13px; font-weight: 600; color: #a0aec0; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 12px; }
  label { display: block; font-size: 13px; color: #a0aec0; margin-bottom: 4px; margin-top: 10px; }
  select, input[type=text] {
    width: 100%; padding: 8px 10px; background: #0f1117; border: 1px solid #4a5568;
    border-radius: 6px; color: #e2e8f0; font-size: 13px; outline: none;
  }
  select:focus, input[type=text]:focus { border-color: #90cdf4; }
  .btn-row { display: flex; gap: 8px; margin-top: 14px; }
  button {
    flex: 1; padding: 9px 0; border: none; border-radius: 6px; font-size: 13px;
    font-weight: 600; cursor: pointer; transition: opacity .15s;
  }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .btn-start { background: #276749; color: #c6f6d5; }
  .btn-start:hover:not(:disabled) { background: #2f855a; }
  .btn-stop  { background: #742a2a; color: #fed7d7; }
  .btn-stop:hover:not(:disabled)  { background: #9b2c2c; }
  .instances { margin-top: 4px; }
  .inst-row { display: flex; align-items: center; gap: 8px; padding: 7px 0; border-bottom: 1px solid #2d3748; font-size: 13px; }
  .inst-row:last-child { border-bottom: none; }
  .badge { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .badge.alive { background: #68d391; }
  .badge.dead  { background: #fc8181; }
  .inst-name { font-weight: 600; width: 42px; }
  .inst-role { color: #a0aec0; width: 52px; font-size: 11px; }
  .inst-port { color: #90cdf4; width: 50px; }
  .inst-dev  { color: #b794f4; font-size: 11px; flex: 1; }
  .npu-row { display: flex; justify-content: space-between; padding: 5px 0; font-size: 12px; border-bottom: 1px solid #2d3748; }
  .npu-row:last-child { border-bottom: none; }
  .npu-key { color: #a0aec0; }
  .npu-val { color: #fbd38d; font-family: monospace; }
  .task-state { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 8px; }
  .state-running { background:#2b4c7e; color:#90cdf4; }
  .state-done    { background:#22543d; color:#c6f6d5; }
  .state-failed  { background:#742a2a; color:#fed7d7; }
  .log-box {
    margin-top: 10px; background: #0f1117; border: 1px solid #2d3748; border-radius: 6px;
    padding: 10px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 11.5px;
    line-height: 1.55; color: #a0aec0; height: 460px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;
  }
  .log-box .ts { color: #4a5568; }
  .toast { position:fixed; top:20px; right:20px; padding:10px 16px; border-radius:6px; font-size:13px; z-index:999; opacity:0; transition:opacity .3s; pointer-events:none; }
  .toast.show { opacity:1; }
  .toast.ok  { background:#22543d; color:#c6f6d5; }
  .toast.err { background:#742a2a; color:#fed7d7; }
  .status-bar { font-size: 12px; color: #718096; margin-top: 8px; }
  .right-col { display: flex; flex-direction: column; gap: 16px; }
</style>
</head>
<body>
<div class="header">
  <div class="dot" id="conn-dot"></div>
  <h1>PD 服务控制台</h1>
  <span id="cluster-name" style="color:#718096;font-size:13px;margin-left:8px;"></span>
</div>
<div class="container">
  <!-- 左栏：控制 -->
  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <h2>启动控制</h2>
      <label>配置文件</label>
      <select id="cfg-select"></select>
      <label>日志目录</label>
      <input type="text" id="log-dir" value="logs/" placeholder="logs/">
      <div class="btn-row">
        <button class="btn-start" id="btn-start" onclick="doStart()">启 动</button>
        <button class="btn-stop"  id="btn-stop"  onclick="doStop()">停 止</button>
      </div>
      <p class="status-bar" id="busy-status"></p>
    </div>
    <div class="card">
      <h2>实例状态 <span id="inst-summary" style="font-weight:400;color:#718096;font-size:12px;text-transform:none;letter-spacing:0;"></span></h2>
      <div class="instances" id="inst-list"><span style="color:#4a5568;font-size:12px;">—</span></div>
    </div>
    <div class="card">
      <h2>NPU HBM 用量</h2>
      <div id="npu-list"><span style="color:#4a5568;font-size:12px;">—</span></div>
    </div>
  </div>
  <!-- 右栏：任务日志 -->
  <div class="right-col">
    <div class="card" style="flex:1;">
      <h2>
        任务日志
        <span id="task-id-badge" style="color:#718096;font-weight:400;font-size:11px;text-transform:none;letter-spacing:0;margin-left:4px;"></span>
        <span id="task-state-badge" class="task-state" style="display:none;"></span>
        <span style="float:right;">
          <label style="display:inline;margin:0;cursor:pointer;">
            <input type="checkbox" id="auto-scroll" checked style="width:auto;margin-right:4px;">
            <span style="font-size:11px;color:#718096;font-weight:400;text-transform:none;letter-spacing:0;">自动滚动</span>
          </label>
        </span>
      </h2>
      <div class="log-box" id="log-box"></div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const API = '';  // 同源
let _taskId = null;
let _polling = false;
let _statusTimer = null;
let _taskTimer = null;

// ─── 初始化 ───
async function init() {
  await loadConfigs();
  refreshStatus();
  refreshTask();
  _statusTimer = setInterval(refreshStatus, 3000);
  _taskTimer   = setInterval(refreshTask,   2000);
}

async function loadConfigs() {
  try {
    const r = await fetch(API + '/configs');
    const data = await r.json();
    const sel = document.getElementById('cfg-select');
    sel.innerHTML = '';
    for (const c of data.configs) {
      const o = document.createElement('option');
      o.value = c.path;
      o.textContent = c.name;
      if (c.active) o.selected = true;
      sel.appendChild(o);
    }
  } catch(e) { showToast('加载配置列表失败: ' + e, 'err'); }
}

// ─── 状态刷新 ───
async function refreshStatus() {
  try {
    const r = await fetch(API + '/status');
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('conn-dot').className = 'dot';
    renderInstances(d.instances || []);
    renderNpu(d.npu_hbm_mb);
    document.getElementById('busy-status').textContent =
      d.busy ? '⏳ 正在执行操作…' : '';
    if (d.busy && d.current_task) {
      _taskId = d.current_task.task_id;
    }
    const n = d.cluster_name || '';
    document.getElementById('cluster-name').textContent = n ? '— ' + n : '';
    document.getElementById('btn-start').disabled = d.busy;
    document.getElementById('btn-stop').disabled  = d.busy;
  } catch(e) {
    document.getElementById('conn-dot').className = 'dot offline';
  }
}

function renderInstances(instances) {
  const el = document.getElementById('inst-list');
  if (!instances.length) { el.innerHTML = '<span style="color:#4a5568;font-size:12px;">无实例</span>'; return; }
  const alive = instances.filter(i => i.alive).length;
  document.getElementById('inst-summary').textContent = alive + '/' + instances.length + ' 在线';
  el.innerHTML = instances.map(i => `
    <div class="inst-row">
      <div class="badge ${i.alive ? 'alive' : 'dead'}"></div>
      <div class="inst-name">${i.name}</div>
      <div class="inst-role">${i.role}</div>
      <div class="inst-port">:${i.port}</div>
      <div class="inst-dev">${i.devices || ''}</div>
    </div>`).join('');
}

function renderNpu(npu) {
  const el = document.getElementById('npu-list');
  if (!npu) { el.innerHTML = '<span style="color:#4a5568;font-size:12px;">—</span>'; return; }
  el.innerHTML = Object.entries(npu).map(([k,v]) =>
    `<div class="npu-row"><span class="npu-key">${k}</span><span class="npu-val">${v.toLocaleString()} MB</span></div>`
  ).join('');
}

// ─── 任务日志刷新 ───
async function refreshTask() {
  try {
    const url = _taskId ? API + '/task/' + _taskId : API + '/task';
    const r = await fetch(url);
    if (!r.ok) return;
    const d = await r.json();
    renderTaskLogs(d);
    if (d.state !== 'running') _taskId = d.task_id;
  } catch(e) {}
}

function renderTaskLogs(task) {
  const badge = document.getElementById('task-state-badge');
  const idBadge = document.getElementById('task-id-badge');
  badge.style.display = 'inline';
  badge.className = 'task-state state-' + task.state;
  badge.textContent = { running:'运行中', done:'完成', failed:'失败' }[task.state] || task.state;
  const elapsed = task.elapsed_s != null ? ` (${task.elapsed_s}s)` : '';
  idBadge.textContent = task.op + ' #' + task.task_id + elapsed;

  const box = document.getElementById('log-box');
  const logs = task.logs || [];
  box.innerHTML = logs.map(l => {
    const m = l.match(/^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) (.*)$/s);
    if (m) return `<span class="ts">${m[1]}</span> ${escHtml(m[2])}`;
    return escHtml(l);
  }).join('\\n');

  if (document.getElementById('auto-scroll').checked) {
    box.scrollTop = box.scrollHeight;
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── 操作 ───
async function doStart() {
  const cfg = document.getElementById('cfg-select').value;
  const logDir = document.getElementById('log-dir').value.trim() || 'logs/';
  const body = { log_dir: logDir };
  if (cfg) body.config = cfg;
  try {
    const r = await fetch(API + '/start', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const d = await r.json();
    if (r.status === 202) {
      _taskId = d.task_id;
      showToast('启动任务已提交 #' + d.task_id, 'ok');
    } else if (r.status === 409) {
      showToast('当前已有操作运行中', 'err');
    } else {
      showToast('启动失败: ' + (d.error || r.status), 'err');
    }
  } catch(e) { showToast('请求失败: ' + e, 'err'); }
}

async function doStop() {
  try {
    const r = await fetch(API + '/stop', { method: 'POST' });
    const d = await r.json();
    if (r.status === 202) {
      _taskId = d.task_id;
      showToast('停止任务已提交 #' + d.task_id, 'ok');
    } else if (r.status === 409) {
      showToast('当前已有操作运行中', 'err');
    } else {
      showToast('停止失败: ' + (d.error || r.status), 'err');
    }
  } catch(e) { showToast('请求失败: ' + e, 'err'); }
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast ' + type + ' show';
  setTimeout(() => t.className = 'toast', 3000);
}

init();
</script>
</body>
</html>
"""

from pd_service_ctl import (
    ClusterConfig,
    PdServiceCtl,
    _get_npu_hbm_usage,
    _pid_alive,
    _pid_file,
    load_config,
    log_default,
)

# ---------------------------------------------------------------------------
# 任务模型
# ---------------------------------------------------------------------------

_MAX_LOG_LINES = 2000
_TASK_HISTORY_SIZE = 10


class Task:
    """单次操作的执行状态和日志容器。"""

    def __init__(self, task_id: str, op: str) -> None:
        self.task_id = task_id
        self.op = op
        self.state = "running"          # running | done | failed
        self.rc: Optional[int] = None
        self.logs: Deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{ts} {msg}")
        log_default(msg)

    def finish(self, rc: int) -> None:
        self.rc = rc
        self.state = "done" if rc == 0 else "failed"
        self.end_time = time.time()

    def to_dict(self, *, tail: Optional[int] = None) -> Dict[str, Any]:
        """序列化为字典。``tail`` 为 None 时返回全量日志，否则返回最后 N 行。"""
        logs = list(self.logs)
        if tail is not None:
            logs = logs[-tail:]
        return {
            "task_id": self.task_id,
            "op": self.op,
            "state": self.state,
            "rc": self.rc,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3) if self.end_time else None,
            "elapsed_s": round((self.end_time or time.time()) - self.start_time, 1),
            "logs": logs,
        }


# ---------------------------------------------------------------------------
# 服务全局状态
# ---------------------------------------------------------------------------


class ServiceState:
    """
    持有集群配置、后台任务状态、历史记录。

    同一时间只允许一个操作运行（start/stop/restart），通过 ``_op_lock`` 串行化。
    """

    def __init__(self, cfg: ClusterConfig) -> None:
        self.cfg = cfg
        self._op_lock = threading.Lock()
        self._current_task: Optional[Task] = None
        self._history: Deque[Task] = deque(maxlen=_TASK_HISTORY_SIZE)
        self.log_dir: Optional[Path] = None  # 最近一次 start 使用的日志目录

    # ---------- 查询 ----------

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    def is_busy(self) -> bool:
        t = self._current_task
        return t is not None and t.state == "running"

    def get_task(self, task_id: Optional[str] = None) -> Optional[Task]:
        """返回指定 ID 的任务；task_id 为 None 时返回最新任务（运行中优先）。"""
        if task_id is None:
            if self._current_task:
                return self._current_task
            return self._history[0] if self._history else None
        if self._current_task and self._current_task.task_id == task_id:
            return self._current_task
        for t in self._history:
            if t.task_id == task_id:
                return t
        return None

    # ---------- 提交 ----------

    def submit(self, op: str, fn) -> Optional[Task]:
        """
        提交后台任务。fn 签名：``fn(task: Task) -> int``。

        若当前已有任务运行，返回 None（拒绝）；否则返回新建的 Task。
        """
        with self._op_lock:
            if self.is_busy():
                return None
            task = Task(uuid.uuid4().hex[:8], op)
            self._current_task = task

        thread = threading.Thread(target=self._run, args=(task, fn), daemon=True)
        thread.start()
        return task

    def _run(self, task: Task, fn) -> None:
        try:
            rc = fn(task)
            task.finish(rc if rc is not None else 0)
        except Exception as exc:
            task.log(f"ERROR: 操作异常: {exc}")
            task.finish(1)
        finally:
            self._history.appendleft(task)
            with self._op_lock:
                if self._current_task is task:
                    self._current_task = None

    # ---------- 状态查询 ----------

    def instance_status(self) -> List[Dict[str, Any]]:
        """返回各实例的存活状态（基于 PID 文件）。"""
        result: List[Dict[str, Any]] = []
        cname = self.cfg.cluster_name
        for inst in self.cfg.prefill_instances + self.cfg.decode_instances:
            pid, alive = _read_pid_alive(_pid_file(cname, inst.name))
            result.append({
                "name": inst.name,
                "role": inst.role,
                "port": inst.port,
                "devices": inst.devices,
                "pid": pid,
                "alive": alive,
            })
        if self.cfg.proxy_port is not None:
            pid, alive = _read_pid_alive(_pid_file(cname, "proxy"))
            result.append({
                "name": "proxy",
                "role": "proxy",
                "port": self.cfg.proxy_port,
                "devices": None,
                "pid": pid,
                "alive": alive,
            })
        return result

    def npu_hbm_status(self) -> Optional[Dict[str, int]]:
        """返回 NPU HBM 用量（MB）字典，key 为 'npu{index}'。"""
        hbm = _get_npu_hbm_usage(lambda _: None)
        if hbm is None:
            return None
        return {f"npu{k}": v for k, v in sorted(hbm.items())}


def _read_pid_alive(pid_file: Path):
    """读取 PID 文件并判断进程是否存活，返回 (pid, alive)。"""
    if not pid_file.is_file():
        return None, False
    try:
        pid = int(pid_file.read_text().strip())
        return pid, _pid_alive(pid)
    except (ValueError, OSError):
        return None, False


# ---------------------------------------------------------------------------
# HTTP 请求处理
# ---------------------------------------------------------------------------

_state: Optional[ServiceState] = None
_configs_dir: Optional[Path] = None   # configs/ 目录，由 main() 初始化


def _add_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _tail_file(path: Path, n: int) -> List[str]:
    """高效读取文件末尾 n 行（从文件尾部向前扫描）。"""
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        with open(path, "rb") as f:
            # 对于小文件直接全部读取
            if size <= 256 * 1024:
                lines = f.read().decode(errors="replace").splitlines()
                return lines[-n:]
            # 大文件从尾部分块读取
            buf = b""
            chunk = min(size, 64 * 1024)
            pos = size
            while pos > 0 and buf.count(b"\n") <= n:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
            lines = buf.decode(errors="replace").splitlines()
            return lines[-n:]
    except Exception:
        return []


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: Any) -> None:
    data = json.dumps(body, ensure_ascii=False, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    _add_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw) or {}
    except json.JSONDecodeError:
        return {}


class PdControlHandler(BaseHTTPRequestHandler):
    """HTTP 请求路由。"""

    def log_message(self, fmt, *args):
        log_default(f"HTTP [{self.address_string()}] {fmt % args}")

    # ---------- OPTIONS (CORS preflight) ----------

    def do_OPTIONS(self):
        self.send_response(204)
        _add_cors_headers(self)
        self.end_headers()

    # ---------- GET ----------

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/status"):
            self._status()
        elif path in ("/ui", "/ui/"):
            self._ui()
        elif path == "/configs":
            self._configs()
        elif path == "/task" or path.startswith("/task/"):
            self._get_task(path)
        elif path == "/logs" or path.startswith("/logs/"):
            self._get_logs(path)
        else:
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def _ui(self):
        data = _FRONTEND_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _configs(self):
        result = []
        active_cfg = _state.cfg
        if _configs_dir and _configs_dir.is_dir():
            for p in sorted(_configs_dir.glob("*.yaml")):
                result.append({
                    "name": p.stem,
                    "path": str(p),
                    "active": (p.resolve() == Path(active_cfg.config_path).resolve()
                               if hasattr(active_cfg, "config_path") else False),
                })
        _json_response(self, 200, {"configs": result})

    def _status(self):
        instances = _state.instance_status()
        npu = _state.npu_hbm_status()
        alive_count = sum(1 for i in instances if i["alive"])
        task = _state.current_task
        _json_response(self, 200, {
            "busy": _state.is_busy(),
            "cluster_name": _state.cfg.cluster_name,
            "alive_instances": alive_count,
            "total_instances": len(instances),
            "current_task": task.to_dict(tail=20) if task else None,
            "instances": instances,
            "npu_hbm_mb": npu,
        })

    def _get_task(self, path: str):
        parts = [p for p in path.split("/") if p]
        task_id = parts[1] if len(parts) >= 2 else None
        task = _state.get_task(task_id)
        if task is None:
            _json_response(self, 404, {"error": "task not found"})
            return
        _json_response(self, 200, task.to_dict())

    def _get_logs(self, path: str):
        """返回指定实例的日志文件内容（尾部 N 行）。

        GET /logs          — 返回可用日志文件列表
        GET /logs/<name>   — 返回指定实例日志（支持 ?tail=500）
        """
        log_dir = _state.log_dir
        if log_dir is None or not log_dir.is_dir():
            _json_response(self, 404, {"error": "日志目录不存在，请先启动服务"})
            return

        parts = [p for p in path.split("/") if p]  # ["logs"] or ["logs", "<name>"]
        if len(parts) < 2:
            # 列出可用日志文件
            files = []
            for f in sorted(log_dir.glob("*.log")):
                files.append({"name": f.stem, "file": f.name, "size": f.stat().st_size})
            _json_response(self, 200, {"log_dir": str(log_dir), "files": files})
            return

        name = parts[1]
        # 解析 ?tail=N
        tail = 200
        qs = self.path.split("?", 1)
        if len(qs) == 2:
            for param in qs[1].split("&"):
                if param.startswith("tail="):
                    try:
                        tail = max(1, min(int(param[5:]), 5000))
                    except ValueError:
                        pass

        log_file = log_dir / f"{name}.log"
        if not log_file.is_file():
            _json_response(self, 404, {"error": f"日志文件不存在: {name}.log"})
            return

        try:
            lines = _tail_file(log_file, tail)
        except Exception as e:
            _json_response(self, 500, {"error": f"读取日志失败: {e}"})
            return

        _json_response(self, 200, {"name": name, "lines": lines, "tail": tail})

    # ---------- POST ----------

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/start":
            self._start()
        elif path == "/stop":
            self._stop()
        else:
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def _start(self):
        body = _read_json_body(self)
        if "log_dir" not in body:
            _json_response(self, 400, {"error": "missing required field: log_dir"})
            return
        no_wait = bool(body.get("no_wait", False))
        log_dir = Path(body["log_dir"])

        # 支持通过请求体切换配置文件
        cfg_path_str = body.get("config")
        if cfg_path_str:
            try:
                new_cfg = load_config(Path(cfg_path_str))
                _state.cfg = new_cfg
            except Exception as exc:
                _json_response(self, 400, {"error": f"加载配置失败: {exc}"})
                return

        _state.log_dir = log_dir.resolve()

        def fn(task: Task) -> int:
            return PdServiceCtl(_state.cfg, log=task.log).start_stack(
                log_dir, wait_ready=not no_wait
            )

        self._submit("start", fn)

    def _stop(self):
        def fn(task: Task) -> int:
            PdServiceCtl(_state.cfg, log=task.log).stop()
            return 0

        self._submit("stop", fn)

    def _submit(self, op: str, fn) -> None:
        task = _state.submit(op, fn)
        if task is None:
            current = _state.current_task
            _json_response(self, 409, {
                "error": "busy — another operation is already running",
                "current_task": current.to_dict(tail=5) if current else None,
            })
            return
        _json_response(self, 202, {
            "task_id": task.task_id,
            "op": op,
            "state": "running",
        })


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    global _state, _configs_dir

    parser = argparse.ArgumentParser(
        description="PD 服务 HTTP 控制服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8088, help="监听端口")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _state = ServiceState(cfg)
    _configs_dir = args.config.resolve().parent   # configs/ 目录

    log_default(f"PD 控制服务器启动  http://{args.host}:{args.port}")
    log_default(f"配置文件: {args.config.resolve()}")
    log_default(f"前端页面: http://{args.host}:{args.port}/ui")
    log_default("接口: POST /start  POST /stop  GET /status  GET /task[/<id>]  GET /configs  GET /ui")

    server = HTTPServer((args.host, args.port), PdControlHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_default("服务器停止。")


if __name__ == "__main__":
    main()
