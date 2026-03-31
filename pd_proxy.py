#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PD 分离负载均衡代理。

基于 vLLM disagg_proxy_demo.py 精简而来，支持：
  - 多 Prefill / Decode 实例轮询调度
  - /v1/models 返回代理列出的模型名
  - /v1/completions 与 /v1/chat/completions
  - /status 查看集群状态
  - /health 健康检查

用法：
  python pd_proxy.py --model <model_name> \
       --prefill localhost:9000 --decode localhost:9010 localhost:9011 \
       --port 8000
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import aiohttp
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger("pd_proxy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)


class Proxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy | None = None,
        prefill_only: bool = False,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances) if decode_instances else None
        self.model = model
        self.scheduling_policy = scheduling_policy or RoundRobinSchedulingPolicy()
        self.prefill_only = prefill_only
        self.router = APIRouter()
        self._setup_routes()
        if prefill_only:
            logger.info("Running in PREFILL-ONLY mode: decode stage is stubbed out")

    def _setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self._validate_json)]
        )(self.create_completion)
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self._validate_json)]
        )(self.create_chat_completion)
        self.router.get("/v1/models", response_class=JSONResponse)(self.list_models)
        self.router.post("/v1/release_kv_cache", response_class=JSONResponse)(self.release_kv_cache)
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.get("/health", response_class=JSONResponse)(self.health)

    async def _validate_json(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="Only application/json allowed")

    def _schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def _forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
            try:
                async with session.post(url=url, json=data, headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk in response.content.iter_chunked(1024):
                                yield chunk
                        else:
                            yield await response.read()
                    else:
                        error_content = await response.text()
                        logger.error("Request failed %s: %s", response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Upstream error {response.status}: {error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError: %s", e)
                raise HTTPException(status_code=502, detail="Bad Gateway") from e

    def _remove_instance(self, instance_type: str, instance: str):
        if instance_type == "decode" and instance in self.decode_instances:
            self.decode_instances.remove(instance)
            self.decode_cycler = itertools.cycle(self.decode_instances)
        elif instance_type == "prefill" and instance in self.prefill_instances:
            self.prefill_instances.remove(instance)
            self.prefill_cycler = itertools.cycle(self.prefill_instances)

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()
            prefill = self._schedule(self.prefill_cycler)

            if self.prefill_only:
                # prefill-only 模式：无 KV connector，直接将 prefill 响应返回客户端
                # max_tokens=1 确保 prefill 完成后立即返回，测量真实 TTFT
                kv_prepare = request.copy()
                kv_prepare["max_tokens"] = 1
                try:
                    generator = self._forward_request(
                        f"http://{prefill}/v1/completions", kv_prepare
                    )
                except HTTPException as e:
                    self._remove_instance("prefill", prefill)
                    raise e
                return StreamingResponse(generator)

            # 标准 PD 流程：prefill → KV transfer → decode
            kv_prepare = request.copy()
            kv_prepare["max_tokens"] = 1
            try:
                async for _ in self._forward_request(
                    f"http://{prefill}/v1/completions", kv_prepare
                ):
                    continue
            except HTTPException as e:
                self._remove_instance("prefill", prefill)
                raise e

            decode = self._schedule(self.decode_cycler)
            try:
                generator = self._forward_request(
                    f"http://{decode}/v1/completions", request
                )
            except HTTPException as e:
                self._remove_instance("decode", decode)
                raise e

            return StreamingResponse(generator)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error in create_completion")
            raise HTTPException(status_code=500, detail="Internal proxy error")

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()
            prefill = self._schedule(self.prefill_cycler)

            if self.prefill_only:
                kv_prepare = request.copy()
                kv_prepare["max_tokens"] = 1
                kv_prepare["max_completion_tokens"] = 1
                try:
                    generator = self._forward_request(
                        f"http://{prefill}/v1/chat/completions", kv_prepare
                    )
                except HTTPException as e:
                    self._remove_instance("prefill", prefill)
                    raise e
                return StreamingResponse(content=generator)

            # 标准 PD 流程：prefill → KV transfer → decode
            kv_prepare = request.copy()
            kv_prepare["max_tokens"] = 1
            if "max_completion_tokens" in kv_prepare:
                kv_prepare["max_completion_tokens"] = 1
            try:
                async for _ in self._forward_request(
                    f"http://{prefill}/v1/chat/completions", kv_prepare
                ):
                    continue
            except HTTPException as e:
                self._remove_instance("prefill", prefill)
                raise e

            decode = self._schedule(self.decode_cycler)
            try:
                generator = self._forward_request(
                    f"http://{decode}/v1/chat/completions", request
                )
            except HTTPException as e:
                self._remove_instance("decode", decode)
                raise e

            return StreamingResponse(content=generator)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error in create_chat_completion")
            raise HTTPException(status_code=500, detail="Internal proxy error")

    async def list_models(self):
        """兼容 OpenAI /v1/models 接口，返回代理配置的模型名。"""
        import time as _time
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model,
                    "object": "model",
                    "created": int(_time.time()),
                    "owned_by": "pd-proxy",
                }
            ],
        }

    async def release_kv_cache(self, raw_request: Request):
        """广播 POST /v1/release_kv_cache 到所有 prefill + decode 实例。"""
        try:
            body = await raw_request.json()
        except Exception:
            body = {}

        all_instances = self.prefill_instances + self.decode_instances
        results = {}
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            for inst in all_instances:
                url = f"http://{inst}/v1/release_kv_cache"
                try:
                    async with session.post(url, json=body) as resp:
                        results[inst] = {"status": resp.status, "body": await resp.text()}
                except aiohttp.ClientError as e:
                    logger.error("release_kv_cache to %s failed: %s", inst, e)
                    results[inst] = {"status": 502, "body": str(e)}

        failed = {k: v for k, v in results.items() if v["status"] >= 400}
        if failed:
            logger.warning("release_kv_cache partial failure: %s", failed)

        return {"total": len(all_instances), "success": len(all_instances) - len(failed), "details": results}

    async def get_status(self):
        return {
            "prefill_count": len(self.prefill_instances),
            "decode_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
            "prefill_only": self.prefill_only,
        }

    async def health(self):
        return {"status": "ok"}


def create_app(
    prefill_instances: list[str],
    decode_instances: list[str],
    model: str,
    prefill_only: bool = False,
) -> FastAPI:
    """创建 FastAPI app（供 pd_service_ctl 直接调用）。"""
    app = FastAPI(title="PD Proxy")
    proxy = Proxy(prefill_instances, decode_instances, model, prefill_only=prefill_only)
    app.include_router(proxy.router)
    return app


PKG_DIR = Path(__file__).resolve().parent
PID_DIR = PKG_DIR / ".pid"


def _pid_file() -> Path:
    PID_DIR.mkdir(exist_ok=True)
    return PID_DIR / "proxy.pid"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _collect_pid_tree(root: int) -> set[int]:
    import subprocess as _sp
    try:
        out = _sp.run(
            ["pgrep", "-P", str(root)],
            capture_output=True, text=True, timeout=5, check=False,
        )
        children = {int(p) for p in out.stdout.split() if p.strip()}
    except Exception:
        children = set()
    result = {root}
    for child in children:
        result |= _collect_pid_tree(child)
    return result


def _kill_pid_tree(pid: int) -> None:
    import signal as _signal
    tree = _collect_pid_tree(pid)
    others = sorted(tree - {pid}, reverse=True)
    for p in others:
        try:
            os.kill(p, _signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, _signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_proxy(
    prefill_instances: list[str],
    decode_instances: list[str],
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    prefill_only: bool = False,
) -> None:
    """一键启动代理服务（阻塞）。启动时写入 PID 文件，退出时清理。"""
    pid_file = _pid_file()
    pid_file.write_text(str(os.getpid()))
    logger.info("PID %d 已写入 %s", os.getpid(), pid_file)
    try:
        app = create_app(prefill_instances, decode_instances, model, prefill_only=prefill_only)
        config = uvicorn.Config(app, host=host, port=port, loop="uvloop")
        server = uvicorn.Server(config)
        server.run()
    finally:
        pid_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_start(args) -> int:
    """start 子命令：从 YAML 配置加载参数并启动代理（前台阻塞）。"""
    from pd_service_ctl import load_config  # 按需导入，避免启动时拖慢

    cfg = load_config(args.config)

    if cfg.proxy_port is None:
        logger.error("配置文件中未定义 proxy.port，无法启动代理")
        return 1

    prefill_addrs = [f"{cfg.local_ip}:{inst.port}" for inst in cfg.prefill_instances]
    decode_addrs  = [f"{cfg.local_ip}:{inst.port}" for inst in cfg.decode_instances]
    # 命令行显式指定时覆盖配置文件中的值
    if args.prefill_only is not None:
        prefill_only = args.prefill_only
    else:
        prefill_only = cfg.proxy_prefill_only

    logger.info(
        "启动代理 port=%d prefill=%s decode=%s prefill_only=%s",
        cfg.proxy_port, prefill_addrs, decode_addrs, prefill_only,
    )
    run_proxy(
        prefill_instances=prefill_addrs,
        decode_instances=decode_addrs,
        model=cfg.served_model_name,
        host="0.0.0.0",
        port=cfg.proxy_port,
        prefill_only=prefill_only,
    )
    return 0


def _cmd_stop(_args) -> int:
    """stop 子命令：通过 PID 文件停止代理进程。"""
    pf = _pid_file()
    if not pf.is_file():
        logger.warning("PID 文件 %s 不存在，尝试按进程名兜底查杀", pf)
        return _stop_by_name_fallback()

    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        pf.unlink(missing_ok=True)
        logger.error("PID 文件内容无效，已删除")
        return 1

    if not _pid_alive(pid):
        pf.unlink(missing_ok=True)
        logger.info("PID %d 已不存在，清理 PID 文件", pid)
        return 0

    logger.info("停止代理进程 PID %d ...", pid)
    _kill_pid_tree(pid)
    pf.unlink(missing_ok=True)
    logger.info("代理已停止")
    return 0


def _stop_by_name_fallback() -> int:
    import subprocess as _sp
    my_pid = os.getpid()
    try:
        out = _sp.run(
            ["pgrep", "-f", "pd_proxy.py"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        pids = [int(p) for p in out.stdout.split() if p.strip() and int(p) != my_pid]
    except Exception:
        pids = []
    if not pids:
        logger.info("未找到运行中的 pd_proxy.py 进程")
        return 0
    for pid in pids:
        logger.info("兜底停止 PID %d", pid)
        _kill_pid_tree(pid)
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PD 代理启停（配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python pd_proxy.py start --config configs/qwen3_32b_1p2_1d2.yaml
  python pd_proxy.py stop
""",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="启动代理服务（前台运行）")
    p_start.add_argument(
        "--config", "-c", type=Path, required=True,
        help="YAML 配置文件路径",
    )
    po = p_start.add_mutually_exclusive_group()
    po.add_argument(
        "--prefill-only", dest="prefill_only", action="store_true", default=None,
        help="强制开启 prefill-only 模式（decode 打桩），覆盖配置文件中的值",
    )
    po.add_argument(
        "--no-prefill-only", dest="prefill_only", action="store_false",
        help="强制关闭 prefill-only 模式，覆盖配置文件中的值",
    )

    sub.add_parser("stop", help="停止代理服务（通过 PID 文件）")

    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    sys.exit(_cmd_start(args) if args.cmd == "start" else _cmd_stop(args))
