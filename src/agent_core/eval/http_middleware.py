"""`EvalMiddleware` —— FastAPI HTTP 中间件，记录每次请求的耗时/状态码/是否流式/TTFT。

原样迁移自 `diit-agent-server` 的 `src/core/evaluation/middleware.py`，HTTP 层
埋点与 Agent 架构无关，逻辑不变。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable

import redis.asyncio as aioredis
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

_STREAM_HTTP = "eval:http"
_SCHEMA_VERSION = "2"
_MAXLEN = 100_000

_redis_client: aioredis.Redis | None = None
_redis_checked: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_redis() -> aioredis.Redis | None:
    """懒加载单例：首次请求时读取 `REDIS_URL`，避免启动阶段 env 未就绪。"""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        from src.config.settings import get_settings

        redis_url = get_settings().REDIS_URL
        if not redis_url:
            return None
        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
    except Exception as exc:
        logger.warning(f"[EvalMiddleware] Redis 客户端初始化失败: {exc}")
    return _redis_client


async def aclose_middleware_redis() -> None:
    """应用关闭时显式释放连接，供 lifespan 调用。"""
    global _redis_client, _redis_checked
    if _redis_client is not None:
        await _redis_client.aclose()
    _redis_client = None
    _redis_checked = False


async def _xadd(redis: aioredis.Redis, fields: dict) -> None:
    try:
        await redis.xadd(_STREAM_HTTP, fields, maxlen=_MAXLEN, approximate=True)
    except Exception as exc:
        logger.warning(f"[EvalMiddleware] XADD {_STREAM_HTTP} 失败: {exc}")


class EvalMiddleware(BaseHTTPMiddleware):
    """挂载方式：`app.add_middleware(EvalMiddleware)`（见 `main.py`）。"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        redis = _get_redis()
        if redis is None:
            return await call_next(request)

        start_ms = int(time.time() * 1000)
        trace_id = ""
        conversation_id = ""

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = int(time.time() * 1000) - start_ms
            asyncio.create_task(_xadd(redis, {
                "schema_version": _SCHEMA_VERSION, "trace_id": trace_id, "conversation_id": conversation_id,
                "method": request.method, "path": request.url.path, "status_code": "500",
                "latency_ms": str(latency_ms), "is_stream": "false", "ttft_ms": "0", "created_at": _now_iso(),
            }))
            raise

        trace_id = response.headers.get("X-Trace-Id") or request.headers.get("X-Trace-Id", "")
        conversation_id = response.headers.get("X-Conversation-Id", "")

        content_type = response.headers.get("content-type", "")
        is_stream = "text/event-stream" in content_type or request.url.path.startswith("/ws")

        if is_stream:
            original_iterator = response.body_iterator
            request_method = request.method
            request_path = request.url.path
            status_code = str(response.status_code)

            async def ttft_wrapper():
                ttft_ms = 0
                first = True
                async for chunk in original_iterator:
                    if first:
                        ttft_ms = int(time.time() * 1000) - start_ms
                        first = False
                    yield chunk
                latency_ms = int(time.time() * 1000) - start_ms
                asyncio.create_task(_xadd(redis, {
                    "schema_version": _SCHEMA_VERSION, "trace_id": trace_id, "conversation_id": conversation_id,
                    "method": request_method, "path": request_path, "status_code": status_code,
                    "latency_ms": str(latency_ms), "is_stream": "true", "ttft_ms": str(ttft_ms),
                    "created_at": _now_iso(),
                }))

            response.body_iterator = ttft_wrapper()
            return response

        latency_ms = int(time.time() * 1000) - start_ms
        asyncio.create_task(_xadd(redis, {
            "schema_version": _SCHEMA_VERSION, "trace_id": trace_id, "conversation_id": conversation_id,
            "method": request.method, "path": request.url.path, "status_code": str(response.status_code),
            "latency_ms": str(latency_ms), "is_stream": "false", "ttft_ms": "0", "created_at": _now_iso(),
        }))
        return response


async def emit_ws_chat_record(trace_id: str, conversation_id: str, latency_ms: int, status_code: int = 200) -> None:
    """WebSocket 对话完成后，从业务层主动写入 `eval:http` 记录（中间件感知不到 WS 消息粒度）。"""
    redis = _get_redis()
    if redis is None:
        return
    asyncio.create_task(_xadd(redis, {
        "schema_version": _SCHEMA_VERSION, "trace_id": trace_id, "conversation_id": conversation_id,
        "method": "WS", "path": "/ws/chat", "status_code": str(status_code),
        "latency_ms": str(int(latency_ms)), "is_stream": "true", "ttft_ms": "0", "created_at": _now_iso(),
    }))
