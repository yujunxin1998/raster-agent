"""Redis Streams 可观测性埋点（`eval:trace`/`eval:tool`/`eval:http`）。

原样迁移自 `diit-agent-server` 的 `src/core/evaluation/`，`REDIS_URL` 未配置时
全部调用静默降级（不写入、不报错），不影响对话主流程。
"""
from __future__ import annotations

from src.agent_core.eval.emitter import EvalEventEmitter, aclose_emitter_redis
from src.agent_core.eval.http_middleware import (
    EvalMiddleware,
    aclose_middleware_redis,
    emit_ws_chat_record,
)

__all__ = [
    "EvalEventEmitter",
    "aclose_emitter_redis",
    "EvalMiddleware",
    "aclose_middleware_redis",
    "emit_ws_chat_record",
]
