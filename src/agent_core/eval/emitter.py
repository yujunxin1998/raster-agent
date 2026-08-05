"""`EvalEventEmitter` —— 对话级监控状态机，推送到 Redis Streams。

原样迁移自 `diit-agent-server` 的 `src/core/evaluation/emitter.py`，做了一处
结构性简化：原实现的 `record_routing`/`record_supervisor_call`/`on_llm_chunk`
按 LangGraph 节点名检测"agent 切换"，是给旧架构的多节点 Supervisor 图设计的。
新架构下 Lead Agent 只有一个模型节点，委派工具（`delegate_to_rag_agent` 等）
经 `sub_agent_factory.py` 的 config 隔离处理后，在外层看来是不透明的工具调用，
不再有"agent 切换"这个概念——本轮从头到尾只有"Lead Agent 本轮"这一个整体记录，
`_finalize` 只在 `flush_trace` 时调用一次。保留：`on_tool_start_sync`/
`on_tool_end_sync`（写 `eval:tool`）、`on_token_usage`/`on_agent_text`、
`flush_trace`（写 `eval:trace`）。

Stream 说明：
    eval:trace — 对话级完整追踪，`flush_trace()` 写入，每次对话一条。
    eval:tool  — 工具调用明细，`on_tool_end_sync()` 写入，每次工具调用一条。
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

_STREAM_TRACE = "eval:trace"
_STREAM_TOOL = "eval:tool"
_SCHEMA_VERSION = "2"
_MAXLEN = 100_000
_LEAD_AGENT_NAME = "lead_agent"

_redis_client: aioredis.Redis | None = None
_redis_checked: bool = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_call_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_redis() -> aioredis.Redis | None:
    """懒加载单例：首次调用时读取 `REDIS_URL`，未配置时返回 None（调用方全部静默跳过）。"""
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
        logger.warning(f"[EvalEmitter] Redis 客户端初始化失败: {exc}")
    return _redis_client


async def aclose_emitter_redis() -> None:
    """应用关闭时显式释放连接，供 lifespan 调用。"""
    global _redis_client, _redis_checked
    if _redis_client is not None:
        await _redis_client.aclose()
    _redis_client = None
    _redis_checked = False


async def _xadd(redis: aioredis.Redis, stream: str, fields: dict) -> None:
    try:
        await redis.xadd(stream, fields, maxlen=_MAXLEN, approximate=True)
    except Exception as exc:
        logger.warning(f"[EvalEmitter] XADD {stream} 失败: {exc}")


class EvalEventEmitter:
    """一次对话请求的监控状态机，构造后贯穿整个请求生命周期。"""

    def __init__(self, trace_id: str, conversation_id: str, user_id: str, user_input: str) -> None:
        """初始化。

        Args:
            trace_id: 链路追踪 ID。
            conversation_id: 会话 ID。
            user_id: 归属用户 ID。
            user_input: 本轮用户发言，用于写入 `eval:trace`。
        """
        self.trace_id = trace_id
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.user_input = user_input

        self._start_ms = int(time.time() * 1000)
        self._call_id = _new_call_id()

        self._token_input = 0
        self._token_output = 0
        self._tools_used: list[str] = []
        self._output_parts: list[str] = []

        self._pending_tools: dict[str, dict] = {}

    def on_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """累加 token 使用量，max-delta 模式防止 provider 重复发送累计值导致重复计数。"""
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
        if input_tokens > self._token_input:
            self._token_input = input_tokens
        if output_tokens > self._token_output:
            self._token_output = output_tokens

    def on_agent_text(self, text: str) -> None:
        """捕获流式输出片段，供 `flush_trace` 未显式传入 `final_answer` 时兜底使用。"""
        if text:
            self._output_parts.append(text)

    def on_tool_start_sync(self, tool_name: str, run_id: str, tool_input: Any) -> None:
        """工具调用开始，记录待配对的元数据（同步方法，调用方无需 await）。"""
        try:
            args_str = json.dumps(tool_input, ensure_ascii=False) if isinstance(tool_input, dict) else str(tool_input)
        except Exception:
            args_str = str(tool_input)
        self._pending_tools[run_id] = {"name": tool_name, "args": args_str, "start_ms": int(time.time() * 1000)}
        self._tools_used.append(tool_name)

    async def on_tool_end_sync(self, run_id: str, result: str, status: str = "success", error_msg: str = "") -> None:
        """工具调用结束，写入 `eval:tool`。`run_id` 未匹配到（如未配置 Redis 时的调用）静默跳过。"""
        redis = _get_redis()
        if redis is None:
            return
        pending = self._pending_tools.pop(run_id, None)
        if pending is None:
            return

        latency_ms = int(time.time() * 1000) - pending["start_ms"]
        fields = {
            "schema_version": _SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "run_id": run_id,
            "agent_call_id": self._call_id,
            "agent_name": _LEAD_AGENT_NAME,
            "tool_name": pending["name"],
            "tool_args": pending["args"][:2048],
            "tool_result": result[:4096],
            "status": status,
            "latency_ms": str(latency_ms),
            "retry_count": "0",
            "error_msg": error_msg[:1024],
            "created_at": _now_iso(),
        }
        await _xadd(redis, _STREAM_TOOL, fields)

    async def flush_trace(self, final_answer: str, task_status: str = "success", error_message: str = "") -> None:
        """对话结束后写入 `eval:trace`，一次对话一条。"""
        redis = _get_redis()
        if redis is None:
            return

        output = final_answer or "".join(self._output_parts)
        agents_called = [{
            "call_id": self._call_id,
            "agent_name": _LEAD_AGENT_NAME,
            "latency_ms": int(time.time() * 1000) - self._start_ms,
            "status": task_status,
            "output": output[:300],
            "error": error_message[:300],
            "tools_used": list(dict.fromkeys(self._tools_used)),
            "token_input": self._token_input,
            "token_output": self._token_output,
        }]

        total_latency = int(time.time() * 1000) - self._start_ms
        fields = {
            "schema_version": _SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "user_input": self.user_input[:2048],
            "final_answer": final_answer[:4096],
            "agents_called": json.dumps(agents_called, ensure_ascii=False),
            "route_history": "[]",
            "total_latency": str(total_latency),
            "token_input": str(self._token_input),
            "token_output": str(self._token_output),
            "task_status": task_status,
            "error_message": error_message[:1024],
            "created_at": _now_iso(),
        }
        await _xadd(redis, _STREAM_TRACE, fields)
