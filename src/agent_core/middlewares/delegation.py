"""`task` 工具的可靠委派中间件：合并并发重复调用并短期复用终态。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from src.agent_core.agents.delegation_protocol import build_delegation_task_id
from src.agent_core.middlewares.context import AgentRuntimeContext


@dataclass
class _DelegationEntry:
    future: asyncio.Task
    expires_at: float


class DelegationMiddleware(AgentMiddleware[object, AgentRuntimeContext]):
    """保证同一会话内相同 `task` 委派不会被并发重复执行。

    该缓存只覆盖一次 Lead Agent 实例的生命周期；持久化和跨进程恢复应由后续
    TaskStore 实现。失败只短暂缓存，避免故障期间立即形成重试风暴。
    """

    def __init__(self, *, success_ttl_seconds: float = 300, failure_ttl_seconds: float = 10) -> None:
        super().__init__()
        self._success_ttl = success_ttl_seconds
        self._failure_ttl = failure_ttl_seconds
        self._entries: dict[str, _DelegationEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _clone_message(message: ToolMessage, *, tool_call_id: str, task_id: str) -> ToolMessage:
        extra = dict(message.additional_kwargs or {})
        extra["delegation_task_id"] = task_id
        return ToolMessage(
            content=message.content,
            tool_call_id=tool_call_id,
            status=message.status,
            name=message.name,
            artifact=message.artifact,
            additional_kwargs=extra,
            response_metadata=dict(message.response_metadata or {}),
        )

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        call = request.tool_call
        if call.get("name") != "task":
            return await handler(request)

        context = request.runtime.context
        args = call.get("args") or {}
        task_id = build_delegation_task_id(
            conversation_id=context.conversation_id,
            parent_task_id=context.task_id,
            subagent_type=str(args.get("subagent_type", "")),
            task=str(args.get("task", "")),
        )
        now = time.monotonic()

        async with self._lock:
            expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired:
                self._entries.pop(key, None)

            entry = self._entries.get(task_id)
            if entry is None:
                future = asyncio.create_task(handler(request))
                entry = _DelegationEntry(future=future, expires_at=float("inf"))
                self._entries[task_id] = entry
                owner = True
            else:
                owner = False

        if not owner:
            logger.info(f"[DelegationMiddleware] 合并重复委派 task_id={task_id}")

        try:
            result = await asyncio.shield(entry.future)
        except BaseException:
            async with self._lock:
                if self._entries.get(task_id) is entry:
                    self._entries.pop(task_id, None)
            raise

        if not isinstance(result, ToolMessage):
            async with self._lock:
                self._entries.pop(task_id, None)
            return result

        ttl = self._failure_ttl if result.status == "error" else self._success_ttl
        entry.expires_at = time.monotonic() + ttl
        return self._clone_message(result, tool_call_id=call["id"], task_id=task_id)
