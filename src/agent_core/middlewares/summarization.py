"""会话历史压缩中间件（设计文档 4.2 节 #9）。

触发条件、压缩逻辑完全不变（见 `MemoryCompressor.compress_messages`），
只是调用时机从 `chat_service.py` 手写的 `asyncio.create_task` 改为中间件
的 `aafter_agent` 收尾钩子——返回的 `state_update` 会被框架自动合并进
checkpoint，不需要像旧版 `compress_if_needed(graph, config)` 那样手动
`graph.aupdate_state`。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.memory.memory_manager import MemoryManager
from src.agent_core.middlewares.context import AgentRuntimeContext


class SummarizationMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """会话结束后检查消息数量，超阈值时压缩为结构化摘要。"""

    def __init__(self, memory_manager: MemoryManager) -> None:
        """初始化中间件。

        Args:
            memory_manager: 记忆机制门面，通常传入 `get_memory_manager()` 单例。
        """
        super().__init__()
        self._memory_manager = memory_manager

    async def aafter_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """检查并按需压缩本轮结束后的完整消息历史。

        Args:
            state: 当前 Agent 状态，含本轮结束后的完整 `messages`。
            runtime: 运行时上下文，携带 `user_id`/`conversation_id`。

        Returns:
            触发压缩时返回压缩产出的 state 更新（`messages` 字段的删除 +
            摘要替换）；未触发压缩或缺少必要上下文时返回 None。
        """
        context = runtime.context
        if context is None or not context.user_id or not context.conversation_id:
            return None

        messages = state.get("messages") or []
        state_update = await self._memory_manager.compress_messages_after_chat(
            messages=messages, user_id=context.user_id, conversation_id=context.conversation_id,
        )
        if state_update:
            logger.debug(f"[SummarizationMiddleware] 已压缩会话历史 conversation_id={context.conversation_id}")
        return state_update
