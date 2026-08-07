"""记忆后处理提取中间件（设计文档 4.2 节 #11）。

对应 `src/agent_core/memory/memory_extractor.py`：逻辑完全不变，仍然是
fire-and-forget——只是触发时机从 `chat_service.py` 手写的
`asyncio.create_task` 改为中间件的 `aafter_agent` 收尾钩子。

三层记忆架构下这里同一个钩子额外起了第二个独立的 fire-and-forget 任务，
更新用户画像与时间线（L1/L2，`UserProfileUpdater`）——跟 Facts 提取
（L3，`MemoryExtractor`）是两次独立的 LLM 调用，互不依赖，一个失败不影响
另一个。
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.memory.memory_manager import MemoryManager
from src.agent_core.middlewares.context import AgentRuntimeContext


class MemoryExtractionMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """会话结束后，异步提取并保存值得长期记忆的内容。"""

    def __init__(self, memory_manager: MemoryManager) -> None:
        """初始化中间件。

        Args:
            memory_manager: 记忆机制门面，通常传入 `get_memory_manager()` 单例。
        """
        super().__init__()
        self._memory_manager = memory_manager

    async def aafter_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """取本轮最后一组用户发言/AI 回复，异步触发记忆提取。

        Args:
            state: 当前 Agent 状态，含本轮结束后的完整 `messages`。
            runtime: 运行时上下文，携带 `user_id`/`conversation_id`。

        Returns:
            始终返回 None（本中间件不更新 state，只是 fire-and-forget 触发
            后台提取任务）。
        """
        context = runtime.context
        if context is None or not context.user_id or not context.conversation_id:
            return None

        messages = state.get("messages") or []
        user_message = self._last_text(messages, HumanMessage)
        ai_response = self._last_text(messages, AIMessage)
        if not user_message or not ai_response:
            return None

        asyncio.create_task(
            self._extract(user_message, ai_response, context.user_id, context.conversation_id)
        )
        asyncio.create_task(
            self._update_profile(user_message, ai_response, context.user_id)
        )
        return None

    async def _extract(self, user_message: str, ai_response: str, user_id: str, conversation_id: str) -> None:
        try:
            await self._memory_manager.extract_after_chat(
                user_message=user_message, ai_response=ai_response,
                user_id=user_id, conversation_id=conversation_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryExtractionMiddleware] 后台提取任务异常，已忽略: {exc}")

    async def _update_profile(self, user_message: str, ai_response: str, user_id: str) -> None:
        try:
            await self._memory_manager.update_profile_after_chat(
                user_message=user_message, ai_response=ai_response, user_id=user_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryExtractionMiddleware] 后台画像更新任务异常，已忽略: {exc}")

    @staticmethod
    def _last_text(messages: list, message_type: type) -> str:
        """取消息列表里最后一条指定类型消息的文本内容。"""
        for message in reversed(messages):
            if isinstance(message, message_type) and isinstance(message.content, str):
                return message.content
        return ""
