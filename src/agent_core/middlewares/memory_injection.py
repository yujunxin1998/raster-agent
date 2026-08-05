"""长期记忆注入中间件（设计文档 4.2 节 #3）。

对应原 `chat_service.py::_build_memory_context_message` 的手写胶水代码：
检索长期记忆，构造临时的提示词内容注入本轮模型调用，不写入持久化历史
（沿用现有做法——`request.system_prompt` 只对本次 `handler(request)` 生效，
不会被 checkpoint 持久化，天然满足"不落库"这个约束）。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from src.agent_core.memory.memory_manager import MemoryManager
from src.agent_core.middlewares.context import AgentRuntimeContext


class MemoryInjectionMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """在模型调用前检索相关长期记忆，拼进本次调用的 system_prompt。"""

    def __init__(self, memory_manager: MemoryManager) -> None:
        """初始化中间件。

        Args:
            memory_manager: 记忆机制门面，通常传入 `get_memory_manager()` 单例。
        """
        super().__init__()
        self._memory_manager = memory_manager

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """检索长期记忆并注入本次模型调用的 system_prompt。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            模型调用结果。
        """
        context = request.runtime.context
        query = self._last_human_text(request.messages)

        if context is not None and context.user_id and query:
            memory_context = await self._memory_manager.get_relevant_context(
                query=query, user_id=context.user_id, conversation_id=context.conversation_id,
            )
            if memory_context:
                merged_prompt = f"{request.system_prompt}\n\n{memory_context}" if request.system_prompt else memory_context
                request = request.override(system_prompt=merged_prompt)

        return await handler(request)

    @staticmethod
    def _last_human_text(messages: list) -> str:
        """取消息列表里最后一条 HumanMessage 的文本内容，用作记忆检索的查询语句。"""
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                return message.content
        return ""
