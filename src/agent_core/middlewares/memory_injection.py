"""长期记忆注入中间件（Memory v2，设计文档 §8）。

跟旧版本最大的区别：Profile（L1/L2）与相关 Facts（L3）不再是两条独立预算、
各自查询的链路，而是交给统一的 `MemoryContextBuilder` 合并渲染进同一个
`MEMORY_MAX_CONTEXT_TOKENS` 预算；同一次 Agent Run 内可能有多次模型调用
（工具循环），借助 `AgentRuntimeContext.memory_cache` 只在第一次调用时真正
查询一次，后续复用（§8.2）。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from src.agent_core.memory.memory_context_builder import MemoryContextBuilder
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.config.settings import get_settings


class MemoryInjectionMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """在模型调用前构建长期记忆上下文，拼进本次调用的 system_prompt。"""

    def __init__(self, context_builder: MemoryContextBuilder) -> None:
        """初始化中间件。

        Args:
            context_builder: 统一的记忆上下文构建器。
        """
        super().__init__()
        self._context_builder = context_builder

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """构建长期记忆上下文并注入本次模型调用的 system_prompt。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            模型调用结果。
        """
        context = request.runtime.context
        settings = get_settings()

        if context is not None and context.user_id and settings.MEMORY_ENABLED and settings.MEMORY_INJECTION_ENABLED:
            query = self._last_human_text(request.messages)
            memory_context = await self._context_builder.build(
                user_id=context.user_id, query=query, cache=context.memory_cache,
            )
            if memory_context:
                merged_prompt = (
                    f"{request.system_prompt}\n\n{memory_context}" if request.system_prompt else memory_context
                )
                request = request.override(system_prompt=merged_prompt)

        return await handler(request)

    @staticmethod
    def _last_human_text(messages: list) -> str:
        """取消息列表里最后一条 HumanMessage 的文本内容，用作记忆检索的查询语句。"""
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                return message.content
        return ""
