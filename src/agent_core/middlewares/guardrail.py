"""权限校验中间件（设计文档 4.2 节 #4、5.1 节）。

工具调用前置授权：把 `GuardrailProvider.check()` 的校验动作从"技能工具
自己在 `_invoke` 里调用"（见 `skill_tool_factory.py`）提升为一层通用的、
覆盖全部工具（不只是技能）的中间件。拒绝时返回 error `ToolMessage` 而不是
抛异常中断整轮对话，与 `GuardrailProvider` 本身的容错哲学完全一致。
"""
from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.middlewares.context import AgentRuntimeContext


class GuardrailMiddleware(AgentMiddleware[object, AgentRuntimeContext]):
    """工具调用前置权限校验。必须排在 ToolErrorHandlingMiddleware 之前

    （即在组装时列在更靠前的位置——LangChain 中间件"列表里靠前的在最外层"，
    拒绝短路和执行报错包裹是两层不同的职责，见 `loop.py` 的组装顺序说明）。
    """

    def __init__(self, guardrail_provider: GuardrailProvider) -> None:
        """初始化中间件。

        Args:
            guardrail_provider: 权限校验器，通常传入 `get_guardrail_provider()` 单例。
        """
        super().__init__()
        self._guardrail_provider = guardrail_provider

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """校验本次工具调用是否被允许，拒绝时短路返回 error ToolMessage。

        Args:
            request: 工具调用请求。
            handler: 执行工具调用的回调。

        Returns:
            被拒绝时返回携带拒绝原因的 error `ToolMessage`；放行时返回
            `handler(request)` 的结果。
        """
        context = request.runtime.context
        tool_call = request.tool_call

        decision = await self._guardrail_provider.check(
            user_id=context.user_id if context else None,
            tool_name=tool_call["name"],
            tool_args=tool_call.get("args", {}),
            context=GuardrailContext(
                conversation_id=context.conversation_id if context else None,
                thinking=context.thinking if context else False,
            ),
        )
        if not decision.is_allowed:
            return ToolMessage(content=decision.reason, tool_call_id=tool_call["id"], status="error")

        return await handler(request)
