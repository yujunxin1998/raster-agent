"""GuardrailMiddleware 单元测试：DENY 时短路返回 error ToolMessage，不调用 handler。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.guardrail import GuardrailMiddleware


class _FakeGuardrailProvider:
    def __init__(self, decision: GuardrailDecision) -> None:
        self._decision = decision
        self.last_call_kwargs: dict | None = None

    async def check(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._decision


def _make_request(tool_name: str = "some_tool") -> ToolCallRequest:
    runtime = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", user_id="u1"))
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": {"x": 1}, "id": "call-1"},
        tool=None,
        state={},
        runtime=runtime,
    )


async def test_allow_calls_handler_and_returns_its_result() -> None:
    provider = _FakeGuardrailProvider(GuardrailDecision.allow())
    middleware = GuardrailMiddleware(provider)
    request = _make_request()
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_awaited_once_with(request)
    assert result.content == "ok"
    assert provider.last_call_kwargs["tool_name"] == "some_tool"
    assert provider.last_call_kwargs["user_id"] == "u1"


async def test_deny_short_circuits_without_calling_handler() -> None:
    provider = _FakeGuardrailProvider(GuardrailDecision.deny("未获授权执行"))
    middleware = GuardrailMiddleware(provider)
    request = _make_request()
    handler = AsyncMock(return_value=ToolMessage(content="不应该被返回", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_not_awaited()
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.content == "未获授权执行"
    assert result.tool_call_id == "call-1"
