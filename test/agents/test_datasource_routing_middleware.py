"""`DatasourceRoutingMiddleware` 单元测试。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.agent_core.agents.datasource_routing_middleware import DatasourceRoutingMiddleware
from src.agent_core.middlewares.context import AgentRuntimeContext


def _make_request(*, datasource_id: str | None, messages: list) -> ModelRequest:
    return ModelRequest(
        model=None, system_prompt="you are Lead Agent", messages=messages, tool_choice=None,
        tools=[], response_format=None, state={},
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", datasource_id=datasource_id)),
    )


async def test_retries_once_when_datasource_configured_but_not_delegated() -> None:
    middleware = DatasourceRoutingMiddleware()
    request = _make_request(datasource_id="5", messages=[HumanMessage(content="查一下上个月的销量")])

    first_response = ModelResponse(result=[AIMessage(content="好的，我看看")])
    second_response = ModelResponse(result=[AIMessage(
        content="", tool_calls=[{"name": "delegate_to_database_agent", "args": {"task": "查销量"}, "id": "call-1"}],
    )])
    handler = AsyncMock(side_effect=[first_response, second_response])

    result = await middleware.awrap_model_call(request, handler)

    assert handler.await_count == 2
    retried_request = handler.await_args_list[1].args[0]
    assert retried_request.messages == request.messages
    assert "delegate_to_database_agent" in retried_request.system_prompt
    assert result is second_response


async def test_no_retry_when_model_already_delegates() -> None:
    middleware = DatasourceRoutingMiddleware()
    request = _make_request(datasource_id="5", messages=[HumanMessage(content="查一下上个月的销量")])

    response = ModelResponse(result=[AIMessage(
        content="", tool_calls=[{"name": "delegate_to_database_agent", "args": {}, "id": "call-1"}],
    )])
    handler = AsyncMock(return_value=response)

    result = await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()
    assert result is response


async def test_no_datasource_id_skips_check_entirely() -> None:
    middleware = DatasourceRoutingMiddleware()
    request = _make_request(datasource_id=None, messages=[HumanMessage(content="你好")])

    response = ModelResponse(result=[AIMessage(content="你好，有什么可以帮你")])
    handler = AsyncMock(return_value=response)

    result = await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()
    assert result is response


async def test_not_first_response_this_turn_skips_check() -> None:
    """已经进行过一轮工具调用后的第二次响应，不再重复校验（避免死循环）。"""
    middleware = DatasourceRoutingMiddleware()
    messages = [
        HumanMessage(content="查一下上个月的销量"),
        AIMessage(content="", tool_calls=[{"name": "some_other_tool", "args": {}, "id": "call-1"}]),
    ]
    request = _make_request(datasource_id="5", messages=messages)

    response = ModelResponse(result=[AIMessage(content="继续处理中")])
    handler = AsyncMock(return_value=response)

    result = await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()
    assert result is response
