"""ToolErrorHandlingMiddleware 单元测试：异常转 error ToolMessage，不向上传播。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware


def _make_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "flaky_tool", "args": {}, "id": "call-1"},
        tool=None,
        state={},
        runtime=SimpleNamespace(context=None),
    )


async def test_handler_success_passes_through() -> None:
    middleware = ToolErrorHandlingMiddleware()
    request = _make_request()
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    assert result.content == "ok"


async def test_handler_exception_converted_to_error_tool_message() -> None:
    middleware = ToolErrorHandlingMiddleware()
    request = _make_request()
    handler = AsyncMock(side_effect=RuntimeError("boom"))

    result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "boom" in result.content
    assert result.tool_call_id == "call-1"
