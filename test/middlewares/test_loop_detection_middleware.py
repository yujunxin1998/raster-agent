"""LoopDetectionMiddleware 单元测试：连续 3 次相同工具调用，第 3 次应被短路。

`recent_tool_calls` 由中间件通过 `Command.update` 返回，真实运行时框架会
把它合并回 state 供下一次工具调用读取——这里手动把上一次返回的
`recent_tool_calls` 喂给下一次调用的 `state`，模拟这个合并过程。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware


def _make_request(state: dict, call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "same_tool", "args": {"x": 1}, "id": call_id},
        tool=None,
        state=state,
        runtime=SimpleNamespace(context=None),
    )


async def test_third_consecutive_identical_call_is_short_circuited() -> None:
    middleware = LoopDetectionMiddleware(threshold=3)
    handler = AsyncMock(side_effect=lambda request: ToolMessage(content="ok", tool_call_id=request.tool_call["id"]))

    state: dict = {}

    result_1 = await middleware.awrap_tool_call(_make_request(state, "call-1"), handler)
    assert isinstance(result_1, Command)
    state = {"recent_tool_calls": result_1.update["recent_tool_calls"]}

    result_2 = await middleware.awrap_tool_call(_make_request(state, "call-2"), handler)
    state = {"recent_tool_calls": result_2.update["recent_tool_calls"]}

    assert handler.await_count == 2  # 前两次都正常执行

    result_3 = await middleware.awrap_tool_call(_make_request(state, "call-3"), handler)

    assert handler.await_count == 2  # 第三次被短路，未再调用 handler
    tool_message = result_3.update["messages"][0]
    assert tool_message.status == "error"
    assert "连续 3 次" in tool_message.content


async def test_different_args_do_not_count_as_repeats() -> None:
    middleware = LoopDetectionMiddleware(threshold=3)
    handler = AsyncMock(side_effect=lambda request: ToolMessage(content="ok", tool_call_id=request.tool_call["id"]))

    request_a = ToolCallRequest(
        tool_call={"name": "same_tool", "args": {"x": 1}, "id": "call-1"},
        tool=None, state={}, runtime=SimpleNamespace(context=None),
    )
    result_a = await middleware.awrap_tool_call(request_a, handler)
    state = {"recent_tool_calls": result_a.update["recent_tool_calls"]}

    request_b = ToolCallRequest(
        tool_call={"name": "same_tool", "args": {"x": 2}, "id": "call-2"},
        tool=None, state=state, runtime=SimpleNamespace(context=None),
    )
    await middleware.awrap_tool_call(request_b, handler)

    assert handler.await_count == 2
