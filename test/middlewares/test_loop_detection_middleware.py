"""LoopDetectionMiddleware 单元测试：连续 3 次相同工具调用，第 3 次应被短路。

`recent_tool_calls` 由中间件通过 `Command.update` 返回——中间件每次只提交本次
这一条新签名（`[signature]`），不是拼好的完整历史（并行工具调用场景下拼接
必须交给 `PipelineState` 声明的 reducer 做，见 `loop_detection.py::
awrap_tool_call` 的说明）。真实运行时框架会调用这个 reducer
（`state.py::_append_recent_tool_calls`）把新签名合并回 state 供下一次工具
调用读取——这里手动调用同一个 reducer 函数模拟这个合并过程，而不是直接覆盖。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.state import _append_recent_tool_calls


def _merge(state: dict, update: dict) -> dict:
    """模拟 LangGraph 用 `_append_recent_tool_calls` reducer 合并一次 `Command.update`。"""
    return {"recent_tool_calls": _append_recent_tool_calls(state.get("recent_tool_calls"), update["recent_tool_calls"])}


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
    state = _merge(state, result_1.update)

    result_2 = await middleware.awrap_tool_call(_make_request(state, "call-2"), handler)
    state = _merge(state, result_2.update)

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
    state = _merge({}, result_a.update)

    request_b = ToolCallRequest(
        tool_call={"name": "same_tool", "args": {"x": 2}, "id": "call-2"},
        tool=None, state=state, runtime=SimpleNamespace(context=None),
    )
    await middleware.awrap_tool_call(request_b, handler)

    assert handler.await_count == 2


async def test_parallel_tool_calls_in_same_step_do_not_conflict_on_merge() -> None:
    """回归测试：同一步内并行发起的多个工具调用各自写 `[signature]`，
    reducer 必须能把它们依次叠加而不是互相覆盖/冲突（真实场景下 LangGraph 用
    `PipelineState.recent_tool_calls` 的 reducer 合并同一步内的多次
    `Command.update`；这里直接调用 reducer 模拟"同一步内两次并行写入"）。
    """
    middleware = LoopDetectionMiddleware(threshold=3)
    handler = AsyncMock(side_effect=lambda request: ToolMessage(content="ok", tool_call_id=request.tool_call["id"]))

    state: dict = {}
    request_a = ToolCallRequest(
        tool_call={"name": "tool_a", "args": {}, "id": "call-a"},
        tool=None, state=state, runtime=SimpleNamespace(context=None),
    )
    request_b = ToolCallRequest(
        tool_call={"name": "tool_b", "args": {}, "id": "call-b"},
        tool=None, state=state, runtime=SimpleNamespace(context=None),
    )
    result_a = await middleware.awrap_tool_call(request_a, handler)
    result_b = await middleware.awrap_tool_call(request_b, handler)

    # 同一步内两个并行分支各自提交的 update，LangGraph 会依次喂给同一个 reducer
    # （而不是分别用不同的初始 state）——这里模拟这个顺序合并。
    merged = _append_recent_tool_calls(None, result_a.update["recent_tool_calls"])
    merged = _append_recent_tool_calls(merged, result_b.update["recent_tool_calls"])

    assert merged == [
        "tool_a:[]",
        "tool_b:[]",
    ]
