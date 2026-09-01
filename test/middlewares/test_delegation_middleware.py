"""DelegationMiddleware 的幂等合并、缓存与隔离测试。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.delegation import DelegationMiddleware


def _request(call_id: str, *, task: str = "查天气", tool_name: str = "task") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"subagent_type": "web-researcher", "task": task},
            "id": call_id,
        },
        tool=None,
        state={},
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", user_id="u1")),
    )


async def test_non_task_tool_passes_through() -> None:
    middleware = DelegationMiddleware()
    handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(_request("call-1", tool_name="read_file"), handler)

    assert result.content == "ok"
    handler.assert_awaited_once()


async def test_concurrent_duplicate_delegations_share_one_execution() -> None:
    middleware = DelegationMiddleware()
    release = asyncio.Event()
    executions = 0

    async def handler(request):
        nonlocal executions
        executions += 1
        await release.wait()
        return ToolMessage(content="result", tool_call_id=request.tool_call["id"])

    first = asyncio.create_task(middleware.awrap_tool_call(_request("call-1"), handler))
    second = asyncio.create_task(middleware.awrap_tool_call(_request("call-2"), handler))
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert executions == 1
    assert first_result.content == second_result.content == "result"
    assert first_result.tool_call_id == "call-1"
    assert second_result.tool_call_id == "call-2"
    assert first_result.additional_kwargs["delegation_task_id"].startswith("subtask_")


async def test_different_task_payloads_execute_separately() -> None:
    middleware = DelegationMiddleware()
    handler = AsyncMock(side_effect=lambda request: ToolMessage(
        content=request.tool_call["args"]["task"], tool_call_id=request.tool_call["id"],
    ))

    one = await middleware.awrap_tool_call(_request("call-1", task="任务一"), handler)
    two = await middleware.awrap_tool_call(_request("call-2", task="任务二"), handler)

    assert handler.await_count == 2
    assert one.content == "任务一"
    assert two.content == "任务二"


async def test_exception_is_not_cached() -> None:
    middleware = DelegationMiddleware()
    handler = AsyncMock(side_effect=[RuntimeError("boom"), ToolMessage(content="ok", tool_call_id="call-2")])

    try:
        await middleware.awrap_tool_call(_request("call-1"), handler)
    except RuntimeError:
        pass

    result = await middleware.awrap_tool_call(_request("call-2"), handler)

    assert handler.await_count == 2
    assert result.content == "ok"
