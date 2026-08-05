"""`sanitize_dangling_tool_calls` 单元测试：修复悬空的工具调用。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_core.agents.dangling_tool_calls import sanitize_dangling_tool_calls


def _fake_agent(messages: list):
    agent = SimpleNamespace()
    agent.aget_state = AsyncMock(return_value=SimpleNamespace(values={"messages": messages}))
    agent.aupdate_state = AsyncMock()
    return agent


async def test_dangling_tool_call_gets_placeholder_response() -> None:
    messages = [
        HumanMessage(content="帮我查一下天气"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "天气"}, "id": "call-1"}]),
        # 进程崩溃，call-1 从未收到 ToolMessage
    ]
    agent = _fake_agent(messages)

    await sanitize_dangling_tool_calls(agent, {"configurable": {"thread_id": "c1"}})

    agent.aupdate_state.assert_awaited_once()
    _, update = agent.aupdate_state.await_args.args
    fix_messages = update["messages"]
    assert len(fix_messages) == 1
    assert isinstance(fix_messages[0], ToolMessage)
    assert fix_messages[0].tool_call_id == "call-1"
    assert fix_messages[0].content == "[工具调用已中断，跳过]"


async def test_no_dangling_calls_is_a_noop() -> None:
    messages = [
        HumanMessage(content="帮我查一下天气"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "天气"}, "id": "call-1"}]),
        ToolMessage(content="晴天", tool_call_id="call-1"),
        AIMessage(content="今天是晴天"),
    ]
    agent = _fake_agent(messages)

    await sanitize_dangling_tool_calls(agent, {"configurable": {"thread_id": "c1"}})

    agent.aupdate_state.assert_not_awaited()


async def test_empty_history_is_a_noop() -> None:
    agent = _fake_agent([])

    await sanitize_dangling_tool_calls(agent, {"configurable": {"thread_id": "c1"}})

    agent.aupdate_state.assert_not_awaited()
