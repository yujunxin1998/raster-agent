"""DanglingToolCallMiddleware 单元测试：修复消息历史里两类不合法的
tool_calls/ToolMessage 配对状态。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import RemoveMessage

from src.agent_core.middlewares.dangling_tool_call import DanglingToolCallMiddleware


async def test_dangling_tool_call_gets_placeholder_response() -> None:
    messages = [
        HumanMessage(content="帮我查一下天气"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "天气"}, "id": "call-1"}]),
        # 进程崩溃，call-1 从未收到 ToolMessage
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    fix_messages = result["messages"]
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
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is None


async def test_empty_history_is_a_noop() -> None:
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": []}, runtime=None)

    assert result is None


async def test_orphaned_tool_message_gets_removed() -> None:
    """回归测试：对应 `MemoryCompressor` 曾经把发起调用的 AIMessage 压缩掉、
    只留下 ToolMessage 的历史遗留场景（见 memory_compressor.py 的修复）。
    """
    messages = [
        HumanMessage(content="帮我查一下天气", id="0"),
        # AIMessage(tool_calls=[call-1]) 已经被压缩逻辑删掉了
        ToolMessage(content="晴天", tool_call_id="call-1", id="1"),
        AIMessage(content="今天是晴天", id="2"),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    fix_messages = result["messages"]
    assert len(fix_messages) == 1
    assert isinstance(fix_messages[0], RemoveMessage)
    assert fix_messages[0].id == "1"


async def test_no_orphaned_tool_messages_is_a_noop() -> None:
    messages = [
        HumanMessage(content="帮我查一下天气"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "天气"}, "id": "call-1"}]),
        ToolMessage(content="晴天", tool_call_id="call-1"),
        AIMessage(content="今天是晴天"),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is None


async def test_both_dangling_and_orphaned_fixed_in_one_pass() -> None:
    """两类问题可能同时出现在同一段历史里，一次 abefore_agent 调用应该把两边都修好。"""
    messages = [
        ToolMessage(content="孤儿响应", tool_call_id="orphan-call", id="0"),
        HumanMessage(content="再查一次"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "call-2"}]),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    fix_messages = result["messages"]
    assert len(fix_messages) == 2
    remove_fixes = [m for m in fix_messages if isinstance(m, RemoveMessage)]
    placeholder_fixes = [m for m in fix_messages if isinstance(m, ToolMessage)]
    assert {m.id for m in remove_fixes} == {"0"}
    assert {m.tool_call_id for m in placeholder_fixes} == {"call-2"}
