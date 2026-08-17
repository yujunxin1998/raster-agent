"""DanglingToolCallMiddleware 单元测试：修复消息历史里两类不合法的
tool_calls/ToolMessage 配对状态。

`abefore_agent` 命中修复时返回的是一次 `REMOVE_ALL_MESSAGES` + 重建后的完整
消息列表（不是零散的增量），所以这里断言的是 `result["messages"]` 整体形状：
第一条必须是 `RemoveMessage(id=REMOVE_ALL_MESSAGES)`，之后的顺序必须是修复后
的完整历史（用 `langgraph.graph.message.add_messages` 实际跑一遍验证最终顺序，
而不是只看中间产物），理由见 `dangling_tool_call.py::_fix_message_order` 的
模块内说明。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage, add_messages

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
    update = result["messages"]
    assert isinstance(update[0], RemoveMessage) and update[0].id == REMOVE_ALL_MESSAGES

    final = add_messages(messages, update)
    assert [type(m).__name__ for m in final] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert final[-1].tool_call_id == "call-1"
    assert final[-1].content == "[工具调用已中断，跳过]"


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
    final = add_messages(messages, result["messages"])
    assert [m.id for m in final] == ["0", "2"]


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
        HumanMessage(content="再查一次", id="1"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "call-2"}], id="2"),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    final = add_messages(messages, result["messages"])
    assert [m.id for m in final[:2]] == ["1", "2"]
    assert final[2].tool_call_id == "call-2"


async def test_misplaced_response_gets_relocated_even_though_it_exists() -> None:
    """回归测试：真实生产场景复现——两个并行 tool_calls 的响应*确实存在*，
    但中间插了一条用户新消息，响应被挤到了新消息后面（这正是这个中间件
    早期版本"只查有没有响应、不查响应是否紧跟其后"漏掉的那一类，参见
    `dangling_tool_call.py` 模块文档第 3 类问题——线上真实复现过：两次
    `task` 并行调用的占位 ToolMessage 都在，只是位置错了，仍然触发了
    "assistant message with tool_calls must be followed by tool messages"
    400 错误）。
    """
    messages = [
        HumanMessage(content="对比几个云平台", id="h1"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "task", "args": {"subagent_type": "web-researcher", "task": "a"}, "id": "call-1"},
                {"name": "task", "args": {"subagent_type": "web-researcher", "task": "b"}, "id": "call-2"},
            ],
            id="a1",
        ),
        HumanMessage(content="继续维度补充", id="h2"),  # 结果还没落盘，用户抢先又发了一条
        ToolMessage(content="[工具调用已中断，跳过]", tool_call_id="call-1", id="t1"),
        ToolMessage(content="[工具调用已中断，跳过]", tool_call_id="call-2", id="t2"),
        HumanMessage(content="还需要总结报告", id="h3"),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    final = add_messages(messages, result["messages"])
    assert [m.id for m in final] == ["h1", "a1", "t1", "t2", "h2", "h3"]


async def test_dangling_placeholder_is_inserted_before_new_turn_message_not_appended_after() -> None:
    """回归测试：上一轮工具执行过程中进程崩溃，留下悬空 tool_calls；本轮新的
    `HumanMessage` 在 `abefore_agent` 触发时已经先合并进 `state["messages"]`
    （模块文档里明确的既有行为）。占位 `ToolMessage` 必须重新排到紧跟着那条
    悬空 `AIMessage` 之后、新 `HumanMessage` 之前——如果只是简单 append 到
    列表末尾，顺序会变成 AIMessage -> HumanMessage -> ToolMessage，对模型
    供应商 API 来说仍然不合法，会继续触发
    "assistant message with tool_calls must be followed by tool messages" 400
    错误（这正是这个中间件本该防住、但旧实现没有防住的那类线上问题）。
    """
    messages = [
        HumanMessage(content="对比一下几个云平台", id="h1"),
        AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "call-1"}], id="a1"),
        # 进程在这里崩溃，call-1 从未收到 ToolMessage；下一轮用户消息紧接着到来
        HumanMessage(content="还在吗", id="h2"),
    ]
    middleware = DanglingToolCallMiddleware()

    result = await middleware.abefore_agent({"messages": messages}, runtime=None)

    assert result is not None
    final = add_messages(messages, result["messages"])
    assert [type(m).__name__ for m in final] == [
        "HumanMessage", "AIMessage", "ToolMessage", "HumanMessage",
    ]
    assert final[2].tool_call_id == "call-1"
    assert final[3].id == "h2"
