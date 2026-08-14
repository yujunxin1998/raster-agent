"""`MemoryCaptureMiddleware` 单元测试。

核心回归点：只捕获 watermark 之后的新增消息，过滤掉工具消息/悬挂 Tool Call/
附件说明块，单轮问候不入队，watermark 缺失时全量捕获。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.memory_capture import MemoryCaptureMiddleware


def _settings(**overrides) -> MagicMock:
    defaults = {
        "MEMORY_ENABLED": True, "MEMORY_AUTO_UPDATE_ENABLED": True, "MEMORY_UPDATE_DEBOUNCE_SECONDS": 30,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _runtime(*, user_id="u1", conversation_id="c1") -> SimpleNamespace:
    context = AgentRuntimeContext(conversation_id=conversation_id, user_id=user_id) if conversation_id else None
    return SimpleNamespace(context=context)


def _middleware(watermark=None, create_result="job-1"):
    event_store = MagicMock()
    event_store.get_watermark = AsyncMock(return_value=watermark)
    event_store.create_event_and_job = AsyncMock(return_value=create_result)
    job_store = MagicMock()
    return MemoryCaptureMiddleware(event_store, job_store), event_store, job_store


def _human(text: str, msg_id: str) -> HumanMessage:
    return HumanMessage(content=text, id=msg_id)


def _ai(text: str, msg_id: str, tool_calls=None) -> AIMessage:
    return AIMessage(content=text, id=msg_id, tool_calls=tool_calls or [])


async def test_missing_context_skips_capture() -> None:
    middleware, event_store, _ = _middleware()
    state = {"messages": [_human("hi", "m1"), _ai("hello", "m2")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime(conversation_id=None))

    event_store.get_watermark.assert_not_awaited()


async def test_memory_disabled_skips_capture() -> None:
    middleware, event_store, _ = _middleware()
    state = {"messages": [_human("介绍一下你自己的能力边界", "m1"), _ai("我可以...", "m2")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings",
               return_value=_settings(MEMORY_ENABLED=False)):
        await middleware.aafter_agent(state, _runtime())

    event_store.get_watermark.assert_not_awaited()


async def test_captures_full_history_when_no_watermark() -> None:
    middleware, event_store, job_store = _middleware(watermark=None)
    state = {"messages": [
        _human("我在字节跳动做后端工程师", "m1"),
        _ai("好的，了解了", "m2"),
    ]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    event_store.create_event_and_job.assert_awaited_once()
    kwargs = event_store.create_event_and_job.await_args.kwargs
    assert kwargs["from_message_id"] == ""
    assert kwargs["to_message_id"] == "m2"
    assert kwargs["conversation"] == [
        {"role": "user", "content": "我在字节跳动做后端工程师", "id": "m1"},
        {"role": "assistant", "content": "好的，了解了", "id": "m2"},
    ]


async def test_only_captures_messages_after_watermark() -> None:
    middleware, event_store, _ = _middleware(watermark="m2")
    state = {"messages": [
        _human("老消息", "m1"), _ai("老回复", "m2"),
        _human("我计划下季度学习 Rust", "m3"), _ai("好的，记下了", "m4"),
    ]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    kwargs = event_store.create_event_and_job.await_args.kwargs
    assert kwargs["from_message_id"] == "m2"
    assert kwargs["to_message_id"] == "m4"
    assert kwargs["conversation"] == [
        {"role": "user", "content": "我计划下季度学习 Rust", "id": "m3"},
        {"role": "assistant", "content": "好的，记下了", "id": "m4"},
    ]


async def test_no_new_messages_after_watermark_skips_capture() -> None:
    middleware, event_store, _ = _middleware(watermark="m2")
    state = {"messages": [_human("老消息", "m1"), _ai("老回复", "m2")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    event_store.create_event_and_job.assert_not_awaited()


async def test_filters_out_tool_messages_and_dangling_tool_call() -> None:
    middleware, event_store, _ = _middleware(watermark=None)
    state = {"messages": [
        _human("上证指数今天涨了多少", "m1"),
        AIMessage(content="", id="m2", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]),
        ToolMessage(content="搜索结果...", tool_call_id="call-1", id="m3"),
        _ai("上证指数今天上涨 0.96%", "m4"),
    ]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    kwargs = event_store.create_event_and_job.await_args.kwargs
    assert kwargs["conversation"] == [
        {"role": "user", "content": "上证指数今天涨了多少", "id": "m1"},
        {"role": "assistant", "content": "上证指数今天上涨 0.96%", "id": "m4"},
    ]


async def test_pure_tool_loop_without_final_reply_skips_capture() -> None:
    middleware, event_store, _ = _middleware(watermark=None)
    state = {"messages": [
        AIMessage(content="", id="m1", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]),
        ToolMessage(content="结果", tool_call_id="call-1", id="m2"),
    ]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    event_store.create_event_and_job.assert_not_awaited()


async def test_single_greeting_turn_skips_capture() -> None:
    middleware, event_store, _ = _middleware(watermark=None)
    state = {"messages": [_human("你好", "m1"), _ai("你好，有什么可以帮你", "m2")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    event_store.create_event_and_job.assert_not_awaited()


async def test_strips_attachment_block_from_human_message() -> None:
    middleware, event_store, _ = _middleware(watermark=None)
    content = "帮我看看这份报表\n\n【本轮附件】\n- report.xlsx 可通过 read_file 访问"
    state = {"messages": [_human(content, "m1"), _ai("好的，我看一下", "m2")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    kwargs = event_store.create_event_and_job.await_args.kwargs
    assert kwargs["conversation"][0]["content"] == "帮我看看这份报表"


async def test_watermark_not_found_in_history_falls_back_to_full_capture() -> None:
    """checkpoint 被压缩后旧 watermark 消息可能已经不在 state["messages"] 里。"""
    middleware, event_store, _ = _middleware(watermark="stale-id-not-in-history")
    state = {"messages": [_human("新会话第一句", "m10"), _ai("回复", "m11")]}

    with patch("src.agent_core.middlewares.memory_capture.get_settings", return_value=_settings()):
        await middleware.aafter_agent(state, _runtime())

    event_store.create_event_and_job.assert_awaited_once()
    kwargs = event_store.create_event_and_job.await_args.kwargs
    assert kwargs["from_message_id"] == "stale-id-not-in-history"
