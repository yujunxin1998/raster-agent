"""InputSanitizationMiddleware 单元测试：清洗末尾 HumanMessage 的控制字符/多余空白。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent_core.middlewares.input_sanitization import InputSanitizationMiddleware


async def test_dirty_input_gets_cleaned_and_message_id_preserved() -> None:
    middleware = InputSanitizationMiddleware()
    dirty = HumanMessage(content="hello\x00world  extra\x1f  spaces", id="msg-1")
    state = {"messages": [dirty]}

    result = await middleware.abefore_model(state, runtime=None)

    assert result is not None
    cleaned = result["messages"][0]
    assert cleaned.id == "msg-1"
    assert "\x00" not in cleaned.content
    assert "\x1f" not in cleaned.content
    assert "  " not in cleaned.content


async def test_clean_input_returns_none() -> None:
    middleware = InputSanitizationMiddleware()
    state = {"messages": [HumanMessage(content="already clean text", id="msg-1")]}

    result = await middleware.abefore_model(state, runtime=None)

    assert result is None


async def test_no_human_message_returns_none() -> None:
    middleware = InputSanitizationMiddleware()
    state = {"messages": [AIMessage(content="assistant only")]}

    result = await middleware.abefore_model(state, runtime=None)

    assert result is None
