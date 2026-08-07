"""`chat_pipeline.py::_find_last_human_index` 单元测试。

对应"重新生成"功能：撤回上一轮 AI 回复前，需要先找到最后一条
`HumanMessage` 的位置，之后的消息（上一轮的 AI/Tool 消息）才是需要撤回的
范围，这条 `HumanMessage` 本身要保留不变。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_core.agents.chat_pipeline import _find_last_human_index


def test_returns_none_for_empty_messages() -> None:
    assert _find_last_human_index([]) is None


def test_returns_none_when_no_human_message() -> None:
    messages = [AIMessage(content="你好")]
    assert _find_last_human_index(messages) is None


def test_returns_index_of_last_human_message() -> None:
    messages = [
        HumanMessage(content="第一句"),
        AIMessage(content="第一次回复"),
        HumanMessage(content="第二句"),
        AIMessage(content="第二次回复"),
    ]
    assert _find_last_human_index(messages) == 2


def test_ignores_tool_and_ai_messages_after_last_human() -> None:
    messages = [
        HumanMessage(content="帮我查一下"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "call_1"}]),
        ToolMessage(content="结果", tool_call_id="call_1"),
        AIMessage(content="查到了"),
    ]
    assert _find_last_human_index(messages) == 0
