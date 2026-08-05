"""`PatchedChatDeepSeek._get_request_payload` 的 reasoning_content 回填单测。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.agent_core.model.patched_deepseek import _restore_reasoning_content


def test_restores_reasoning_content_onto_assistant_payload() -> None:
    original_messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，有什么可以帮你", additional_kwargs={"reasoning_content": "用户在打招呼"}),
    ]
    payload_messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你"},
    ]

    _restore_reasoning_content(payload_messages, original_messages)

    assert payload_messages[1]["reasoning_content"] == "用户在打招呼"
    assert "reasoning_content" not in payload_messages[0]


def test_no_reasoning_content_leaves_payload_untouched() -> None:
    original_messages = [AIMessage(content="ok")]
    payload_messages = [{"role": "assistant", "content": "ok"}]

    _restore_reasoning_content(payload_messages, original_messages)

    assert "reasoning_content" not in payload_messages[0]


def test_message_count_mismatch_is_a_noop() -> None:
    original_messages = [AIMessage(content="ok", additional_kwargs={"reasoning_content": "x"})]
    payload_messages = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]

    _restore_reasoning_content(payload_messages, original_messages)

    assert "reasoning_content" not in payload_messages[0]
    assert "reasoning_content" not in payload_messages[1]
