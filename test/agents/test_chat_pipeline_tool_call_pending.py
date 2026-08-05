"""`chat_pipeline.py::_handle_tool_call_pending` 的单元测试。

对应 `StreamingModelMiddleware` 提前广播的"仅工具名"标记——参数还没流完时
就能预告一张 pending 卡片，让前端不用干等参数流完才有反应（原来"叙述文字
已经说完了，但卡片迟迟不出现"的观感问题）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent_core.agents.chat_pipeline import _handle_tool_call_pending
from src.agent_core.tools.tool_filter import tool_filter_registry


def _patch_skill_manager(skill_names: list[str]):
    manager = MagicMock()
    manager.registry.names = set(skill_names)
    return patch("src.agent_core.agents.chat_pipeline.get_skill_manager", return_value=manager)


def test_pending_event_has_no_args_and_is_marked_pending() -> None:
    with _patch_skill_manager([]):
        event = _handle_tool_call_pending({"id": "call_1", "name": "write_file"})

    assert event == {
        "type": "tool_call",
        "content": {
            "tool_name": "write_file",
            "tool_args": None,
            "request_id": "call_1",
            "tool_type": "custom",
            "pending": True,
        },
    }


def test_pending_event_for_skill_uses_skill_call_type() -> None:
    with _patch_skill_manager(["some-skill"]):
        event = _handle_tool_call_pending({"id": "call_2", "name": "some-skill"})

    assert event["type"] == "skill_call"
    assert event["content"]["tool_type"] == "skill"


def test_pending_event_returns_none_when_id_or_name_missing() -> None:
    with _patch_skill_manager([]):
        assert _handle_tool_call_pending({"id": "", "name": "write_file"}) is None
        assert _handle_tool_call_pending({"id": "call_1", "name": ""}) is None


def test_pending_event_returns_none_for_filtered_tool() -> None:
    tool_filter_registry.register("save_memory")
    try:
        with _patch_skill_manager([]):
            assert _handle_tool_call_pending({"id": "call_1", "name": "save_memory"}) is None
    finally:
        tool_filter_registry.unregister("save_memory")
