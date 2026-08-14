"""`MemoryDelta`/`ProfilePatch`/`FactOperation` 的 Pydantic 校验规则测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch


def test_profile_patch_set_requires_value() -> None:
    with pytest.raises(ValidationError):
        ProfilePatch(field="top_of_mind", op="set", value="", evidence_message_ids=["m-1"])


def test_profile_patch_clear_allows_empty_value() -> None:
    patch = ProfilePatch(field="top_of_mind", op="clear", evidence_message_ids=["m-1"])
    assert patch.value == ""


def test_profile_patch_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ProfilePatch(field="not_a_real_field", op="set", value="x", evidence_message_ids=["m-1"])


def test_profile_patch_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=[])


def test_fact_operation_add_requires_content_category_confidence_evidence() -> None:
    with pytest.raises(ValidationError):
        FactOperation(op="add")

    op = FactOperation(
        op="add", content="用户偏好使用 Vim", category="preference",
        confidence=0.9, evidence_message_ids=["m-1"],
    )
    assert op.content == "用户偏好使用 Vim"


def test_fact_operation_update_requires_target_fact_id() -> None:
    with pytest.raises(ValidationError):
        FactOperation(op="update")

    op = FactOperation(op="update", target_fact_id="f-1", importance=7)
    assert op.target_fact_id == "f-1"


def test_fact_operation_supersede_requires_full_fields() -> None:
    with pytest.raises(ValidationError):
        FactOperation(op="supersede", target_fact_id="f-1")

    op = FactOperation(
        op="supersede", target_fact_id="f-1", content="新内容", category="context",
        confidence=0.95, reason="用户明确纠正", evidence_message_ids=["m-1"],
    )
    assert op.reason == "用户明确纠正"


def test_fact_operation_archive_requires_target_and_evidence() -> None:
    with pytest.raises(ValidationError):
        FactOperation(op="archive", target_fact_id="f-1")

    op = FactOperation(op="archive", target_fact_id="f-1", evidence_message_ids=["m-1"])
    assert op.target_fact_id == "f-1"


def test_fact_operation_reinforce_requires_target_fact_id() -> None:
    with pytest.raises(ValidationError):
        FactOperation(op="reinforce")

    op = FactOperation(op="reinforce", target_fact_id="f-1")
    assert op.target_fact_id == "f-1"


def test_memory_delta_defaults() -> None:
    delta = MemoryDelta()
    assert delta.schema_version == 1
    assert delta.profile_patches == []
    assert delta.fact_operations == []


def test_memory_delta_parses_full_example_from_design_doc() -> None:
    delta = MemoryDelta.model_validate({
        "schema_version": 1,
        "source_event_ids": ["e-001"],
        "profile_patches": [
            {
                "field": "top_of_mind", "op": "merge",
                "value": "正在重构 Agent 的 Skill 加载方式和长期记忆更新链路。",
                "confidence": 0.94, "evidence_message_ids": ["m-101"],
            }
        ],
        "fact_operations": [
            {
                "op": "add",
                "content": "用户计划将 Skill 设计为按需加载的工作流上下文，而不是每个 Skill 都注册为业务 Tool。",
                "category": "goal", "importance": 8, "confidence": 0.95,
                "evidence_message_ids": ["m-101"],
            }
        ],
    })
    assert delta.profile_patches[0].field == "top_of_mind"
    assert delta.fact_operations[0].category == "goal"
