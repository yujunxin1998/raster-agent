"""`validate_delta()` 的运行时上下文校验规则测试。"""
from __future__ import annotations

from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch
from src.agent_core.memory.memory_delta_validator import validate_delta

_DEFAULTS = {
    "max_profile_patches": 5, "max_fact_operations": 10, "max_text_length": 2000, "sensitive_filter_enabled": True,
}


def _delta(**kwargs) -> MemoryDelta:
    return MemoryDelta(source_event_ids=["e-1"], **kwargs)


def test_valid_delta_passes() -> None:
    delta = _delta(
        profile_patches=[ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=["m-1"])],
        fact_operations=[FactOperation(
            op="add", content="用户偏好使用 Vim", category="preference",
            confidence=0.9, evidence_message_ids=["m-1"],
        )],
    )

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is None


def test_rejects_evidence_outside_visible_messages() -> None:
    delta = _delta(profile_patches=[
        ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=["m-999"])
    ])

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is not None
    assert "evidence_message_ids" in rejection.reason


def test_rejects_target_fact_id_outside_visible_candidates() -> None:
    delta = _delta(fact_operations=[
        FactOperation(op="archive", target_fact_id="ghost-fact", evidence_message_ids=["m-1"])
    ])

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is not None
    assert "不在本次可见候选范围内" in rejection.reason


def test_rejects_operation_on_archived_fact() -> None:
    delta = _delta(fact_operations=[
        FactOperation(op="reinforce", target_fact_id="f-1")
    ])

    rejection = validate_delta(
        delta, visible_message_ids=set(),
        visible_facts={"f-1": {"status": "archived"}}, **_DEFAULTS,
    )

    assert rejection is not None
    assert "不允许再变更" in rejection.reason


def test_allows_operation_on_active_or_pending_fact() -> None:
    delta = _delta(fact_operations=[
        FactOperation(op="reinforce", target_fact_id="f-1"),
        FactOperation(op="update", target_fact_id="f-2", importance=6),
    ])

    rejection = validate_delta(
        delta, visible_message_ids=set(),
        visible_facts={"f-1": {"status": "active"}, "f-2": {"status": "pending"}}, **_DEFAULTS,
    )

    assert rejection is None


def test_rejects_sensitive_content_in_profile_patch() -> None:
    delta = _delta(profile_patches=[
        ProfilePatch(field="work_context", op="set", value="api_key=sk-abcdefghijklmnop1234", evidence_message_ids=["m-1"])
    ])

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is not None
    assert "敏感信息" in rejection.reason


def test_rejects_sensitive_content_in_fact_operation() -> None:
    delta = _delta(fact_operations=[
        FactOperation(
            op="add", content="密码 password=hunter2hunter2", category="context",
            confidence=0.9, evidence_message_ids=["m-1"],
        )
    ])

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is not None
    assert "敏感信息" in rejection.reason


def test_rejects_too_many_profile_patches() -> None:
    patches = [
        ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=["m-1"])
        for _ in range(6)
    ]
    delta = _delta(profile_patches=patches)

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is not None
    assert "上限" in rejection.reason


def test_rejects_text_exceeding_max_length() -> None:
    delta = _delta(profile_patches=[
        ProfilePatch(field="top_of_mind", op="set", value="x" * 10, evidence_message_ids=["m-1"])
    ])

    rejection = validate_delta(
        delta, visible_message_ids={"m-1"}, visible_facts={},
        max_profile_patches=5, max_fact_operations=10, max_text_length=5, sensitive_filter_enabled=False,
    )

    assert rejection is not None
    assert "长度上限" in rejection.reason


def test_add_operation_skips_target_fact_lookup() -> None:
    delta = _delta(fact_operations=[
        FactOperation(
            op="add", content="用户计划学习 Rust", category="goal",
            confidence=0.9, evidence_message_ids=["m-1"],
        )
    ])

    rejection = validate_delta(delta, visible_message_ids={"m-1"}, visible_facts={}, **_DEFAULTS)

    assert rejection is None
