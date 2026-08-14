"""`MemoryApplyEngine` 单元测试：mock Profile/Fact/Audit Store。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_core.memory.memory_apply_engine import MemoryApplyEngine, ProfilePatchConflictError
from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch


def _engine(profile_store=None, fact_store=None, audit_store=None, active_confidence_threshold=0.85):
    profile_store = profile_store or MagicMock()
    fact_store = fact_store or MagicMock()
    return MemoryApplyEngine(
        profile_store, fact_store, audit_store, active_confidence_threshold=active_confidence_threshold,
    )


async def test_apply_with_no_operations_is_noop() -> None:
    engine = _engine()

    outcome = await engine.apply(MemoryDelta(), user_id="user-1", source_event_id="evt-1")

    assert outcome.profile_patched is False
    assert outcome.fact_results == []


async def test_apply_profile_patches_reads_snapshot_and_applies() -> None:
    profile_store = MagicMock()
    profile_store.get_with_revision = AsyncMock(return_value={"revision": 3})
    profile_store.apply_patches = AsyncMock(return_value=True)
    engine = _engine(profile_store=profile_store)
    delta = MemoryDelta(profile_patches=[
        ProfilePatch(field="top_of_mind", op="set", value="重构记忆模块", evidence_message_ids=["m-1"])
    ])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    assert outcome.profile_patched is True
    profile_store.apply_patches.assert_awaited_once()
    assert profile_store.apply_patches.await_args.kwargs["expected_revision"] == 3


async def test_apply_profile_patches_retries_once_on_revision_conflict() -> None:
    profile_store = MagicMock()
    profile_store.get_with_revision = AsyncMock(side_effect=[{"revision": 3}, {"revision": 4}])
    profile_store.apply_patches = AsyncMock(side_effect=[False, True])
    engine = _engine(profile_store=profile_store)
    delta = MemoryDelta(profile_patches=[
        ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=["m-1"])
    ])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    assert outcome.profile_patched is True
    assert profile_store.apply_patches.await_count == 2
    assert profile_store.get_with_revision.await_count == 2


async def test_apply_profile_patches_raises_after_two_conflicts() -> None:
    profile_store = MagicMock()
    profile_store.get_with_revision = AsyncMock(return_value={"revision": 3})
    profile_store.apply_patches = AsyncMock(return_value=False)
    engine = _engine(profile_store=profile_store)
    delta = MemoryDelta(profile_patches=[
        ProfilePatch(field="top_of_mind", op="set", value="x", evidence_message_ids=["m-1"])
    ])

    with pytest.raises(ProfilePatchConflictError):
        await engine.apply(delta, user_id="user-1", source_event_id="evt-1")


async def test_add_operation_below_active_threshold_saved_as_pending() -> None:
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "pending"})
    engine = _engine(fact_store=fact_store, active_confidence_threshold=0.85)
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="add", content="用户可能喜欢 Rust", category="preference",
                      confidence=0.75, evidence_message_ids=["m-1"])
    ])

    await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    assert fact_store.add_or_reinforce.await_args.kwargs["status"] == "pending"


async def test_add_operation_above_active_threshold_saved_as_active() -> None:
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "active"})
    engine = _engine(fact_store=fact_store, active_confidence_threshold=0.85)
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="add", content="用户在字节跳动工作", category="context",
                      confidence=0.95, evidence_message_ids=["m-1"])
    ])

    await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    assert fact_store.add_or_reinforce.await_args.kwargs["status"] == "active"


async def test_update_operation_calls_fact_store_update() -> None:
    fact_store = MagicMock()
    fact_store.update = AsyncMock(return_value=True)
    engine = _engine(fact_store=fact_store)
    delta = MemoryDelta(fact_operations=[FactOperation(op="update", target_fact_id="f-1", importance=9)])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    fact_store.update.assert_awaited_once()
    assert outcome.fact_results[0]["action"] == "updated"


async def test_supersede_operation_calls_fact_store_supersede() -> None:
    fact_store = MagicMock()
    fact_store.supersede = AsyncMock(return_value="new-fact-1")
    engine = _engine(fact_store=fact_store)
    delta = MemoryDelta(fact_operations=[FactOperation(
        op="supersede", target_fact_id="old-1", content="新内容", category="context",
        confidence=0.97, reason="用户明确纠正", evidence_message_ids=["m-1"],
    )])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    fact_store.supersede.assert_awaited_once()
    assert outcome.fact_results[0]["fact_id"] == "new-fact-1"
    assert outcome.fact_results[0]["old_fact_id"] == "old-1"


async def test_archive_operation_calls_fact_store_archive() -> None:
    fact_store = MagicMock()
    fact_store.archive = AsyncMock(return_value=True)
    engine = _engine(fact_store=fact_store)
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="archive", target_fact_id="f-1", evidence_message_ids=["m-1"])
    ])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    fact_store.archive.assert_awaited_once_with("f-1", "user-1")
    assert outcome.fact_results[0]["action"] == "archived"


async def test_reinforce_operation_calls_fact_store_reinforce() -> None:
    fact_store = MagicMock()
    fact_store.reinforce = AsyncMock(return_value=True)
    engine = _engine(fact_store=fact_store)
    delta = MemoryDelta(fact_operations=[FactOperation(op="reinforce", target_fact_id="f-1")])

    outcome = await engine.apply(delta, user_id="user-1", source_event_id="evt-1")

    fact_store.reinforce.assert_awaited_once()
    assert outcome.fact_results[0]["action"] == "reinforced"


async def test_audit_store_none_does_not_raise() -> None:
    fact_store = MagicMock()
    fact_store.archive = AsyncMock(return_value=True)
    engine = _engine(fact_store=fact_store, audit_store=None)
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="archive", target_fact_id="f-1", evidence_message_ids=["m-1"])
    ])

    await engine.apply(delta, user_id="user-1", source_event_id="evt-1")  # 不应抛异常


async def test_audit_store_failure_is_swallowed() -> None:
    fact_store = MagicMock()
    fact_store.archive = AsyncMock(return_value=True)
    audit_store = MagicMock()
    audit_store.record = AsyncMock(side_effect=Exception("db down"))
    engine = _engine(fact_store=fact_store, audit_store=audit_store)
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="archive", target_fact_id="f-1", evidence_message_ids=["m-1"])
    ])

    await engine.apply(delta, user_id="user-1", source_event_id="evt-1")  # 不应向上抛出
