"""`UserMemoryFactStore` 单元测试：mock asyncpg pool，不依赖真实 Postgres 连接。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from _fake_asyncpg import FakeConnection, FakePool

from src.storage.user_memory_fact_store import UserMemoryFactStore, normalize_fact


def test_normalize_fact_collapses_whitespace() -> None:
    assert normalize_fact("  用户   计划\n重构   记忆模块  ") == "用户 计划 重构 记忆模块"


def _fact_row(**overrides) -> dict:
    base = {
        "fact_id": "11111111-1111-1111-1111-111111111111",
        "user_id": "user-1", "agent_name": None, "content": "c", "normalized_content": "c",
        "category": "goal", "importance": 8, "confidence": 0.9, "status": "active",
        "source_event_id": None, "source_conversation_id": None,
        "evidence_message_ids": "[]", "supersedes_fact_id": None, "superseded_by_fact_id": None,
        "created_at": "2026-08-13T00:00:00", "updated_at": "2026-08-13T00:00:00",
        "expires_at": None, "last_accessed_at": None, "access_count": 0, "revision": 0,
    }
    base.update(overrides)
    return base


async def test_add_or_reinforce_inserts_new_fact_when_no_conflict() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[
        {"fact_id": "11111111-1111-1111-1111-111111111111", "status": "active"},  # INSERT ... RETURNING
        _fact_row(),  # _fetch_snapshot
    ])
    pool = FakePool(conn)

    result = await UserMemoryFactStore(pool).add_or_reinforce(
        user_id="user-1", content="用户计划重构记忆模块", category="goal",
        importance=8, confidence=0.95, status="active",
    )

    assert result["action"] == "added"
    assert result["status"] == "active"
    assert conn.execute.await_count == 1  # 只写了一条 outbox


async def test_add_or_reinforce_reinforces_on_duplicate_normalized_content() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[
        None,  # INSERT ... ON CONFLICT DO NOTHING 命中冲突，无返回行
        {"fact_id": "11111111-1111-1111-1111-111111111111", "status": "active"},  # 查已存在记录
        _fact_row(),  # _fetch_snapshot
    ])
    pool = FakePool(conn)

    result = await UserMemoryFactStore(pool).add_or_reinforce(
        user_id="user-1", content="用户计划重构记忆模块", category="goal",
        importance=8, confidence=0.95, status="active",
    )

    assert result["action"] == "reinforced"
    update_calls = [call for call in conn.execute.await_args_list if "access_count = access_count + 1" in call.args[0]]
    assert len(update_calls) == 1


async def test_supersede_returns_none_when_target_not_found() -> None:
    conn = FakeConnection(fetchrow_result=None)
    pool = FakePool(conn)

    result = await UserMemoryFactStore(pool).supersede(
        target_fact_id="missing", user_id="user-1", content="新内容", category="context",
        importance=7, confidence=0.9, source_event_id=None, evidence_message_ids=None,
    )

    assert result is None


async def test_supersede_returns_none_when_target_already_superseded() -> None:
    conn = FakeConnection(fetchrow_result={"fact_id": "x", "status": "superseded"})
    pool = FakePool(conn)

    result = await UserMemoryFactStore(pool).supersede(
        target_fact_id="x", user_id="user-1", content="新内容", category="context",
        importance=7, confidence=0.9, source_event_id=None, evidence_message_ids=None,
    )

    assert result is None


async def test_supersede_creates_new_fact_and_marks_old_superseded() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[
        {"fact_id": "old-1", "status": "active"},  # 校验旧 Fact
        _fact_row(fact_id="new-1"),  # snapshot(new)
        _fact_row(fact_id="old-1", status="superseded"),  # snapshot(old)
    ])
    pool = FakePool(conn)

    new_id = await UserMemoryFactStore(pool).supersede(
        target_fact_id="old-1", user_id="user-1", content="用户目前在 B 公司", category="context",
        importance=7, confidence=0.97, source_event_id="evt-1", evidence_message_ids=["m-1"],
    )

    assert new_id is not None
    insert_call = conn.execute.await_args_list[0]
    assert "INSERT INTO user_memory_fact" in insert_call.args[0]
    archive_old_call = conn.execute.await_args_list[1]
    assert "superseded_by_fact_id" in archive_old_call.args[0]
    assert archive_old_call.args[2] == "superseded"


async def test_update_returns_false_when_fact_not_mutable() -> None:
    conn = FakeConnection(fetchrow_result={"status": "archived"})
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).update("f-1", "user-1", importance=9)

    assert ok is False


async def test_update_writes_only_provided_fields() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[{"status": "active"}, _fact_row()])
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).update("f-1", "user-1", importance=9)

    assert ok is True
    update_call = conn.execute.await_args_list[0]
    assert "importance = $3" in update_call.args[0]
    assert "content =" not in update_call.args[0]


async def test_update_allow_any_status_bypasses_mutable_check() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[{"status": "archived"}, _fact_row(status="active")])
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).update("f-1", "user-1", status="active", allow_any_status=True)

    assert ok is True
    update_call = conn.execute.await_args_list[0]
    assert "status = $3" in update_call.args[0]


async def test_update_clear_expires_at_sets_null() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[{"status": "active"}, _fact_row()])
    pool = FakePool(conn)

    await UserMemoryFactStore(pool).update("f-1", "user-1", clear_expires_at=True)

    update_call = conn.execute.await_args_list[0]
    assert update_call.args[3] is None


async def test_update_noop_when_no_fields_given() -> None:
    pool = MagicMock()
    pool.acquire = MagicMock()

    ok = await UserMemoryFactStore(pool).update("f-1", "user-1")

    assert ok is True
    pool.acquire.assert_not_called()


async def test_reinforce_returns_false_when_not_found() -> None:
    conn = FakeConnection(fetchrow_result=None)
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).reinforce("missing", "user-1")

    assert ok is False


async def test_reinforce_keeps_higher_confidence() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[{"status": "active", "confidence": 0.9}, _fact_row()])
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).reinforce("f-1", "user-1", confidence=0.6)

    assert ok is True
    update_call = conn.execute.await_args_list[0]
    assert update_call.args[3] == 0.9  # max(0.9, 0.6)


async def test_archive_returns_false_when_not_found() -> None:
    conn = FakeConnection(fetchrow_result=None)
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).archive("missing", "user-1")

    assert ok is False


async def test_delete_returns_false_when_not_found() -> None:
    conn = FakeConnection(fetchrow_result=None)
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).delete("missing", "user-1")

    assert ok is False
    conn.execute.assert_not_awaited()


async def test_delete_removes_row_and_writes_deleted_outbox() -> None:
    conn = FakeConnection(fetchrow_result={"fact_id": "f-1"})
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).delete("f-1", "user-1")

    assert ok is True
    outbox_call = conn.execute.await_args_list[0]
    assert "memory_outbox" in outbox_call.args[0]
    assert outbox_call.args[2] == "fact_deleted"


async def test_purge_user_deletes_every_fact() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[{"fact_id": "f-1"}, {"fact_id": "f-2"}])
    store = UserMemoryFactStore(pool)
    store.delete = AsyncMock(side_effect=[True, False])

    deleted = await store.purge_user("user-1")

    assert deleted == 1
    assert store.delete.await_count == 2


async def test_confirm_pending_only_matches_pending_status() -> None:
    conn = FakeConnection()
    conn.fetchrow = AsyncMock(side_effect=[{"fact_id": "f-1"}, _fact_row(status="active")])
    pool = FakePool(conn)

    ok = await UserMemoryFactStore(pool).confirm_pending("f-1", "user-1")

    assert ok is True
    # UPDATE ... RETURNING 由 fetchrow 承担；校验 SQL 里带上了来源状态过滤
    sql = conn.fetchrow.await_args_list[0].args[0]
    assert "status = $4" in sql


async def test_get_returns_none_when_missing() -> None:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)

    result = await UserMemoryFactStore(pool).get("missing", "user-1")

    assert result is None


async def test_get_many_returns_empty_list_for_empty_input() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock()

    result = await UserMemoryFactStore(pool).get_many([], "user-1")

    assert result == []
    pool.fetch.assert_not_awaited()


async def test_list_pending_filters_by_pending_status() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[_fact_row(status="pending")])

    result = await UserMemoryFactStore(pool).list_pending("user-1")

    assert result[0]["status"] == "pending"
    args = pool.fetch.await_args.args
    assert args[2] == ["pending"]


async def test_sweep_stale_archives_matching_rows() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[
        {"fact_id": "f-1", "user_id": "user-1"}, {"fact_id": "f-2", "user_id": "user-2"},
    ])
    store = UserMemoryFactStore(pool)
    store.archive = AsyncMock(side_effect=[True, False])

    archived = await store.sweep_stale(max_age_days=90, low_importance_threshold=3)

    assert archived == 1
    assert store.archive.await_count == 2


async def test_sweep_stale_returns_zero_when_nothing_matches() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])

    archived = await UserMemoryFactStore(pool).sweep_stale(max_age_days=90, low_importance_threshold=3)

    assert archived == 0


async def test_claim_outbox_batch_parses_snapshot_json() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[
        {"outbox_id": 1, "fact_id": "11111111-1111-1111-1111-111111111111",
         "action": "fact_created", "snapshot": json.dumps({"content": "x"})},
    ])

    result = await UserMemoryFactStore(pool).claim_outbox_batch(10)

    assert result[0]["snapshot"] == {"content": "x"}


async def test_to_dict_parses_evidence_message_ids_json_string() -> None:
    row = _fact_row(evidence_message_ids=json.dumps(["m-1", "m-2"]))

    result = UserMemoryFactStore._to_dict(row)

    assert result["evidence_message_ids"] == ["m-1", "m-2"]
