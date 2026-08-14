"""`backfill_facts_from_es.backfill()` 单元测试：mock ES scan + asyncpg pool。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.storage.migrations.backfill_facts_from_es import backfill

_DOC = {
    "_id": "11111111-1111-1111-1111-111111111111",
    "_source": {
        "user_id": "user-1", "agent_name": None, "content": "用户偏好使用 Vim",
        "memory_type": "preference", "importance": 7, "status": "active",
        "conversation_id": "conv-1", "superseded_by": None,
        "created_at": "2026-01-01T00:00:00", "expires_at": None,
        "last_accessed_at": None, "access_count": 3,
    },
}


async def _fake_scan(hits):
    for hit in hits:
        yield hit


async def test_backfill_skips_already_migrated_fact_id() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)  # 已存在
    pool.fetchrow = AsyncMock()

    with patch("src.storage.migrations.backfill_facts_from_es.async_scan", return_value=_fake_scan([_DOC])):
        migrated, skipped = await backfill(MagicMock(), "agent_memories", pool)

    assert migrated == 0
    assert skipped == 1
    pool.fetchrow.assert_not_awaited()


async def test_backfill_inserts_new_fact() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=None)  # 尚未迁移过
    pool.fetchrow = AsyncMock(return_value={"fact_id": "11111111-1111-1111-1111-111111111111"})

    with patch("src.storage.migrations.backfill_facts_from_es.async_scan", return_value=_fake_scan([_DOC])):
        migrated, skipped = await backfill(MagicMock(), "agent_memories", pool)

    assert migrated == 1
    assert skipped == 0
    insert_call = pool.fetchrow.await_args
    assert insert_call.args[7] == 7  # importance
    assert insert_call.args[8] == 1.0  # 统一置信度


async def test_backfill_skips_on_unique_conflict() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(return_value=None)  # ON CONFLICT DO NOTHING 命中

    with patch("src.storage.migrations.backfill_facts_from_es.async_scan", return_value=_fake_scan([_DOC])):
        migrated, skipped = await backfill(MagicMock(), "agent_memories", pool)

    assert migrated == 0
    assert skipped == 1


async def test_backfill_skips_documents_missing_required_fields() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock()
    pool.fetchrow = AsyncMock()
    bad_doc = {"_id": "x", "_source": {"user_id": "user-1"}}  # 缺 content/memory_type

    with patch("src.storage.migrations.backfill_facts_from_es.async_scan", return_value=_fake_scan([bad_doc])):
        migrated, skipped = await backfill(MagicMock(), "agent_memories", pool)

    assert (migrated, skipped) == (0, 0)
    pool.fetchval.assert_not_awaited()
