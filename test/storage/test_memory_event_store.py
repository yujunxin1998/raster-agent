"""`MemoryEventStore` 单元测试：mock asyncpg pool，不依赖真实 Postgres 连接。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from _fake_asyncpg import FakeConnection, FakePool

from src.storage.memory_event_store import MemoryEventStore


async def test_get_watermark_returns_none_for_new_conversation() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=None)

    result = await MemoryEventStore(pool).get_watermark("user-1", "conv-1")

    assert result is None


async def test_create_event_and_job_inserts_event_then_creates_job() -> None:
    conn = FakeConnection(fetchval_result="event-id-1")
    pool = FakePool(conn)
    job_store = MagicMock()
    job_store.create_job = AsyncMock(return_value="job-1")

    job_id = await MemoryEventStore(pool).create_event_and_job(
        job_store,
        user_id="user-1", conversation_id="conv-1",
        from_message_id="m-1", to_message_id="m-2",
        conversation=[{"role": "user", "content": "hi"}],
        content_hash="hash-1",
    )

    assert job_id == "job-1"
    conn.fetchval.assert_awaited_once()
    job_store.create_job.assert_awaited_once()
    assert job_store.create_job.await_args.args[0] is conn


async def test_create_event_and_job_returns_none_on_duplicate_window() -> None:
    conn = FakeConnection(fetchval_result=None)  # ON CONFLICT DO NOTHING 命中，无返回行
    pool = FakePool(conn)
    job_store = MagicMock()
    job_store.create_job = AsyncMock()

    job_id = await MemoryEventStore(pool).create_event_and_job(
        job_store,
        user_id="user-1", conversation_id="conv-1",
        from_message_id="m-1", to_message_id="m-2",
        conversation=[], content_hash="hash-1",
    )

    assert job_id is None
    job_store.create_job.assert_not_awaited()


async def test_get_conversation_for_job_concatenates_events_in_order() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[
        {"conversation_json": json.dumps([{"role": "user", "content": "第一句", "id": "m1"}])},
        {"conversation_json": json.dumps([{"role": "assistant", "content": "第二句", "id": "m2"}])},
    ])

    conversation = await MemoryEventStore(pool).get_conversation_for_job("user-1", "conv-1", "evt-1", "evt-2")

    assert conversation == [
        {"role": "user", "content": "第一句", "id": "m1"},
        {"role": "assistant", "content": "第二句", "id": "m2"},
    ]
