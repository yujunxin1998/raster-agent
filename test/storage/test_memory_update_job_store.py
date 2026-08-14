"""`MemoryUpdateJobStore` 单元测试：mock asyncpg pool，不依赖真实 Postgres 连接。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from _fake_asyncpg import FakeConnection, FakePool

from src.storage.memory_update_job_store import MemoryUpdateJobStore


async def test_create_job_merges_into_existing_pending_job_within_debounce_window() -> None:
    conn = FakeConnection(fetchval_result="existing-job-1")
    store = MemoryUpdateJobStore(MagicMock())

    job_id = await store.create_job(
        conn, user_id="user-1", conversation_id="conv-1", agent_name=None,
        first_event_id="evt-2", last_event_id="evt-2", debounce_seconds=30,
    )

    assert job_id == "existing-job-1"
    conn.fetchval.assert_awaited_once()
    conn.execute.assert_not_awaited()  # 没有走 INSERT 分支


async def test_create_job_inserts_new_job_when_no_mergeable_pending_job() -> None:
    conn = FakeConnection(fetchval_result=None)
    store = MemoryUpdateJobStore(MagicMock())

    job_id = await store.create_job(
        conn, user_id="user-1", conversation_id="conv-1", agent_name=None,
        first_event_id="evt-1", last_event_id="evt-1", debounce_seconds=30,
    )

    assert job_id
    conn.execute.assert_awaited_once()
    insert_sql = conn.execute.await_args.args[0]
    assert "INSERT INTO memory_update_job" in insert_sql


async def test_claim_next_returns_none_when_no_job_available() -> None:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)
    store = MemoryUpdateJobStore(pool)

    result = await store.claim_next(debounce_seconds=30, lease_seconds=120)

    assert result is None


async def test_claim_next_returns_claimed_job() -> None:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={
        "job_id": "job-1", "user_id": "user-1", "conversation_id": "conv-1",
        "agent_name": None, "first_event_id": "evt-1", "last_event_id": "evt-1",
        "trace_id": None, "attempts": 0,
    })
    store = MemoryUpdateJobStore(pool)

    result = await store.claim_next(debounce_seconds=30, lease_seconds=120)

    assert result["job_id"] == "job-1"


async def test_mark_succeeded_updates_status() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()
    store = MemoryUpdateJobStore(pool)

    await store.mark_succeeded("job-1")

    pool.execute.assert_awaited_once()
    assert "status = $2" in pool.execute.await_args.args[0]


async def test_mark_retry_transitions_to_dead_when_attempts_exhausted() -> None:
    conn = FakeConnection(fetchval_result=5)  # 新 attempts 已达上限
    pool = FakePool(conn)
    store = MemoryUpdateJobStore(pool)

    await store.mark_retry("job-1", "boom", max_attempts=5, backoff_seconds=[30, 120])

    dead_call = conn.execute.await_args_list[-1]
    assert dead_call.args[1] == "job-1"
    assert dead_call.args[2] == "dead"


async def test_mark_retry_schedules_backoff_when_attempts_remain() -> None:
    conn = FakeConnection(fetchval_result=2)
    pool = FakePool(conn)
    store = MemoryUpdateJobStore(pool)

    await store.mark_retry("job-1", "boom", max_attempts=5, backoff_seconds=[30, 120, 600])

    retry_call = conn.execute.await_args_list[-1]
    assert retry_call.args[1] == "job-1"
    assert retry_call.args[2] == "pending"
    assert retry_call.args[3] == 120  # backoff_seconds[min(2-1, ...)] = backoff_seconds[1]


async def test_mark_retry_noop_when_job_missing() -> None:
    conn = FakeConnection(fetchval_result=None)
    pool = FakePool(conn)
    store = MemoryUpdateJobStore(pool)

    await store.mark_retry("missing-job", "boom", max_attempts=5, backoff_seconds=[30])

    # attempts+1 那次 UPDATE 走的是 fetchval（RETURNING），命中不到行直接 return，
    # 不会再有第二次 execute。
    conn.execute.assert_not_awaited()


async def test_reclaim_expired_leases_returns_count() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock(return_value="UPDATE 3")
    store = MemoryUpdateJobStore(pool)

    count = await store.reclaim_expired_leases()

    assert count == 3
