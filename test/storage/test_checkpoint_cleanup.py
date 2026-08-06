"""`CheckpointCleanup` 单元测试：mock asyncpg pool + BaseCheckpointSaver，
不依赖真实 Postgres 连接。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.storage.checkpoint_cleanup import CheckpointCleanup


def _cleanup(pool, checkpointer) -> CheckpointCleanup:
    return CheckpointCleanup(pool, checkpointer)


async def test_delete_for_thread_calls_adelete_thread() -> None:
    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()

    await _cleanup(MagicMock(), checkpointer).delete_for_thread("conv-1")

    checkpointer.adelete_thread.assert_awaited_once_with("conv-1")


async def test_delete_for_thread_swallows_exception() -> None:
    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock(side_effect=Exception("db down"))

    # 不应该向上抛出异常
    await _cleanup(MagicMock(), checkpointer).delete_for_thread("conv-1")


async def test_cleanup_orphans_deletes_each_found_thread() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[{"thread_id": "orphan-1"}, {"thread_id": "orphan-2"}])
    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()

    cleaned = await _cleanup(pool, checkpointer).cleanup_orphans()

    assert cleaned == 2
    checkpointer.adelete_thread.assert_any_await("orphan-1")
    checkpointer.adelete_thread.assert_any_await("orphan-2")


async def test_cleanup_orphans_returns_zero_when_none_found() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    checkpointer = MagicMock()
    checkpointer.adelete_thread = AsyncMock()

    cleaned = await _cleanup(pool, checkpointer).cleanup_orphans()

    assert cleaned == 0
    checkpointer.adelete_thread.assert_not_awaited()


async def test_cleanup_orphans_returns_zero_on_query_failure() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(side_effect=Exception("query failed"))
    checkpointer = MagicMock()

    cleaned = await _cleanup(pool, checkpointer).cleanup_orphans()

    assert cleaned == 0
