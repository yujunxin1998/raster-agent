"""`MemoryStalenessReviewer` 单元测试：mock `BaseMemoryStore`，不依赖真实 ES。"""
from __future__ import annotations

from unittest.mock import AsyncMock

from src.agent_core.memory.memory_staleness_reviewer import MemoryStalenessReviewer


def _reviewer(store) -> MemoryStalenessReviewer:
    return MemoryStalenessReviewer(store, max_age_days=90, low_importance_threshold=3)


async def test_run_once_returns_archived_count() -> None:
    store = AsyncMock()
    store.sweep_stale = AsyncMock(return_value=7)

    archived = await _reviewer(store).run_once()

    assert archived == 7
    store.sweep_stale.assert_awaited_once_with(max_age_days=90, low_importance_threshold=3)


async def test_run_once_swallows_exception_and_returns_zero() -> None:
    store = AsyncMock()
    store.sweep_stale = AsyncMock(side_effect=Exception("es down"))

    archived = await _reviewer(store).run_once()

    assert archived == 0
