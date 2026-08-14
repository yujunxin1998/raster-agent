"""`FactProjector` 单元测试：mock `UserMemoryFactStore`/`ElasticsearchMemoryStore`。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.agent_core.memory.fact_projector import FactProjector


def _projector(fact_store=None, es_store=None) -> tuple[FactProjector, MagicMock, MagicMock]:
    fact_store = fact_store or MagicMock()
    es_store = es_store or MagicMock()
    fact_store.mark_outbox_processed = AsyncMock()
    fact_store.mark_outbox_failed = AsyncMock()
    es_store._index_projection = AsyncMock()
    es_store._remove_projection = AsyncMock()
    projector = FactProjector(fact_store, es_store, poll_interval_seconds=3, batch_size=20, max_attempts=10)
    return projector, fact_store, es_store


async def test_run_once_indexes_active_fact() -> None:
    projector, fact_store, es_store = _projector()
    fact_store.claim_outbox_batch = AsyncMock(return_value=[
        {"outbox_id": 1, "fact_id": "f-1", "action": "fact_created", "snapshot": {"fact_id": "f-1", "status": "active"}}
    ])

    processed = await projector.run_once()

    assert processed == 1
    es_store._index_projection.assert_awaited_once_with({"fact_id": "f-1", "status": "active"})
    es_store._remove_projection.assert_not_awaited()
    fact_store.mark_outbox_processed.assert_awaited_once_with(1)


async def test_run_once_removes_projection_for_non_active_status() -> None:
    projector, fact_store, es_store = _projector()
    fact_store.claim_outbox_batch = AsyncMock(return_value=[
        {"outbox_id": 2, "fact_id": "f-2", "action": "fact_status_changed",
         "snapshot": {"fact_id": "f-2", "status": "archived"}}
    ])

    await projector.run_once()

    es_store._remove_projection.assert_awaited_once_with("f-2")
    es_store._index_projection.assert_not_awaited()


async def test_run_once_marks_failed_on_projection_error() -> None:
    projector, fact_store, es_store = _projector()
    fact_store.claim_outbox_batch = AsyncMock(return_value=[
        {"outbox_id": 3, "fact_id": "f-3", "action": "fact_created", "snapshot": {"fact_id": "f-3", "status": "active"}}
    ])
    es_store._index_projection = AsyncMock(side_effect=Exception("es down"))

    await projector.run_once()

    fact_store.mark_outbox_failed.assert_awaited_once()
    assert fact_store.mark_outbox_failed.await_args.args[0] == 3
    fact_store.mark_outbox_processed.assert_not_awaited()


async def test_run_once_returns_batch_size_for_backpressure_decision() -> None:
    projector, fact_store, _ = _projector()
    fact_store.claim_outbox_batch = AsyncMock(return_value=[])

    processed = await projector.run_once()

    assert processed == 0
