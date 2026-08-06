"""`ElasticsearchMemoryStore.sweep_stale` 单元测试：mock `AsyncElasticsearch`，
验证归档查询的行为，不依赖真实 ES 连接。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from src.agent_core.memory.elasticsearch_memory_store import ElasticsearchMemoryStore


def _store() -> ElasticsearchMemoryStore:
    store = ElasticsearchMemoryStore(
        es_url="http://es:9200",
        es_memory_index="agent_memories",
        embedding_base_url="",
        embedding_model="",
        reranker_base_url="",
        reranker_model="",
        sensitive_filter_enabled=False,
    )
    store._es = AsyncMock()
    return store


async def test_sweep_stale_returns_updated_count() -> None:
    store = _store()
    store._es.update_by_query = AsyncMock(return_value={"updated": 5})

    archived = await store.sweep_stale(max_age_days=90, low_importance_threshold=3)

    assert archived == 5
    call_kwargs = store._es.update_by_query.call_args.kwargs
    assert call_kwargs["conflicts"] == "proceed"
    body = call_kwargs["body"]
    assert body["script"]["source"] == "ctx._source.status = 'archived'"
    filter_clause = body["query"]["bool"]["filter"]
    assert {"term": {"status": "active"}} in filter_clause
    should_clause = body["query"]["bool"]["should"]
    assert {"range": {"expires_at": {"lt": "now"}}} in should_clause


async def test_sweep_stale_returns_zero_when_es_unavailable() -> None:
    store = ElasticsearchMemoryStore(
        es_url="",
        es_memory_index="agent_memories",
        embedding_base_url="",
        embedding_model="",
        reranker_base_url="",
        reranker_model="",
        sensitive_filter_enabled=False,
    )

    archived = await store.sweep_stale(max_age_days=90, low_importance_threshold=3)

    assert archived == 0


async def test_sweep_stale_swallows_exception_and_returns_zero() -> None:
    store = _store()
    store._es.update_by_query = AsyncMock(side_effect=Exception("es error"))

    archived = await store.sweep_stale(max_age_days=90, low_importance_threshold=3)

    assert archived == 0
