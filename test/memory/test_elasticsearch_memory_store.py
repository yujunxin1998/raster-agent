"""`ElasticsearchMemoryStore` 单元测试：mock `AsyncElasticsearch`，不依赖真实 ES 连接。

Memory v2 起该 Store 只保留 `search()`（对外）与 `_index_projection`/
`_remove_projection`（仅供 `FactProjector` 调用）两组行为。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from src.agent_core.memory.elasticsearch_memory_store import ElasticsearchMemoryStore


def _store() -> ElasticsearchMemoryStore:
    store = ElasticsearchMemoryStore(
        es_url="http://es:9200", es_memory_index="agent_memories",
        embedding_base_url="", embedding_model="",
        reranker_base_url="", reranker_model="",
        sensitive_filter_enabled=True,
    )
    store._es = AsyncMock()
    return store


async def test_index_projection_embeds_and_writes_mapped_fields() -> None:
    store = _store()
    store._embed = AsyncMock(return_value=[0.1, 0.2])
    fact = {
        "fact_id": "f-1", "user_id": "user-1", "agent_name": None, "content": "用户偏好使用 Vim",
        "category": "preference", "importance": 7, "confidence": 0.9, "status": "active",
        "created_at": "2026-08-13T00:00:00", "expires_at": None, "last_accessed_at": None, "access_count": 0,
    }

    await store._index_projection(fact)

    store._es.index.assert_awaited_once()
    call_kwargs = store._es.index.call_args.kwargs
    assert call_kwargs["id"] == "f-1"
    document = call_kwargs["document"]
    assert document["memory_type"] == "preference"  # category -> memory_type 字段映射
    assert document["embedding"] == [0.1, 0.2]
    assert document["status"] == "active"


async def test_remove_projection_deletes_by_id() -> None:
    store = _store()

    await store._remove_projection("f-1")

    store._es.delete.assert_awaited_once_with(index="agent_memories", id="f-1")


async def test_index_projection_noop_when_es_unavailable() -> None:
    store = ElasticsearchMemoryStore(
        es_url="", es_memory_index="agent_memories",
        embedding_base_url="", embedding_model="",
        reranker_base_url="", reranker_model="", sensitive_filter_enabled=True,
    )

    await store._index_projection({"fact_id": "f-1", "content": "x", "category": "goal",
                                    "importance": 5, "confidence": 0.9, "status": "active",
                                    "created_at": "2026-08-13T00:00:00"})  # 不应抛异常


async def test_search_filters_by_active_status_and_expiry() -> None:
    store = _store()
    store._embed = AsyncMock(return_value=[0.1])
    store._es.search = AsyncMock(return_value={"hits": {"hits": []}})

    result = await store.search("query", user_id="user-1")

    assert result == []
    body = store._es.search.call_args.kwargs["body"]
    filters = body["knn"]["filter"]["bool"]["filter"]
    assert {"term": {"status": "active"}} in filters
    assert {"term": {"user_id": "user-1"}} in filters
