"""基于 Elasticsearch + Xinference BGE 的向量检索投影（Memory v2）。

设计文档 §4.4/§7.6：PostgreSQL `user_memory_fact` 是 Facts 的规范化主存，这里
降级为纯粹的"可检索投影"——不再提供 save/update/delete/list_all/sweep_stale
这些写路径（那是旧版本把 ES 当主存的问题根源：写入即分片刷新延迟、无法做
跨字段事务、相似度阈值被迫承担事实判断）。唯一对外的公开方法是 `search()`
（kNN 粗召回 + Reranker 精排，供 `MemoryContextBuilder`/`recall_memory` 工具
使用）；`_index_projection()`/`_remove_projection()` 是只给 `FactProjector`
（`src/agent_core/memory/fact_projector.py`）调用的内部方法，按
`memory_outbox` 里的变更异步把 PostgreSQL 的 Fact 状态同步过来。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from elasticsearch import AsyncElasticsearch, NotFoundError
from loguru import logger

from src.agent_core.memory.base_memory_store import BaseMemoryStore

_RESULT_SOURCE_FIELDS = [
    "content", "memory_type", "importance", "confidence", "created_at",
    "status", "expires_at", "last_accessed_at", "access_count", "agent_name",
]

_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "agent_name": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "memory_type": {"type": "keyword"},
            "importance": {"type": "integer"},
            "confidence": {"type": "float"},
            "created_at": {"type": "date"},
            "status": {"type": "keyword"},
            "expires_at": {"type": "date"},
            "last_accessed_at": {"type": "date"},
            "access_count": {"type": "integer"},
        }
    }
}


class ElasticsearchMemoryStore(BaseMemoryStore):
    """基于 Elasticsearch + Xinference BGE 的向量检索投影。"""

    def __init__(
        self,
        es_url: str,
        es_memory_index: str,
        embedding_base_url: str,
        embedding_model: str,
        reranker_base_url: str,
        reranker_model: str,
        sensitive_filter_enabled: bool,
        es_api_key: str = "",
        es_username: str = "",
        es_password: str = "",
        es_verify_certs: bool = True,
    ) -> None:
        """初始化存储实例（不建立连接，连接延迟到 setup() 中完成）。

        Args:
            es_url: Elasticsearch 服务地址，为空时功能整体降级为不可用。
            es_memory_index: 记忆索引名称。
            embedding_base_url: Embedding 服务地址。
            embedding_model: Embedding 模型名称。
            reranker_base_url: Reranker 服务地址。
            reranker_model: Reranker 模型名称。
            sensitive_filter_enabled: 预留字段——敏感信息拦截已经在
                `UserMemoryFactStore`/`MemoryApplyEngine` 写入 PostgreSQL 之前
                做过，投影层不需要重复过滤，这里保留参数只是不改变构造签名。
            es_api_key: ES 认证 API Key。
            es_username: ES 认证用户名（未提供 api_key 时使用）。
            es_password: ES 认证密码。
            es_verify_certs: 是否校验 ES 的 TLS 证书。
        """
        self._es_url = es_url
        self._es_memory_index = es_memory_index
        self._embedding_base_url = embedding_base_url
        self._embedding_model = embedding_model
        self._reranker_base_url = reranker_base_url
        self._reranker_model = reranker_model
        self._sensitive_filter_enabled = sensitive_filter_enabled
        self._es_api_key = es_api_key
        self._es_username = es_username
        self._es_password = es_password
        self._es_verify_certs = es_verify_certs
        self._es: AsyncElasticsearch | None = None

    async def setup(self) -> None:
        """建立 ES 连接、确保索引存在，并主动探测 Embedding/Reranker 可用性。

        ES_URL 未配置时记 warning 并跳过，后续 search/投影写入均静默降级。
        """
        if not self._es_url:
            logger.warning("[MemoryStore] ES_URL 未配置，长期记忆检索功能不可用")
            return

        client_kwargs: dict = {"hosts": [self._es_url], "verify_certs": self._es_verify_certs}
        if self._es_api_key:
            client_kwargs["api_key"] = self._es_api_key
        elif self._es_username:
            client_kwargs["basic_auth"] = (self._es_username, self._es_password)

        self._es = AsyncElasticsearch(**client_kwargs)

        if not await self._es.indices.exists(index=self._es_memory_index):
            await self._es.indices.create(index=self._es_memory_index, body=_INDEX_MAPPING)
            logger.info(f"[MemoryStore] 创建 ES index: {self._es_memory_index}")
        else:
            try:
                await self._es.indices.put_mapping(
                    index=self._es_memory_index, body=_INDEX_MAPPING["mappings"]
                )
            except Exception as exc:
                logger.warning(f"[MemoryStore] 更新索引 mapping 失败（可忽略，不影响现有功能）: {exc}")

        await self._probe_embedding_and_reranker()

    async def close(self) -> None:
        """关闭 ES 连接，应用关闭阶段调用一次。"""
        if self._es:
            await self._es.close()

    async def _probe_embedding_and_reranker(self) -> None:
        """启动时主动探测一次 Embedding/Reranker 可用性，问题提前暴露在启动日志里。"""
        if self._embedding_base_url:
            vector = await self._embed("__memory_startup_probe__")
            if vector:
                logger.info("[MemoryStore] Embedding 探测成功，长期记忆向量化可用")
            else:
                logger.error(
                    "[MemoryStore] Embedding 探测失败，长期记忆将无法正常写入/检索"
                    "（运行期间会静默降级为空向量，不会报错中断对话），"
                    "请检查 EMBEDDING_BASE_URL/EMBEDDING_MODEL 配置"
                )

        if self._reranker_base_url:
            probe_candidates = [{"id": "probe", "content": "__memory_startup_probe__"}]
            reranked = await self._rerank("__memory_startup_probe__", probe_candidates, 1)
            if reranked and "score" in reranked[0]:
                logger.info("[MemoryStore] Reranker 探测成功")
            else:
                logger.warning(
                    "[MemoryStore] Reranker 探测失败，检索时会降级为 kNN 粗召回排序（不影响功能可用性），"
                    "请检查 RERANKER_BASE_URL/RERANKER_MODEL 配置"
                )

    # ── 公开接口：检索 ──────────────────────────────────────────

    async def search(
        self,
        query: str,
        user_id: str = "default",
        candidate_k: int = 20,
        top_n: int = 5,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        if not self._es:
            return []

        embedding = await self._embed(query)
        if not embedding:
            return []

        filters: list[dict] = [
            {"term": {"user_id": user_id}},
            # 正常情况下只有 active Fact 会被投影进 ES（见 _index_projection /
            # FactProjector），这里的 status 过滤是防御性的兜底，防止投影层
            # 出 bug 时把 pending/archived 内容泄漏进召回结果。
            {"term": {"status": "active"}},
            {
                "bool": {
                    "should": [
                        {"bool": {"must_not": {"exists": {"field": "expires_at"}}}},
                        {"range": {"expires_at": {"gt": "now"}}},
                    ]
                }
            },
        ]
        if memory_type:
            filters.append({"term": {"memory_type": memory_type}})

        response = await self._es.search(
            index=self._es_memory_index,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": embedding,
                    "k": candidate_k,
                    "num_candidates": candidate_k * 2,
                    "filter": {"bool": {"filter": filters}},
                },
                "_source": _RESULT_SOURCE_FIELDS,
                "size": candidate_k,
            },
        )

        hits = response["hits"]["hits"]
        if not hits:
            return []

        candidates = [
            {**self._build_response(hit["_id"], hit["_source"]), "score": hit["_score"]}
            for hit in hits
        ]

        reranked = await self._rerank(query, candidates, top_n)

        if reranked:
            asyncio.create_task(self._bump_access([memory["id"] for memory in reranked]))
        return reranked

    # ── 内部接口：仅供 FactProjector 调用 ────────────────────────

    async def _index_projection(self, fact: dict) -> None:
        """把一条 active Fact 写入/刷新 ES 投影（`FactProjector` 消费 outbox 时调用）。

        Args:
            fact: `memory_outbox.snapshot`，字段对齐 `user_memory_fact` 表。
        """
        if not self._es:
            return
        embedding = await self._embed(fact["content"])
        await self._es.index(
            index=self._es_memory_index,
            id=fact["fact_id"],
            document={
                "user_id": fact["user_id"],
                "agent_name": fact.get("agent_name"),
                "content": fact["content"],
                "embedding": embedding,
                "memory_type": fact["category"],
                "importance": fact["importance"],
                "confidence": float(fact["confidence"]),
                "created_at": fact["created_at"],
                "status": fact["status"],
                "expires_at": fact.get("expires_at"),
                "last_accessed_at": fact.get("last_accessed_at"),
                "access_count": fact.get("access_count") or 0,
            },
        )

    async def _remove_projection(self, fact_id: str) -> None:
        """把一条不再是 active 状态的 Fact 从 ES 投影里移除。"""
        if not self._es:
            return
        try:
            await self._es.delete(index=self._es_memory_index, id=fact_id)
        except NotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"[MemoryStore] 移除 ES 投影失败 fact_id={fact_id}: {exc}")

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _is_expired(expires_at_iso: str) -> bool:
        try:
            expires_at = datetime.fromisoformat(expires_at_iso.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at < datetime.now(timezone.utc)
        except (ValueError, AttributeError):
            return False

    def _build_response(self, memory_id: str, source: dict) -> dict:
        """把 ES `_source` 组装为统一的记忆响应 dict（供检索结果格式化使用）。"""
        status = source.get("status") or "active"
        expires_at = source.get("expires_at")
        if status == "active" and expires_at and self._is_expired(expires_at):
            status = "expired"
        return {
            "id": memory_id,
            "content": source["content"],
            "memory_type": source["memory_type"],
            "importance": source["importance"],
            "confidence": source.get("confidence"),
            "created_at": source["created_at"],
            "status": status,
            "expires_at": expires_at,
            "last_accessed_at": source.get("last_accessed_at"),
            "access_count": source.get("access_count") or 0,
            "agent_name": source.get("agent_name"),
        }

    async def _bump_access(self, memory_ids: list[str]) -> None:
        """命中记忆后异步更新 ES 侧访问统计，失败只记 warning，不影响检索主流程。

        注意：这里只更新 ES 投影，不回写 PostgreSQL——`access_count`/
        `last_accessed_at` 的规范值以 PostgreSQL 为准，ES 侧的计数仅用于展示，
        允许短暂不一致。
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        for memory_id in memory_ids:
            try:
                await self._es.update(
                    index=self._es_memory_index,
                    id=memory_id,
                    script={
                        "source": (
                            "ctx._source.access_count = (ctx._source.access_count ?: 0) + 1; "
                            "ctx._source.last_accessed_at = params.now;"
                        ),
                        "params": {"now": now_iso},
                    },
                )
            except Exception as exc:
                logger.warning(f"[MemoryStore] 更新访问统计失败 id={memory_id}: {exc}")

    async def _embed(self, text: str) -> list[float]:
        if not self._embedding_base_url:
            logger.warning("[MemoryStore] EMBEDDING_BASE_URL 未配置，跳过向量化")
            return []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._embedding_base_url}/v1/embeddings",
                    json={"model": self._embedding_model, "input": text},
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
        except httpx.HTTPStatusError as exc:
            logger.error(f"[MemoryStore] Embedding 调用失败: {exc}，响应体: {exc.response.text[:500]}")
            return []
        except Exception as exc:
            logger.error(f"[MemoryStore] Embedding 调用失败: {exc}")
            return []

    async def _rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not self._reranker_base_url or not candidates:
            return candidates[:top_n]

        documents = [candidate["content"] for candidate in candidates]
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._reranker_base_url}/v1/rerank",
                    json={
                        "model": self._reranker_model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                    },
                )
                response.raise_for_status()
                results = response.json()["results"]

            reranked = []
            for result in results:
                item = candidates[result["index"]].copy()
                item["score"] = result["relevance_score"]
                reranked.append(item)
            return reranked

        except httpx.HTTPStatusError as exc:
            logger.error(f"[MemoryStore] Reranker 调用失败，降级返回 kNN 结果: {exc}，响应体: {exc.response.text[:500]}")
            return candidates[:top_n]
        except Exception as exc:
            logger.error(f"[MemoryStore] Reranker 调用失败，降级返回 kNN 结果: {exc}")
            return candidates[:top_n]


_store: ElasticsearchMemoryStore | None = None


def init_elasticsearch_memory_store(store: ElasticsearchMemoryStore) -> None:
    """应用启动时调用一次（`setup()` 已由调用方完成），注册全局单例。

    供没有依赖注入入口的模块级 `@tool` 函数（`recall_memory`/`save_memory`）
    和 `MemoryContextBuilder` 使用。
    """
    global _store
    _store = store


def get_elasticsearch_memory_store() -> ElasticsearchMemoryStore:
    """返回全局唯一的 ElasticsearchMemoryStore 实例。

    Raises:
        RuntimeError: init_elasticsearch_memory_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("ElasticsearchMemoryStore 尚未初始化，请确认应用已完成启动")
    return _store
