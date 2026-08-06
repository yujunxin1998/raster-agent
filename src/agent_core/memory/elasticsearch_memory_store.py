"""基于 Elasticsearch + Xinference BGE 的向量记忆存储。

原样迁移自 `src/core/memory/vector_store.py`，仅将审计写入从"直接调用模块级
函数"改为"通过 storage 层的 MemoryAuditStore 单例"，并把散落的动作/来源
字符串常量替换为 `src.common.constants` 里的枚举，其余检索/写入/降级逻辑
不变。
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
from elasticsearch import AsyncElasticsearch, NotFoundError
from loguru import logger

from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.common.constants import MemoryAuditAction, MemorySource
from src.storage.memory_audit_store import get_memory_audit_store

_RESULT_SOURCE_FIELDS = [
    "content", "memory_type", "importance", "created_at",
    "status", "expires_at", "last_accessed_at", "access_count",
    "source", "conversation_id", "scope_type", "scope_id", "agent_name",
]

_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "conversation_id": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "memory_type": {"type": "keyword"},
            "importance": {"type": "integer"},
            "created_at": {"type": "date"},
            "status": {"type": "keyword"},
            "expires_at": {"type": "date"},
            "last_accessed_at": {"type": "date"},
            "access_count": {"type": "integer"},
            "superseded_by": {"type": "keyword"},
            "source": {"type": "keyword"},
            "scope_type": {"type": "keyword"},
            "scope_id": {"type": "keyword"},
            "agent_name": {"type": "keyword"},
        }
    }
}


def _to_memory_source(raw: str) -> MemorySource:
    """把自由字符串转换为 MemorySource 枚举，非法值兜底为 API 并记录 warning。"""
    try:
        return MemorySource(raw)
    except ValueError:
        logger.warning(f"[MemoryAudit] 未知的 source={raw!r}，按 api 处理")
        return MemorySource.API


class ElasticsearchMemoryStore(BaseMemoryStore):
    """基于 Elasticsearch + Xinference BGE 的向量记忆存储。"""

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
            sensitive_filter_enabled: 是否在 save/update 前做敏感信息拦截。
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

        ES_URL 未配置时记 warning 并跳过，后续 save/search/list_all 均静默降级。
        """
        if not self._es_url:
            logger.warning("[MemoryStore] ES_URL 未配置，长期记忆功能不可用")
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
            await self._backfill_legacy_status()

        await self._probe_embedding_and_reranker()

    async def close(self) -> None:
        """关闭 ES 连接，应用关闭阶段调用一次。"""
        if self._es:
            await self._es.close()

    async def _backfill_legacy_status(self) -> None:
        """为字段上线前已写入的旧文档回填 status/access_count，避免被检索过滤条件排除。"""
        try:
            response = await self._es.update_by_query(
                index=self._es_memory_index,
                body={
                    "script": {"source": "ctx._source.status = 'active'; ctx._source.access_count = 0;"},
                    "query": {"bool": {"must_not": {"exists": {"field": "status"}}}},
                },
                conflicts="proceed",
            )
            updated = response.get("updated", 0)
            if updated:
                logger.info(f"[MemoryStore] 回填 {updated} 条旧记忆的 status/access_count 字段")
        except Exception as exc:
            logger.warning(f"[MemoryStore] 回填旧记忆 status 字段失败（可忽略，不影响新记忆功能）: {exc}")

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

    # ── 公开接口 ──────────────────────────────────────────────

    async def save(
        self,
        content: str,
        user_id: str = "default",
        conversation_id: Optional[str] = None,
        memory_type: str = "fact",
        importance: int = 5,
        source: str = "tool",
        trace_id: Optional[str] = None,
        status: Optional[str] = None,
        expires_at: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> str:
        if not content:
            raise ValueError("content 不能为空")

        if self._sensitive_filter_enabled and contains_sensitive_info(content):
            logger.warning(f"[MemoryStore] 内容命中敏感信息规则，拒绝保存 user={user_id} source={source}")
            await self._record_audit(
                action=MemoryAuditAction.SAVE_REJECTED, user_id=user_id, source=_to_memory_source(source),
                conversation_id=conversation_id, trace_id=trace_id, detail="命中敏感信息过滤规则",
            )
            return ""

        if not self._es:
            logger.warning("[MemoryStore] ES 未初始化，跳过记忆存储")
            return ""

        embedding = await self._embed(content)
        memory_id = str(uuid.uuid4())

        await self._es.index(
            index=self._es_memory_index,
            id=memory_id,
            document={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "content": content,
                "embedding": embedding,
                "memory_type": memory_type,
                "importance": importance,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": status or "active",
                "expires_at": expires_at,
                "last_accessed_at": None,
                "access_count": 0,
                "superseded_by": None,
                "source": source,
                "scope_type": scope_type or "user",
                "scope_id": scope_id,
                "agent_name": agent_name,
            },
        )
        logger.info(f"[MemoryStore] 保存记忆 id={memory_id} user={user_id} type={memory_type}")
        await self._record_audit(
            action=MemoryAuditAction.CREATE, user_id=user_id, source=_to_memory_source(source),
            memory_id=memory_id, conversation_id=conversation_id, trace_id=trace_id,
            detail=f"type={memory_type} importance={importance} status={status or 'active'}",
        )
        return memory_id

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

    async def list_all(self, user_id: str = "default") -> list[dict]:
        if not self._es:
            return []

        response = await self._es.search(
            index=self._es_memory_index,
            body={
                "query": {"term": {"user_id": user_id}},
                "sort": [{"created_at": {"order": "desc"}}],
                "_source": _RESULT_SOURCE_FIELDS,
                "size": 200,
            },
        )
        return [self._build_response(hit["_id"], hit["_source"]) for hit in response["hits"]["hits"]]

    async def get(self, memory_id: str, user_id: str = "default") -> Optional[dict]:
        if not self._es:
            return None
        try:
            document = await self._es.get(index=self._es_memory_index, id=memory_id)
        except NotFoundError:
            return None
        except Exception as exc:
            logger.error(f"[MemoryStore] 获取记忆失败 id={memory_id}: {exc}")
            raise

        source = document["_source"]
        if source.get("user_id") != user_id:
            return None
        return self._build_response(memory_id, source)

    async def update(
        self,
        memory_id: str,
        user_id: str = "default",
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[int] = None,
        source: str = "api",
        trace_id: Optional[str] = None,
        status: Optional[str] = None,
        expires_at: Optional[str] = None,
        clear_expires_at: bool = False,
        superseded_by: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> bool:
        if not self._es:
            return False

        existing = await self.get(memory_id, user_id)
        if existing is None:
            return False

        if content and self._sensitive_filter_enabled and contains_sensitive_info(content):
            logger.warning(f"[MemoryStore] 更新内容命中敏感信息规则，拒绝更新 id={memory_id} user={user_id}")
            await self._record_audit(
                action=MemoryAuditAction.UPDATE_REJECTED, user_id=user_id, source=_to_memory_source(source),
                memory_id=memory_id, trace_id=trace_id, detail="命中敏感信息过滤规则",
            )
            return False

        patch: dict = {}
        if content is not None:
            patch["content"] = content
            patch["embedding"] = await self._embed(content)
        if memory_type is not None:
            patch["memory_type"] = memory_type
        if importance is not None:
            patch["importance"] = importance
        if status is not None:
            patch["status"] = status
        if clear_expires_at:
            patch["expires_at"] = None
        elif expires_at is not None:
            patch["expires_at"] = expires_at
        if superseded_by is not None:
            patch["superseded_by"] = superseded_by
        if scope_type is not None:
            patch["scope_type"] = scope_type
        if scope_id is not None:
            patch["scope_id"] = scope_id
        if agent_name is not None:
            patch["agent_name"] = agent_name

        if not patch:
            return True

        try:
            await self._es.update(index=self._es_memory_index, id=memory_id, doc=patch)
        except Exception as exc:
            logger.error(f"[MemoryStore] 更新记忆失败 id={memory_id}: {exc}")
            return False

        await self._record_audit(
            action=MemoryAuditAction.UPDATE, user_id=user_id, source=_to_memory_source(source),
            memory_id=memory_id, trace_id=trace_id, detail=f"fields={list(patch.keys() - {'embedding'})}",
        )
        return True

    async def delete(
        self, memory_id: str, user_id: str = "default", source: str = "api", trace_id: Optional[str] = None
    ) -> bool:
        if not self._es:
            return False

        existing = await self.get(memory_id, user_id)
        if existing is None:
            return False

        try:
            await self._es.delete(index=self._es_memory_index, id=memory_id)
        except Exception as exc:
            logger.error(f"[MemoryStore] 删除记忆失败 id={memory_id}: {exc}")
            return False

        await self._record_audit(
            action=MemoryAuditAction.DELETE, user_id=user_id, source=_to_memory_source(source),
            memory_id=memory_id, trace_id=trace_id,
        )
        return True

    async def sweep_stale(self, *, max_age_days: int, low_importance_threshold: int) -> int:
        if not self._es:
            return 0

        query = {
            "bool": {
                "filter": [{"term": {"status": "active"}}],
                "should": [
                    {"range": {"expires_at": {"lt": "now"}}},
                    {
                        "bool": {
                            "filter": [
                                {"range": {"importance": {"lte": low_importance_threshold}}},
                                {"term": {"access_count": 0}},
                                {"range": {"created_at": {"lt": f"now-{max_age_days}d"}}},
                            ]
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

        try:
            response = await self._es.update_by_query(
                index=self._es_memory_index,
                body={"query": query, "script": {"source": "ctx._source.status = 'archived'"}},
                conflicts="proceed",
            )
        except Exception as exc:
            logger.error(f"[MemoryStore] staleness 归档扫描失败: {exc}")
            return 0

        archived = response.get("updated", 0)
        if archived:
            logger.info(f"[MemoryStore] staleness 扫描归档 {archived} 条记忆")
        return archived

    # ── 内部方法 ──────────────────────────────────────────────

    async def _record_audit(
        self,
        action: MemoryAuditAction,
        user_id: str,
        source: MemorySource,
        memory_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """写审计日志，审计存储未初始化时静默跳过（不影响记忆主流程）。"""
        audit_store = get_memory_audit_store()
        if audit_store is None:
            return
        await audit_store.record(
            action=action, user_id=user_id, source=source, memory_id=memory_id,
            conversation_id=conversation_id, trace_id=trace_id, detail=detail,
        )

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
        """把 ES _source 组装为统一的记忆响应 dict。

        单条读取时仍按 `expires_at` 是否已过去派生展示 expired（不等下一次
        `sweep_stale()` 扫描才让用户看到"过期了"），真正把 status 物理写回
        archived 的动作由 `sweep_stale()` 定期批量执行，见该方法说明。
        """
        status = source.get("status") or "active"
        expires_at = source.get("expires_at")
        if status == "active" and expires_at and self._is_expired(expires_at):
            status = "expired"
        return {
            "id": memory_id,
            "content": source["content"],
            "memory_type": source["memory_type"],
            "importance": source["importance"],
            "created_at": source["created_at"],
            "status": status,
            "expires_at": expires_at,
            "last_accessed_at": source.get("last_accessed_at"),
            "access_count": source.get("access_count") or 0,
            "source": source.get("source"),
            "conversation_id": source.get("conversation_id"),
            "scope_type": source.get("scope_type") or "user",
            "scope_id": source.get("scope_id"),
            "agent_name": source.get("agent_name"),
        }

    async def _bump_access(self, memory_ids: list[str]) -> None:
        """命中记忆后异步更新访问统计，失败只记 warning，不影响检索主流程。"""
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
