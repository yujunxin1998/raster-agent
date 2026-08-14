"""长期记忆检索存储抽象接口。

Memory v2：`user_memory_fact`（PostgreSQL）已经是 Facts 的规范化主存
（`src/storage/user_memory_fact_store.py`），写入/更新/删除/生命周期管理都
直接对着它走，不再需要一个跨存储后端抽象的写路径。这里只保留"语义检索"这
一件事的抽象——`MemoryManager.get_relevant_context()` 和 `recall_memory`
工具只依赖 `search()`，检索后端（当前是 Elasticsearch kNN + Reranker）理论上
仍然可以替换，所以还是拆出接口，但不再假装它是通用的记忆 CRUD 抽象。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseMemoryStore(ABC):
    """长期记忆语义检索的抽象接口。"""

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str = "default",
        candidate_k: int = 20,
        top_n: int = 5,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """两阶段检索：kNN 粗召回 candidate_k 条，Reranker 精排后返回 top_n 条。

        Args:
            query: 检索语句。
            user_id: 归属用户 ID，检索结果按此过滤。
            candidate_k: 粗召回候选数量。
            top_n: 精排后最终返回条数。
            memory_type: 不为空时只在该类型内检索。

        Returns:
            记忆列表，每条包含 `{id, content, memory_type, importance, created_at,
            score, status, ...}`。只召回 status=active 且未过期的记忆——这份
            "active"投影由 `FactProjector` 异步维护，不保证与 PostgreSQL 强一致
            （最终一致，见设计文档 §7.6）。
        """
