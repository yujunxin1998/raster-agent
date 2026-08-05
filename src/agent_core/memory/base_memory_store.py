"""长期记忆存储抽象接口。

原样迁移自 `src/core/memory/base.py`：上层（提取器、检索注入、Agent 工具、
管理 API）只依赖这个抽象接口，不依赖具体实现，存储后端可以自由替换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseMemoryStore(ABC):
    """长期记忆存储抽象接口。"""

    @abstractmethod
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
        """保存一条记忆。

        Args:
            content: 记忆正文。
            user_id: 归属用户 ID。
            conversation_id: 产生该记忆的会话 ID。
            memory_type: 记忆分类，见 `src.common.constants.MemoryType`。
            importance: 重要度 1-10。
            source: 记忆来源，用于审计：extractor（后处理提取器）/ tool（Agent 主动调用）/
                api（管理 API）/ compressor（压缩摘要）。
            trace_id: 链路追踪 ID。
            status: 记忆状态，默认为 active；提取器对低置信度结果可传 pending，
                待用户在管理 API 中确认后改为 active。
            expires_at: 过期时间（ISO 字符串），过期后不再被召回。
            scope_type: 记忆作用域，预留字段，默认 user，当前不参与召回过滤。
            scope_id: 作用域 ID，预留字段。
            agent_name: 专属 Agent 名，预留字段。

        Returns:
            新记忆的 ID；因敏感信息拦截等原因未实际写入时返回空字符串。
        """

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
            memory_type: 不为空时只在该类型内检索（用于冲突检测场景）。

        Returns:
            记忆列表，每条包含 `{id, content, memory_type, importance, created_at,
            score, status, ...}`。只召回 status=active 且未过期的记忆。
        """

    @abstractmethod
    async def list_all(self, user_id: str = "default") -> list[dict]:
        """列出某用户的全部记忆（含 active/pending/archived，已过期的派生展示为 expired）。

        Args:
            user_id: 归属用户 ID。

        Returns:
            按创建时间倒序排列的记忆列表。
        """

    @abstractmethod
    async def get(self, memory_id: str, user_id: str = "default") -> Optional[dict]:
        """按 ID 获取指定用户的一条记忆。

        Args:
            memory_id: 记忆 ID。
            user_id: 归属用户 ID，用于校验记忆归属。

        Returns:
            记忆详情；不存在或不属于该用户时返回 None。
        """

    @abstractmethod
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
        """更新指定用户的一条记忆，content 变化时需重新生成向量。

        Args:
            memory_id: 记忆 ID。
            user_id: 归属用户 ID，用于校验记忆归属。
            content: 新内容，为空则不修改。
            memory_type: 新分类，为空则不修改。
            importance: 新重要度，为空则不修改。
            source: 触发来源，用于审计。
            trace_id: 链路追踪 ID。
            status: 新状态，为空则不修改。
            expires_at: 新过期时间；`expires_at=None` 表示不修改。
            clear_expires_at: 显式清除已设置的过期时间时传 True（`expires_at`
                用 None 同时承担"不修改"和"清除"两种语义会有歧义，故拆成独立参数）。
            superseded_by: 该记忆被哪条新记忆取代（冲突检测场景使用）。
            scope_type: 作用域类型，预留字段。
            scope_id: 作用域 ID，预留字段。
            agent_name: 专属 Agent 名，预留字段。

        Returns:
            是否更新成功。
        """

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        user_id: str = "default",
        source: str = "api",
        trace_id: Optional[str] = None,
    ) -> bool:
        """按 ID 删除指定用户的记忆。

        Args:
            memory_id: 记忆 ID。
            user_id: 归属用户 ID，用于校验记忆归属。
            source: 触发来源，用于审计。
            trace_id: 链路追踪 ID。

        Returns:
            是否删除成功。
        """
