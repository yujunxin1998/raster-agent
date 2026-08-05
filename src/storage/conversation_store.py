"""持久化会话（conversation）元数据。

原样迁移自 `diit-agent-server` 的 `src/storage/conversation_store.py`，改造为类
（原实现是模块级函数 + 模块级连接池，本仓库统一走 `SkillSettingsStore` 等已有
Store 的"类 + 显式注入 pool"约定）。表结构、字段语义未改动。
"""
from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger

_DEFAULT_TITLE = "新对话"


class ConversationStore:
    """`conversations` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT        PRIMARY KEY,
                user_id     TEXT        NOT NULL,
                title       TEXT        NOT NULL DEFAULT '新对话',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC)"
        )
        logger.info("[ConversationStore] 初始化完成")

    async def create(self, conversation_id: str, user_id: str, title: str = _DEFAULT_TITLE) -> dict:
        """新建一条会话记录。

        Args:
            conversation_id: 会话 ID（同时是 checkpointer 的 thread_id）。
            user_id: 归属用户 ID。
            title: 会话标题，默认"新对话"。

        Returns:
            新建的会话记录（含全部列）。
        """
        row = await self._pool.fetchrow(
            """
            INSERT INTO conversations (id, user_id, title)
            VALUES ($1, $2, $3)
            RETURNING id, user_id, title, created_at, updated_at
            """,
            conversation_id, user_id, title,
        )
        return dict(row)

    async def get(self, conversation_id: str, user_id: str) -> Optional[dict]:
        """按 (id, user_id) 查询会话，owner 校验内置在查询条件里。

        Args:
            conversation_id: 会话 ID。
            user_id: 归属用户 ID。

        Returns:
            会话记录；不存在或不属于该用户时返回 None。
        """
        row = await self._pool.fetchrow(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = $1 AND user_id = $2",
            conversation_id, user_id,
        )
        return dict(row) if row else None

    async def list_by_user(self, user_id: str) -> list[dict]:
        """按用户列出全部会话，按 updated_at 倒序。

        Args:
            user_id: 归属用户 ID。

        Returns:
            会话记录列表。
        """
        rows = await self._pool.fetch(
            "SELECT id, user_id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = $1 ORDER BY updated_at DESC",
            user_id,
        )
        return [dict(row) for row in rows]

    async def update_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """更新会话标题。

        Args:
            conversation_id: 会话 ID。
            user_id: 归属用户 ID（owner 校验）。
            title: 新标题。

        Returns:
            是否成功更新（会话不存在或不属于该用户时返回 False）。
        """
        result = await self._pool.execute(
            "UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2 AND user_id = $3",
            title, conversation_id, user_id,
        )
        return result == "UPDATE 1"

    async def touch(self, conversation_id: str) -> None:
        """更新会话的 `updated_at`（不做 owner 校验，供内部调用链使用）。

        Args:
            conversation_id: 会话 ID。
        """
        await self._pool.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = $1", conversation_id
        )

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        """删除一条会话记录。

        Args:
            conversation_id: 会话 ID。
            user_id: 归属用户 ID（owner 校验）。

        Returns:
            是否成功删除。
        """
        result = await self._pool.execute(
            "DELETE FROM conversations WHERE id = $1 AND user_id = $2", conversation_id, user_id
        )
        return result == "DELETE 1"

    async def exists(self, conversation_id: str) -> bool:
        """判断会话是否存在（不做 owner 校验）。

        Args:
            conversation_id: 会话 ID。

        Returns:
            是否存在。
        """
        row = await self._pool.fetchrow("SELECT 1 FROM conversations WHERE id = $1", conversation_id)
        return row is not None


_store: ConversationStore | None = None


async def init_conversation_store(pool: asyncpg.Pool) -> ConversationStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = ConversationStore(pool)
    await _store.setup()
    return _store


def get_conversation_store() -> ConversationStore:
    """返回全局唯一的 ConversationStore 实例。

    Raises:
        RuntimeError: init_conversation_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("ConversationStore 尚未初始化，请确认应用已完成启动")
    return _store
