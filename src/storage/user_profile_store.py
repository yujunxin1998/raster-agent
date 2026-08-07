"""持久化用户长期画像与时间线（三层记忆架构的 L1/L2）。

这两层是"随用户使用持续演进的整段摘要"（画像：用户是谁；时间线：用户做过
什么），不参与语义检索，只按 `user_id` 整体取出/整体覆盖——跟需要 kNN 向量
检索的 Facts（L3，走 `ElasticsearchMemoryStore`）性质完全不同，不适合塞进
向量索引，改用项目已有的 Postgres DAO 套路（对齐 `conversation_store.py`
"类 + 显式注入 pool + `CREATE TABLE IF NOT EXISTS`"的写法），一个用户一行，
`upsert` 整体覆盖更新。
"""
from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger


class UserProfileStore:
    """`user_memory_profile` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory_profile (
                user_id              TEXT        PRIMARY KEY,
                work_context         TEXT        NOT NULL DEFAULT '',
                personal_context     TEXT        NOT NULL DEFAULT '',
                top_of_mind          TEXT        NOT NULL DEFAULT '',
                recent_months        TEXT        NOT NULL DEFAULT '',
                earlier_context      TEXT        NOT NULL DEFAULT '',
                long_term_background TEXT        NOT NULL DEFAULT '',
                updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        logger.info("[UserProfileStore] 初始化完成")

    async def get(self, user_id: str) -> Optional[dict]:
        """查询某用户的画像与时间线。

        Args:
            user_id: 归属用户 ID。

        Returns:
            六个字段 + `updated_at` 的字典；用户还没生成过画像时返回 None
            （这是新用户的正常状态，不是异常）。
        """
        row = await self._pool.fetchrow(
            "SELECT work_context, personal_context, top_of_mind, "
            "recent_months, earlier_context, long_term_background, updated_at "
            "FROM user_memory_profile WHERE user_id = $1",
            user_id,
        )
        return dict(row) if row else None

    async def upsert(
        self,
        user_id: str,
        *,
        work_context: str,
        personal_context: str,
        top_of_mind: str,
        recent_months: str,
        earlier_context: str,
        long_term_background: str,
    ) -> None:
        """整体覆盖写入某用户的画像与时间线（不存在则新建）。

        Args:
            user_id: 归属用户 ID。
            work_context: 职业角色、公司、关键项目、主力技术栈。
            personal_context: 语言能力、沟通偏好、兴趣领域。
            top_of_mind: 当前关注的多个并行焦点，更新频率最高。
            recent_months: 近 1-3 个月的详细活动摘要。
            earlier_context: 3-12 个月前的重要模式。
            long_term_background: 长期不变的基础背景。
        """
        await self._pool.execute(
            """
            INSERT INTO user_memory_profile (
                user_id, work_context, personal_context, top_of_mind,
                recent_months, earlier_context, long_term_background, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                work_context = EXCLUDED.work_context,
                personal_context = EXCLUDED.personal_context,
                top_of_mind = EXCLUDED.top_of_mind,
                recent_months = EXCLUDED.recent_months,
                earlier_context = EXCLUDED.earlier_context,
                long_term_background = EXCLUDED.long_term_background,
                updated_at = NOW()
            """,
            user_id, work_context, personal_context, top_of_mind,
            recent_months, earlier_context, long_term_background,
        )


_store: UserProfileStore | None = None


async def init_user_profile_store(pool: asyncpg.Pool) -> UserProfileStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = UserProfileStore(pool)
    await _store.setup()
    return _store


def get_user_profile_store() -> UserProfileStore:
    """返回全局唯一的 UserProfileStore 实例。

    Raises:
        RuntimeError: init_user_profile_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("UserProfileStore 尚未初始化，请确认应用已完成启动")
    return _store
