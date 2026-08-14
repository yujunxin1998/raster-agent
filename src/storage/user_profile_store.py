"""持久化用户长期画像与时间线（三层记忆架构的 L1/L2）。

Memory v2：整行覆盖的 `upsert()` 已替换为 `apply_patches()`——按 `revision`
做乐观并发控制，只有调用方持有的快照版本号与当前一致才会写入成功，避免两个
并发的 MemoryUpdateWorker 任务互相覆盖对方的更新（旧版本问题，见
docs/raster-agent长期记忆重构设计.md §2.1）。`user_memory_profile_field_meta`
记录每个字段最近一次更新的来源事件与置信度，供管理 API 展示"这条画像信息
是从哪句话推断出来的"。
"""
from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger

_PROFILE_FIELDS = (
    "work_context", "personal_context", "top_of_mind",
    "recent_months", "earlier_context", "long_term_background",
)
_EMPTY_PROFILE = {field: "" for field in _PROFILE_FIELDS}


class UserProfileStore:
    """`user_memory_profile` / `user_memory_profile_field_meta` 表的读写封装。"""

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
                revision             BIGINT      NOT NULL DEFAULT 0,
                updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # 本仓库没有 Alembic，历史库缺列时按幂等 ALTER 补齐（见 main.py 接线说明）。
        await self._pool.execute(
            "ALTER TABLE user_memory_profile ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 0"
        )
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory_profile_field_meta (
                user_id         TEXT        NOT NULL,
                field_name      TEXT        NOT NULL,
                updated_at      TIMESTAMPTZ NOT NULL,
                source_event_id TEXT,
                confidence      NUMERIC(3,2) NOT NULL DEFAULT 1.0,
                PRIMARY KEY (user_id, field_name)
            )
            """
        )
        logger.info("[UserProfileStore] 初始化完成")

    async def get(self, user_id: str) -> Optional[dict]:
        """查询某用户的画像与时间线（外部 API 契约，字段集合不变）。

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

    async def get_with_revision(self, user_id: str) -> dict:
        """供 MemoryUpdateWorker 读取快照使用：始终返回一行，不存在时 revision=0。

        Args:
            user_id: 归属用户 ID。

        Returns:
            六个字段 + `revision` 的字典；新用户返回全空字段、`revision=0`。
        """
        row = await self._pool.fetchrow(
            "SELECT work_context, personal_context, top_of_mind, "
            "recent_months, earlier_context, long_term_background, revision "
            "FROM user_memory_profile WHERE user_id = $1",
            user_id,
        )
        if row is None:
            return {**_EMPTY_PROFILE, "revision": 0}
        return dict(row)

    async def apply_patches(
        self,
        user_id: str,
        patches: list[dict],
        expected_revision: int,
        source_event_id: Optional[str] = None,
    ) -> bool:
        """按乐观锁把一批 Profile Patch 应用到指定用户的画像/时间线。

        `patches` 里每个 patch 至少含 `field`（六个合法字段之一）、
        `op`（`set`/`merge`/`clear`）；`set`/`merge` 需要 `value`（模型已经把
        旧内容与新信息融合成完整字段文本，两种 op 在存储层都是整字段赋值，
        区别只在于语义审计——"merge"代表模型主动保留了旧内容）。`clear`
        忽略 `value`，把字段清空。

        Args:
            user_id: 归属用户 ID。
            patches: Patch 列表，每项为 dict，可选 `confidence`（用于写
                field_meta，默认 1.0）。
            expected_revision: 调用方读取快照时看到的 revision；新用户传 0。
            source_event_id: 触发本次更新的 memory_event ID，写入 field_meta。

        Returns:
            是否写入成功；`expected_revision` 与当前不一致（并发冲突）或新用户
            路径被抢先创建时返回 False，调用方应重新读取快照后决定重试或重放。
        """
        if not patches:
            return True

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT work_context, personal_context, top_of_mind, "
                    "recent_months, earlier_context, long_term_background, revision "
                    "FROM user_memory_profile WHERE user_id = $1 FOR UPDATE",
                    user_id,
                )
                if row is None:
                    if expected_revision != 0:
                        return False
                    current = dict(_EMPTY_PROFILE)
                else:
                    if row["revision"] != expected_revision:
                        return False
                    current = {field: row[field] for field in _PROFILE_FIELDS}

                updated = dict(current)
                for patch in patches:
                    field = patch["field"]
                    if field not in _PROFILE_FIELDS:
                        continue
                    op = patch["op"]
                    if op == "clear":
                        updated[field] = ""
                    else:  # set / merge：模型已产出完整字段文本，存储层不区分处理
                        updated[field] = str(patch.get("value", ""))

                if row is None:
                    await conn.execute(
                        """
                        INSERT INTO user_memory_profile (
                            user_id, work_context, personal_context, top_of_mind,
                            recent_months, earlier_context, long_term_background,
                            revision, updated_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 1, NOW())
                        """,
                        user_id, *(updated[field] for field in _PROFILE_FIELDS),
                    )
                else:
                    # revision 已经在上面的 SELECT ... FOR UPDATE 内校验过，且行锁
                    # 持续到事务提交，这里的 UPDATE 不会再被并发写入抢先。
                    await conn.execute(
                        """
                        UPDATE user_memory_profile SET
                            work_context = $2, personal_context = $3, top_of_mind = $4,
                            recent_months = $5, earlier_context = $6, long_term_background = $7,
                            revision = revision + 1, updated_at = NOW()
                        WHERE user_id = $1 AND revision = $8
                        """,
                        user_id, *(updated[field] for field in _PROFILE_FIELDS), expected_revision,
                    )

                for patch in patches:
                    field = patch["field"]
                    if field not in _PROFILE_FIELDS:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO user_memory_profile_field_meta
                            (user_id, field_name, updated_at, source_event_id, confidence)
                        VALUES ($1, $2, NOW(), $3, $4)
                        ON CONFLICT (user_id, field_name) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            source_event_id = EXCLUDED.source_event_id,
                            confidence = EXCLUDED.confidence
                        """,
                        user_id, field, source_event_id, float(patch.get("confidence", 1.0)),
                    )
        return True

    async def get_field_sources(self, user_id: str) -> list[dict]:
        """查询某用户各画像字段的最近更新来源，供管理 API 展示。

        Args:
            user_id: 归属用户 ID。

        Returns:
            按字段名的来源记录列表，查询失败或用户无记录时返回空列表。
        """
        rows = await self._pool.fetch(
            "SELECT field_name, updated_at, source_event_id, confidence "
            "FROM user_memory_profile_field_meta WHERE user_id = $1",
            user_id,
        )
        return [dict(row) for row in rows]

    async def delete(self, user_id: str) -> None:
        """彻底删除某用户的画像/时间线与字段来源记录（设计文档 §9.2"彻底删除用户记忆"）。

        Args:
            user_id: 归属用户 ID。
        """
        await self._pool.execute("DELETE FROM user_memory_profile WHERE user_id = $1", user_id)
        await self._pool.execute("DELETE FROM user_memory_profile_field_meta WHERE user_id = $1", user_id)


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
