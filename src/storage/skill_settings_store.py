"""按用户持久化技能（skill）启用/禁用状态。

只存储被用户显式关闭的技能（一行 = 一次"禁用"）；未出现在表中的
(user_id, tool_name) 组合视为默认启用，这样大多数用户（从不关闭任何技能）
不会产生任何行，符合"开关默认全开"的产品语义。原样迁移自
diit-agent-server 的 `src/storage/skill_settings_store.py`，仅将模块级
函数改造为类方法。
"""
from __future__ import annotations

import asyncpg
from loguru import logger


class SkillSettingsStore:
    """`user_skill_settings` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_skill_settings (
                user_id     TEXT        NOT NULL,
                tool_name   TEXT        NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, tool_name)
            )
            """
        )
        logger.info("[SkillSettingsStore] 初始化完成")

    async def get_disabled_tools(self, user_id: str) -> set[str]:
        """返回该用户已禁用的全部 tool_name。

        Args:
            user_id: 用户 ID，不能为空。

        Returns:
            已禁用的 tool_name 集合，从未禁用过任何技能时返回空集合。
        """
        if not user_id:
            raise ValueError("user_id 不能为空")
        rows = await self._pool.fetch(
            "SELECT tool_name FROM user_skill_settings WHERE user_id = $1", user_id
        )
        return {row["tool_name"] for row in rows}

    async def is_disabled(self, user_id: str, tool_name: str) -> bool:
        """判断某个技能是否被该用户显式禁用。

        Args:
            user_id: 用户 ID。
            tool_name: 技能对应的工具名。

        Returns:
            True 表示已被禁用；未出现在表中（默认启用）返回 False。
        """
        if not user_id or not tool_name:
            return False
        row = await self._pool.fetchrow(
            "SELECT 1 FROM user_skill_settings WHERE user_id = $1 AND tool_name = $2",
            user_id, tool_name,
        )
        return row is not None

    async def set_enabled(self, user_id: str, tool_name: str, enabled: bool) -> None:
        """切换某个技能对该用户的启用状态。

        Args:
            user_id: 用户 ID，不能为空。
            tool_name: 技能对应的工具名，不能为空。
            enabled: True 时删除禁用记录（恢复默认启用）；False 时插入禁用记录。

        Raises:
            ValueError: user_id 或 tool_name 为空。
        """
        if not user_id or not tool_name:
            raise ValueError("user_id 和 tool_name 均不能为空")

        if enabled:
            await self._pool.execute(
                "DELETE FROM user_skill_settings WHERE user_id = $1 AND tool_name = $2",
                user_id, tool_name,
            )
        else:
            await self._pool.execute(
                """
                INSERT INTO user_skill_settings (user_id, tool_name)
                VALUES ($1, $2)
                ON CONFLICT (user_id, tool_name) DO UPDATE SET updated_at = NOW()
                """,
                user_id, tool_name,
            )


_store: SkillSettingsStore | None = None


async def init_skill_settings_store(pool: asyncpg.Pool) -> SkillSettingsStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = SkillSettingsStore(pool)
    await _store.setup()
    return _store


def get_skill_settings_store() -> SkillSettingsStore:
    """返回全局唯一的 SkillSettingsStore 实例。

    Raises:
        RuntimeError: init_skill_settings_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("SkillSettingsStore 尚未初始化，请确认应用已完成启动")
    return _store
