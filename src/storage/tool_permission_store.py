"""按用户持久化工具级权限拒绝状态（新增，供 Guardrail 权限控制模块使用）。

设计与 `SkillSettingsStore` 保持一致的"默认全放行、只存显式拒绝"语义：
`user_skill_settings` 表管理的是"技能"这一类工具（挂载在各专用 Agent 上、
面向最终用户可见可关闭），本表管理的是更广义的"工具级权限"——覆盖非技能类
工具（如未来的代码执行、文件删除等高风险沙箱工具），二者是互补关系，
`AllowlistGuardrailProvider` 会同时查询这两张表（详见
`src/agent_core/guardrail/allowlist_guardrail_provider.py`）。

`scope` 字段预留用于未来按会话/按角色等更细粒度的授权维度，当前统一传
"default"。
"""
from __future__ import annotations

import asyncpg
from loguru import logger

_DEFAULT_SCOPE = "default"


class ToolPermissionStore:
    """`user_tool_permissions` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_tool_permissions (
                user_id     TEXT        NOT NULL,
                tool_name   TEXT        NOT NULL,
                scope       TEXT        NOT NULL DEFAULT 'default',
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, tool_name, scope)
            )
            """
        )
        logger.info("[ToolPermissionStore] 初始化完成")

    async def is_denied(self, user_id: str, tool_name: str, scope: str = _DEFAULT_SCOPE) -> bool:
        """判断某个工具在指定 scope 下是否被该用户显式拒绝。

        Args:
            user_id: 用户 ID。
            tool_name: 工具名。
            scope: 授权维度，默认 "default"。

        Returns:
            True 表示已被显式拒绝；未出现在表中（默认放行）返回 False。
        """
        if not user_id or not tool_name:
            return False
        row = await self._pool.fetchrow(
            "SELECT 1 FROM user_tool_permissions WHERE user_id = $1 AND tool_name = $2 AND scope = $3",
            user_id, tool_name, scope,
        )
        return row is not None

    async def set_granted(
        self, user_id: str, tool_name: str, granted: bool, scope: str = _DEFAULT_SCOPE
    ) -> None:
        """设置某个工具在指定 scope 下对该用户的授权状态。

        Args:
            user_id: 用户 ID，不能为空。
            tool_name: 工具名，不能为空。
            granted: True 时删除拒绝记录（恢复默认放行）；False 时插入拒绝记录。
            scope: 授权维度，默认 "default"。

        Raises:
            ValueError: user_id 或 tool_name 为空。
        """
        if not user_id or not tool_name:
            raise ValueError("user_id 和 tool_name 均不能为空")

        if granted:
            await self._pool.execute(
                "DELETE FROM user_tool_permissions WHERE user_id = $1 AND tool_name = $2 AND scope = $3",
                user_id, tool_name, scope,
            )
        else:
            await self._pool.execute(
                """
                INSERT INTO user_tool_permissions (user_id, tool_name, scope)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, tool_name, scope) DO UPDATE SET updated_at = NOW()
                """,
                user_id, tool_name, scope,
            )

    async def list_denied_tools(self, user_id: str, scope: str = _DEFAULT_SCOPE) -> set[str]:
        """返回该用户在指定 scope 下已被拒绝的全部工具名。

        Args:
            user_id: 用户 ID。
            scope: 授权维度，默认 "default"。

        Returns:
            已拒绝的 tool_name 集合。
        """
        if not user_id:
            raise ValueError("user_id 不能为空")
        rows = await self._pool.fetch(
            "SELECT tool_name FROM user_tool_permissions WHERE user_id = $1 AND scope = $2",
            user_id, scope,
        )
        return {row["tool_name"] for row in rows}


_store: ToolPermissionStore | None = None


async def init_tool_permission_store(pool: asyncpg.Pool) -> ToolPermissionStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = ToolPermissionStore(pool)
    await _store.setup()
    return _store


def get_tool_permission_store() -> ToolPermissionStore:
    """返回全局唯一的 ToolPermissionStore 实例。

    Raises:
        RuntimeError: init_tool_permission_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("ToolPermissionStore 尚未初始化，请确认应用已完成启动")
    return _store
