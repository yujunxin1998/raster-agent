"""记忆操作审计日志。

审计是辅助能力，未初始化或写入失败时只记 warning 并静默降级，不像
`SkillSettingsStore`/`ToolPermissionStore` 等核心配置存储那样向上抛异常——
审计缺失不应影响记忆主流程的可用性，这一容错哲学与原项目保持一致。
"""
from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger

from src.common.constants import MemoryAuditAction, MemorySource


class MemoryAuditStore:
    """`memory_audit_logs` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表 + 索引，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_audit_logs (
                id              BIGSERIAL   PRIMARY KEY,
                memory_id       TEXT,
                user_id         TEXT        NOT NULL,
                conversation_id TEXT,
                trace_id        TEXT,
                action          TEXT        NOT NULL,
                source          TEXT        NOT NULL,
                detail          TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_user ON memory_audit_logs(user_id, created_at DESC)"
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_memory ON memory_audit_logs(memory_id)"
        )
        logger.info("[MemoryAuditStore] 初始化完成")

    async def record(
        self,
        action: MemoryAuditAction,
        user_id: str,
        source: MemorySource,
        memory_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        """记录一次记忆操作审计日志。

        Args:
            action: 动作类型（create/update/delete/recall/inject/compress/...）。
            user_id: 操作归属用户 ID。
            source: 触发来源（extractor/tool/api/compressor）。
            memory_id: 涉及的记忆 ID，批量场景可为空。
            conversation_id: 触发该操作的会话 ID。
            trace_id: 链路追踪 ID，便于跨系统关联日志。
            detail: 附加说明文本。

        Note:
            失败时只记 warning，不向上抛异常，不影响记忆主流程。
        """
        try:
            await self._pool.execute(
                """
                INSERT INTO memory_audit_logs
                    (memory_id, user_id, conversation_id, trace_id, action, source, detail)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                memory_id, user_id, conversation_id, trace_id,
                action.value, source.value, detail,
            )
        except Exception as exc:
            logger.warning(f"[MemoryAuditStore] 写入审计日志失败 action={action} memory_id={memory_id}: {exc}")

    async def list_by_user(self, user_id: str, limit: int = 200) -> list[dict]:
        """按用户查询审计日志，按时间倒序。

        Args:
            user_id: 用户 ID。
            limit: 最多返回条数。

        Returns:
            审计日志记录列表，查询失败时返回空列表。
        """
        try:
            rows = await self._pool.fetch(
                """
                SELECT id, memory_id, user_id, conversation_id, trace_id, action, source, detail, created_at
                FROM memory_audit_logs
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )
        except Exception as exc:
            logger.warning(f"[MemoryAuditStore] 查询审计日志失败 user_id={user_id}: {exc}")
            return []
        return [dict(row) for row in rows]

    async def list_by_memory(self, memory_id: str) -> list[dict]:
        """按记忆 ID 查询其全部审计日志，按时间倒序。

        Args:
            memory_id: 记忆 ID。

        Returns:
            审计日志记录列表，查询失败时返回空列表。
        """
        try:
            rows = await self._pool.fetch(
                """
                SELECT id, memory_id, user_id, conversation_id, trace_id, action, source, detail, created_at
                FROM memory_audit_logs
                WHERE memory_id = $1
                ORDER BY created_at DESC
                """,
                memory_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryAuditStore] 查询审计日志失败 memory_id={memory_id}: {exc}")
            return []
        return [dict(row) for row in rows]


_store: MemoryAuditStore | None = None


async def init_memory_audit_store(pool: asyncpg.Pool) -> MemoryAuditStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = MemoryAuditStore(pool)
    await _store.setup()
    return _store


def get_memory_audit_store() -> Optional[MemoryAuditStore]:
    """返回全局唯一的 MemoryAuditStore 实例。

    与其它 Store 不同：未初始化时返回 None 而不是抛异常，因为审计是辅助能力，
    调用方（memory_manager）应当在拿到 None 时静默跳过审计，而不是让审计
    缺失中断记忆主流程。
    """
    return _store
