"""记忆更新任务队列（Memory v2 更新流水线的持久化作业，替代旧的 `memory_jobs`）。

对应设计文档 §4.2/§7.2：状态机 `pending -> processing -> succeeded`，失败按
`next_retry_at` 退避重试直到达到 `max_attempts` 转 `dead`。可靠性完全由这张表 +
`FOR UPDATE SKIP LOCKED` 提供，不依赖进程内存——进程崩溃后 `processing` 状态的
任务靠 `locked_until` 租约超时被其它 Worker 自动回收（见 `reclaim_expired_leases`），
不需要单独的"启动时扫描恢复"逻辑（旧 `MemoryJobsStore.list_recoverable` 的问题：
从未被任何调用方使用过，属于死代码，本次一并移除）。

同一 `(user_id, conversation_id, agent_name)` 在防抖窗口内产生的相邻事件会合并
进同一个 pending 任务（`extend_pending_job`），避免连续对话逐轮触发 LLM 调用。
"""
from __future__ import annotations

import uuid
from typing import Optional

import asyncpg
from loguru import logger

from src.common.constants import MemoryUpdateJobStatus


class MemoryUpdateJobStore:
    """`memory_update_job` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表 + 索引，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_update_job (
                job_id           UUID        PRIMARY KEY,
                user_id          TEXT        NOT NULL,
                conversation_id  TEXT        NOT NULL,
                agent_name       TEXT,
                first_event_id   UUID        NOT NULL,
                last_event_id    UUID        NOT NULL,
                idempotency_key  TEXT        NOT NULL UNIQUE,
                trace_id         TEXT,
                status           TEXT        NOT NULL,
                attempts         INT         NOT NULL DEFAULT 0,
                next_retry_at    TIMESTAMPTZ,
                locked_until     TIMESTAMPTZ,
                last_error       TEXT,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at     TIMESTAMPTZ
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_update_job_claim "
            "ON memory_update_job(status, next_retry_at, created_at)"
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_update_job_pending_merge "
            "ON memory_update_job(user_id, conversation_id, agent_name, status, created_at)"
        )
        logger.info("[MemoryUpdateJobStore] 初始化完成")

    async def create_job(
        self,
        conn: asyncpg.Connection,
        *,
        user_id: str,
        conversation_id: str,
        agent_name: Optional[str],
        first_event_id: str,
        last_event_id: str,
        trace_id: Optional[str] = None,
        debounce_seconds: int = 30,
    ) -> str:
        """在调用方已开启的事务里创建/合并一个更新任务。

        同一 `(user_id, conversation_id, agent_name)` 若存在一个仍在防抖窗口内
        （`created_at` 距现在不超过 `debounce_seconds`）的 `pending` 任务，直接把
        新事件合并进该任务（推进 `last_event_id`、刷新 `created_at` 重新计时），
        不新建任务——这是"连续对话合并、不逐轮调 LLM"的具体实现。

        Args:
            conn: 调用方已获取并开启事务的连接（须与写 `memory_event` 用同一个）。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            agent_name: 产生该事件的 Agent 名，可为空。
            first_event_id: 新事件 ID；合并场景下只有全新任务才会用到此值。
            last_event_id: 新事件 ID（与 first_event_id 相同，一次只捕获一个事件）。
            trace_id: 链路追踪 ID。
            debounce_seconds: 防抖窗口秒数。

        Returns:
            合并到的既有任务 ID，或新建任务的 ID。
        """
        merged_job_id = await conn.fetchval(
            """
            UPDATE memory_update_job
            SET last_event_id = $1, created_at = NOW(), trace_id = COALESCE($2, trace_id)
            WHERE user_id = $3 AND conversation_id = $4
              AND COALESCE(agent_name, '') = COALESCE($5, '')
              AND status = $6
              AND created_at > NOW() - make_interval(secs => $7)
            RETURNING job_id
            """,
            last_event_id, trace_id, user_id, conversation_id, agent_name,
            MemoryUpdateJobStatus.PENDING.value, debounce_seconds,
        )
        if merged_job_id is not None:
            return str(merged_job_id)

        job_id = str(uuid.uuid4())
        idempotency_key = f"{user_id}:{conversation_id}:{agent_name or ''}:{first_event_id}"
        await conn.execute(
            """
            INSERT INTO memory_update_job (
                job_id, user_id, conversation_id, agent_name,
                first_event_id, last_event_id, idempotency_key, trace_id, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            job_id, user_id, conversation_id, agent_name,
            first_event_id, last_event_id, idempotency_key, trace_id,
            MemoryUpdateJobStatus.PENDING.value,
        )
        return job_id

    async def claim_next(self, *, debounce_seconds: int, lease_seconds: int) -> Optional[dict]:
        """原子性地认领一个可处理的任务（文档 §7.2 的 `FOR UPDATE SKIP LOCKED`）。

        用可写 CTE 把"挑选 + 加锁 + 置为 processing"合并成一条语句，天然原子，
        不需要调用方额外开事务。

        Args:
            debounce_seconds: 只认领 `created_at` 超过这个窗口的任务（给同一任务
                留出被后续事件合并的时间）。
            lease_seconds: 租约时长；超过这个时长仍未 `mark_succeeded`/`mark_retry`
                的任务会被 `reclaim_expired_leases()` 收回重新排队。

        Returns:
            认领到的任务记录（`job_id/user_id/conversation_id/agent_name/
            first_event_id/last_event_id/attempts`）；没有可处理任务时返回 None。
        """
        row = await self._pool.fetchrow(
            """
            WITH claimed AS (
                SELECT job_id FROM memory_update_job
                WHERE status = $1
                  AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                  AND created_at <= NOW() - make_interval(secs => $2)
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE memory_update_job job
            SET status = $3, locked_until = NOW() + make_interval(secs => $4)
            FROM claimed
            WHERE job.job_id = claimed.job_id
            RETURNING job.job_id, job.user_id, job.conversation_id, job.agent_name,
                      job.first_event_id, job.last_event_id, job.trace_id, job.attempts
            """,
            MemoryUpdateJobStatus.PENDING.value, debounce_seconds,
            MemoryUpdateJobStatus.PROCESSING.value, lease_seconds,
        )
        return dict(row) if row else None

    async def mark_succeeded(self, job_id: str) -> None:
        """把任务标记为成功终态。"""
        await self._pool.execute(
            "UPDATE memory_update_job SET status = $2, completed_at = NOW() WHERE job_id = $1",
            job_id, MemoryUpdateJobStatus.SUCCEEDED.value,
        )

    async def mark_retry(
        self, job_id: str, error_message: str, *, max_attempts: int, backoff_seconds: list[int],
    ) -> None:
        """记录一次失败：未达重试上限则退避重试，否则转入死信。

        Args:
            job_id: 任务 ID。
            error_message: 失败原因（已脱敏，不含记忆正文/密钥）。
            max_attempts: 最大重试次数，达到后转 `dead`。
            backoff_seconds: 退避秒数表，按新的 attempts 次数取值，超出表长取最后一项。
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                new_attempts = await conn.fetchval(
                    "UPDATE memory_update_job SET attempts = attempts + 1, last_error = $2 "
                    "WHERE job_id = $1 RETURNING attempts",
                    job_id, error_message[:2000],
                )
                if new_attempts is None:
                    return
                if new_attempts >= max_attempts:
                    await conn.execute(
                        "UPDATE memory_update_job SET status = $2, completed_at = NOW() WHERE job_id = $1",
                        job_id, MemoryUpdateJobStatus.DEAD.value,
                    )
                    return
                delay = backoff_seconds[min(new_attempts - 1, len(backoff_seconds) - 1)] if backoff_seconds else 60
                await conn.execute(
                    "UPDATE memory_update_job SET status = $2, "
                    "next_retry_at = NOW() + make_interval(secs => $3), locked_until = NULL WHERE job_id = $1",
                    job_id, MemoryUpdateJobStatus.PENDING.value, delay,
                )

    async def reclaim_expired_leases(self) -> int:
        """把租约超时（进程崩溃/长时间挂起）的 `processing` 任务收回为 `pending`。

        Returns:
            本次收回的任务条数。
        """
        result = await self._pool.execute(
            "UPDATE memory_update_job SET status = $1, locked_until = NULL "
            "WHERE status = $2 AND locked_until < NOW()",
            MemoryUpdateJobStatus.PENDING.value, MemoryUpdateJobStatus.PROCESSING.value,
        )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


_store: MemoryUpdateJobStore | None = None


async def init_memory_update_job_store(pool: asyncpg.Pool) -> MemoryUpdateJobStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = MemoryUpdateJobStore(pool)
    await _store.setup()
    return _store


def get_memory_update_job_store() -> MemoryUpdateJobStore:
    """返回全局唯一的 MemoryUpdateJobStore 实例。

    Raises:
        RuntimeError: init_memory_update_job_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("MemoryUpdateJobStore 尚未初始化，请确认应用已完成启动")
    return _store
