"""记忆后处理任务（提取/压缩）的执行记录。

进程内以 `asyncio.create_task` 驱动任务的 pending → running → done/failed
状态流转，不引入独立任务队列；崩溃恢复依赖启动时扫描 pending/running/
未超过重试次数的 failed 记录（见 `list_recoverable`）。与
`MemoryAuditStore` 一致，属于辅助能力，未初始化或写入失败时只记 warning
并静默降级。
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

import asyncpg
from loguru import logger

from src.common.constants import MemoryJobStatus

_LAST_ERROR_MAX_CHARS = 2000


class MemoryJobsStore:
    """`memory_jobs` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表 + 索引，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_jobs (
                id              TEXT        PRIMARY KEY,
                job_type        TEXT        NOT NULL,
                user_id         TEXT        NOT NULL,
                conversation_id TEXT        NOT NULL,
                trace_id        TEXT,
                payload         TEXT        NOT NULL,
                status          TEXT        NOT NULL DEFAULT 'pending',
                attempts        INT         NOT NULL DEFAULT 0,
                last_error      TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_jobs_status ON memory_jobs(status, attempts)"
        )
        logger.info("[MemoryJobsStore] 初始化完成")

    async def create_job(
        self,
        job_type: str,
        user_id: str,
        conversation_id: str,
        payload: dict,
        trace_id: Optional[str] = None,
    ) -> str:
        """创建一条任务记录。

        Args:
            job_type: 任务类型（如 "extract" / "compress"）。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            payload: 任务参数，会被 JSON 序列化后存储。
            trace_id: 链路追踪 ID。

        Returns:
            新生成的任务 ID；写入失败时仍返回一个本地生成的 ID（不影响调用方
            继续执行任务本身，只是这次任务不会被记录用于崩溃恢复）。
        """
        job_id = str(uuid.uuid4())
        try:
            await self._pool.execute(
                """
                INSERT INTO memory_jobs (id, job_type, user_id, conversation_id, trace_id, payload)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                job_id, job_type, user_id, conversation_id, trace_id,
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning(f"[MemoryJobsStore] 创建任务记录失败 job_id={job_id}: {exc}")
        return job_id

    async def mark_running(self, job_id: str) -> None:
        """将任务状态置为 running。"""
        await self._update_status(job_id, MemoryJobStatus.RUNNING)

    async def mark_done(self, job_id: str) -> None:
        """将任务状态置为 done。"""
        await self._update_status(job_id, MemoryJobStatus.DONE)

    async def mark_failed(self, job_id: str, error_message: str) -> None:
        """将任务状态置为 failed，并记录错误信息、递增重试次数。

        Args:
            job_id: 任务 ID。
            error_message: 错误描述，超长部分会被截断。
        """
        try:
            await self._pool.execute(
                """
                UPDATE memory_jobs
                SET status = $2, attempts = attempts + 1, last_error = $3, updated_at = NOW()
                WHERE id = $1
                """,
                job_id, MemoryJobStatus.FAILED.value, error_message[:_LAST_ERROR_MAX_CHARS],
            )
        except Exception as exc:
            logger.warning(f"[MemoryJobsStore] 更新任务失败状态失败 job_id={job_id}: {exc}")

    async def list_recoverable(self, max_attempts: int, limit: int = 100) -> list[dict]:
        """启动时扫描未正常完成的任务。

        范围：pending/running（进程异常退出遗留）以及未超过重试次数的 failed。

        Args:
            max_attempts: 最大重试次数，failed 任务的 attempts 达到此值后不再恢复。
            limit: 最多返回条数。

        Returns:
            待恢复的任务记录列表，payload 已反序列化为 dict；查询失败返回空列表。
        """
        try:
            rows = await self._pool.fetch(
                """
                SELECT id, job_type, user_id, conversation_id, trace_id, payload, attempts
                FROM memory_jobs
                WHERE status IN ('pending', 'running')
                   OR (status = 'failed' AND attempts < $1)
                ORDER BY created_at ASC
                LIMIT $2
                """,
                max_attempts, limit,
            )
        except Exception as exc:
            logger.warning(f"[MemoryJobsStore] 查询待恢复任务失败: {exc}")
            return []

        result: list[dict] = []
        for row in rows:
            record = dict(row)
            try:
                record["payload"] = json.loads(record["payload"])
            except (json.JSONDecodeError, TypeError):
                record["payload"] = {}
            result.append(record)
        return result

    async def _update_status(self, job_id: str, status: MemoryJobStatus) -> None:
        try:
            await self._pool.execute(
                "UPDATE memory_jobs SET status = $2, updated_at = NOW() WHERE id = $1",
                job_id, status.value,
            )
        except Exception as exc:
            logger.warning(f"[MemoryJobsStore] 更新任务状态失败 job_id={job_id} status={status}: {exc}")


_store: MemoryJobsStore | None = None


async def init_memory_jobs_store(pool: asyncpg.Pool) -> MemoryJobsStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = MemoryJobsStore(pool)
    await _store.setup()
    return _store


def get_memory_jobs_store() -> Optional[MemoryJobsStore]:
    """返回全局唯一的 MemoryJobsStore 实例，未初始化时返回 None（辅助能力，静默降级）。"""
    return _store
