"""记忆捕获事件（Memory v2 更新流水线的输入端）。

对应设计文档 §4.1：`memory_event` 是清洗后的不可变输入，只保存用户原话与最终、
无悬挂 Tool Call 的 AI 回复，既不是 Fact，也不直接注入模型。`from_message_id`/
`to_message_id` 取 LangChain 消息对象的 `.id`（`MemoryCaptureMiddleware` 在
`aafter_agent` 钩子里能拿到的就是这个 ID，不做到 SQL `conversation_messages`
整数主键的映射，避免不必要的往返查询）。

`to_message_id` 同时承担 watermark 语义：`get_watermark()` 返回某个
`(user_id, conversation_id)` 最近一条已捕获事件的 `to_message_id`，
`MemoryCaptureMiddleware` 据此只处理这之后的新增消息，不重复处理已入队的对话。
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Optional

import asyncpg
from loguru import logger

if TYPE_CHECKING:
    from src.storage.memory_update_job_store import MemoryUpdateJobStore


class MemoryEventStore:
    """`memory_event` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表 + 索引，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_event (
                event_id           UUID        PRIMARY KEY,
                user_id             TEXT        NOT NULL,
                conversation_id     TEXT        NOT NULL,
                agent_name          TEXT,
                trace_id            TEXT,
                from_message_id     TEXT        NOT NULL,
                to_message_id       TEXT        NOT NULL,
                conversation_json   TEXT        NOT NULL,
                content_hash        TEXT        NOT NULL,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (user_id, conversation_id, from_message_id, to_message_id)
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_event_watermark "
            "ON memory_event(user_id, conversation_id, created_at DESC)"
        )
        logger.info("[MemoryEventStore] 初始化完成")

    async def get_watermark(self, user_id: str, conversation_id: str) -> Optional[str]:
        """查询某会话最近一条已捕获事件的 `to_message_id`，供增量捕获使用。

        Args:
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。

        Returns:
            最近一条事件的 `to_message_id`；该会话还没有任何事件时返回 None
            （代表要从头开始捕获）。
        """
        return await self._pool.fetchval(
            "SELECT to_message_id FROM memory_event "
            "WHERE user_id = $1 AND conversation_id = $2 "
            "ORDER BY created_at DESC LIMIT 1",
            user_id, conversation_id,
        )

    async def get_conversation_for_job(self, user_id: str, conversation_id: str, first_event_id: str, last_event_id: str) -> list[dict]:
        """按 `created_at` 顺序拼接一个（可能被防抖合并过的）任务覆盖的全部事件对话。

        `MemoryUpdateJobStore.create_job()` 把落在防抖窗口内的相邻事件合并进
        同一个任务时只推进了 `last_event_id`，并不会改写更早事件里已经落盘的
        `conversation_json`；所以这里要按时间区间取出 `first_event_id` 到
        `last_event_id` 之间（含两端）该会话的全部事件，依次拼接，而不是只读
        `last_event_id` 一条。

        Args:
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            first_event_id: 任务首次创建时对应的事件 ID。
            last_event_id: 任务当前指向的最新事件 ID（可能与 first_event_id 相同）。

        Returns:
            按时间顺序拼接后的对话消息列表，每条为 `{"role", "content", "id"}`。
        """
        rows = await self._pool.fetch(
            """
            SELECT conversation_json FROM memory_event
            WHERE user_id = $1 AND conversation_id = $2
              AND created_at >= (SELECT created_at FROM memory_event WHERE event_id = $3)
              AND created_at <= (SELECT created_at FROM memory_event WHERE event_id = $4)
            ORDER BY created_at ASC
            """,
            user_id, conversation_id, first_event_id, last_event_id,
        )
        conversation: list[dict] = []
        for row in rows:
            conversation.extend(json.loads(row["conversation_json"]))
        return conversation

    async def create_event_and_job(
        self,
        job_store: "MemoryUpdateJobStore",
        *,
        user_id: str,
        conversation_id: str,
        from_message_id: str,
        to_message_id: str,
        conversation: list[dict],
        content_hash: str,
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        debounce_seconds: int = 30,
    ) -> Optional[str]:
        """在同一事务内写入一条 `memory_event` 并入队对应的 `memory_update_job`。

        两张表在一个事务里写，防止"事件写成功但任务没入队"或反过来的半成品状态
        （设计文档 §4.2）。

        Args:
            job_store: 用于在同一事务内创建任务的 `MemoryUpdateJobStore`。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            from_message_id: 本次捕获窗口起始消息 ID（不含，即上一次 watermark）。
            to_message_id: 本次捕获窗口结束消息 ID（含）。
            conversation: 清洗后的对话内容（User/最终 AI 回复），会被 JSON 序列化。
            content_hash: 对话内容摘要，供去重/审计参考。
            agent_name: 产生该事件的 Agent 名，多 Agent 场景下用于隔离画像/事实。
            trace_id: 链路追踪 ID。

        Returns:
            新建的 `job_id`；命中 `UNIQUE` 约束（重复捕获同一窗口）时返回 None。
        """
        event_id = str(uuid.uuid4())
        conversation_json = json.dumps(conversation, ensure_ascii=False)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                inserted = await conn.fetchval(
                    """
                    INSERT INTO memory_event (
                        event_id, user_id, conversation_id, agent_name, trace_id,
                        from_message_id, to_message_id, conversation_json, content_hash
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (user_id, conversation_id, from_message_id, to_message_id) DO NOTHING
                    RETURNING event_id
                    """,
                    event_id, user_id, conversation_id, agent_name, trace_id,
                    from_message_id, to_message_id, conversation_json, content_hash,
                )
                if inserted is None:
                    return None

                job_id = await job_store.create_job(
                    conn,
                    user_id=user_id, conversation_id=conversation_id, agent_name=agent_name,
                    first_event_id=event_id, last_event_id=event_id, trace_id=trace_id,
                    debounce_seconds=debounce_seconds,
                )
        return job_id


_store: MemoryEventStore | None = None


async def init_memory_event_store(pool: asyncpg.Pool) -> MemoryEventStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = MemoryEventStore(pool)
    await _store.setup()
    return _store


def get_memory_event_store() -> MemoryEventStore:
    """返回全局唯一的 MemoryEventStore 实例。

    Raises:
        RuntimeError: init_memory_event_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("MemoryEventStore 尚未初始化，请确认应用已完成启动")
    return _store
