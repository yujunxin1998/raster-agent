"""长期记忆 Facts 的规范化主存（Memory v2，三层记忆架构的 L3）。

设计文档 §4.4：PostgreSQL 是 Facts 的真相源，`ElasticsearchMemoryStore` 降级为
纯检索投影（kNN + rerank），不再是主存——旧版本直接把 ES 当主存的问题（写入
即分片可见性延迟、无法做跨字段事务、相似度阈值被迫承担事实判断）到这里解决。

去重规则遵循设计文档 §6.2："相似 ≠ 重复"：只有 `normalize_fact()` 规范化后的
文本在同一 `(user_id, agent_name, category)` 范围内完全相同才判定重复，命中唯一
索引 `ux_active_fact_normalized` 时转换为 `reinforce`（补充确认次数），不产生新
Fact，也不依赖 embedding 阈值。

每次写入都在同一事务里追加一条 `memory_outbox` 记录，`FactProjector`
（`src/agent_core/memory/fact_projector.py`）据此异步把变更同步进 ES 投影，
PostgreSQL 事务成功不因 ES 故障回滚（最终一致，设计文档 §7.6）。
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

import asyncpg
from loguru import logger

from src.common.constants import MemoryOutboxAction, MemoryStatus

_MUTABLE_STATUSES = frozenset({MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value})


def normalize_fact(content: str) -> str:
    """把 Fact 内容规范化为去重比较用的文本（设计文档 §6.2，原样实现）。

    Args:
        content: 原始 Fact 内容。

    Returns:
        去除首尾空白、把内部连续空白折叠为单个空格后的文本。
    """
    return " ".join(content.strip().split())


class UserMemoryFactStore:
    """`user_memory_fact` / `memory_outbox` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表 + 索引，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory_fact (
                fact_id                UUID         PRIMARY KEY,
                user_id                TEXT         NOT NULL,
                agent_name             TEXT,
                content                TEXT         NOT NULL,
                normalized_content     TEXT         NOT NULL,
                category               TEXT         NOT NULL,
                importance             SMALLINT     NOT NULL,
                confidence             NUMERIC(3,2) NOT NULL,
                status                 TEXT         NOT NULL,
                source_event_id        TEXT,
                source_conversation_id TEXT,
                evidence_message_ids   JSONB        NOT NULL DEFAULT '[]',
                supersedes_fact_id     UUID,
                superseded_by_fact_id  UUID,
                created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                expires_at             TIMESTAMPTZ,
                last_accessed_at       TIMESTAMPTZ,
                access_count           BIGINT       NOT NULL DEFAULT 0,
                revision               BIGINT       NOT NULL DEFAULT 0
            )
            """
        )
        await self._pool.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_active_fact_normalized
            ON user_memory_fact(user_id, COALESCE(agent_name, ''), category, normalized_content)
            WHERE status IN ('active', 'pending')
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_memory_fact_user ON user_memory_fact(user_id, status)"
        )
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_outbox (
                outbox_id    BIGSERIAL   PRIMARY KEY,
                fact_id      UUID        NOT NULL,
                action       TEXT        NOT NULL,
                snapshot     JSONB       NOT NULL,
                processed_at TIMESTAMPTZ,
                attempts     INT         NOT NULL DEFAULT 0,
                last_error   TEXT,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_outbox_unprocessed "
            "ON memory_outbox(processed_at, created_at) WHERE processed_at IS NULL"
        )
        logger.info("[UserMemoryFactStore] 初始化完成")

    # ── Fact Operation（设计文档 §5.4）──────────────────────────

    async def add_or_reinforce(
        self,
        *,
        user_id: str,
        content: str,
        category: str,
        importance: int,
        confidence: float,
        status: str,
        source_event_id: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        evidence_message_ids: Optional[list[str]] = None,
        agent_name: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> dict:
        """新增一条 Fact；命中确定性去重唯一键时转为 `reinforce`。

        Returns:
            `{"fact_id": str, "action": "added" | "reinforced", "status": str}`。
        """
        normalized = normalize_fact(content)
        fact_id = str(uuid.uuid4())
        evidence_json = json.dumps(evidence_message_ids or [], ensure_ascii=False)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                inserted = await conn.fetchrow(
                    """
                    INSERT INTO user_memory_fact (
                        fact_id, user_id, agent_name, content, normalized_content, category,
                        importance, confidence, status, source_event_id, source_conversation_id,
                        evidence_message_ids, expires_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13)
                    ON CONFLICT (user_id, COALESCE(agent_name, ''), category, normalized_content)
                    WHERE status IN ('active', 'pending')
                    DO NOTHING
                    RETURNING fact_id, status
                    """,
                    fact_id, user_id, agent_name, content, normalized, category,
                    importance, confidence, status, source_event_id, source_conversation_id,
                    evidence_json, expires_at,
                )
                if inserted is not None:
                    snapshot = await self._fetch_snapshot(conn, inserted["fact_id"])
                    await self._append_outbox(
                        conn, inserted["fact_id"],
                        MemoryOutboxAction.FACT_CREATED if inserted["status"] == MemoryStatus.ACTIVE.value
                        else MemoryOutboxAction.FACT_STATUS_CHANGED,
                        snapshot,
                    )
                    return {"fact_id": str(inserted["fact_id"]), "action": "added", "status": inserted["status"]}

                existing = await conn.fetchrow(
                    """
                    SELECT fact_id, status FROM user_memory_fact
                    WHERE user_id = $1 AND COALESCE(agent_name, '') = COALESCE($2, '')
                      AND category = $3 AND normalized_content = $4
                      AND status IN ('active', 'pending')
                    FOR UPDATE
                    """,
                    user_id, agent_name, category, normalized,
                )
                if existing is None:
                    logger.warning(
                        f"[UserMemoryFactStore] 唯一键冲突但未找到已存在记录（并发归档竞态），"
                        f"降级为直接插入 user={user_id} category={category}"
                    )
                    await conn.execute(
                        """
                        INSERT INTO user_memory_fact (
                            fact_id, user_id, agent_name, content, normalized_content, category,
                            importance, confidence, status, source_event_id, source_conversation_id,
                            evidence_message_ids, expires_at
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13)
                        """,
                        fact_id, user_id, agent_name, content, normalized, category,
                        importance, confidence, status, source_event_id, source_conversation_id,
                        evidence_json, expires_at,
                    )
                    snapshot = await self._fetch_snapshot(conn, fact_id)
                    await self._append_outbox(conn, fact_id, MemoryOutboxAction.FACT_CREATED, snapshot)
                    return {"fact_id": fact_id, "action": "added", "status": status}

                await conn.execute(
                    "UPDATE user_memory_fact SET access_count = access_count + 1, "
                    "last_accessed_at = NOW(), updated_at = NOW(), revision = revision + 1 WHERE fact_id = $1",
                    existing["fact_id"],
                )
                snapshot = await self._fetch_snapshot(conn, existing["fact_id"])
                await self._append_outbox(conn, existing["fact_id"], MemoryOutboxAction.FACT_UPDATED, snapshot)
                return {"fact_id": str(existing["fact_id"]), "action": "reinforced", "status": existing["status"]}

    async def supersede(
        self,
        *,
        target_fact_id: str,
        user_id: str,
        content: str,
        category: str,
        importance: int,
        confidence: float,
        source_event_id: Optional[str],
        evidence_message_ids: Optional[list[str]],
        agent_name: Optional[str] = None,
    ) -> Optional[str]:
        """用一条新 Fact 显式取代旧 Fact（设计文档 §6.3 四步）。

        Returns:
            新 Fact 的 ID；旧 Fact 不存在、不属于该用户或已处于终态时返回 None。
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                old = await conn.fetchrow(
                    "SELECT fact_id, status FROM user_memory_fact "
                    "WHERE fact_id = $1 AND user_id = $2 FOR UPDATE",
                    target_fact_id, user_id,
                )
                if old is None or old["status"] not in (MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value):
                    return None

                new_fact_id = str(uuid.uuid4())
                normalized = normalize_fact(content)
                await conn.execute(
                    """
                    INSERT INTO user_memory_fact (
                        fact_id, user_id, agent_name, content, normalized_content, category,
                        importance, confidence, status, source_event_id, evidence_message_ids,
                        supersedes_fact_id
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12)
                    """,
                    new_fact_id, user_id, agent_name, content, normalized, category,
                    importance, confidence, MemoryStatus.ACTIVE.value, source_event_id,
                    json.dumps(evidence_message_ids or [], ensure_ascii=False), target_fact_id,
                )
                await conn.execute(
                    "UPDATE user_memory_fact SET status = $2, superseded_by_fact_id = $3, "
                    "updated_at = NOW(), revision = revision + 1 WHERE fact_id = $1",
                    target_fact_id, MemoryStatus.SUPERSEDED.value, new_fact_id,
                )

                new_snapshot = await self._fetch_snapshot(conn, new_fact_id)
                old_snapshot = await self._fetch_snapshot(conn, target_fact_id)
                await self._append_outbox(conn, new_fact_id, MemoryOutboxAction.FACT_CREATED, new_snapshot)
                await self._append_outbox(conn, target_fact_id, MemoryOutboxAction.FACT_STATUS_CHANGED, old_snapshot)
        return new_fact_id

    async def update(
        self,
        fact_id: str,
        user_id: str,
        *,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[int] = None,
        confidence: Optional[float] = None,
        status: Optional[str] = None,
        expires_at: Optional[str] = None,
        clear_expires_at: bool = False,
        allow_any_status: bool = False,
    ) -> bool:
        """局部更新一条 Fact。

        默认只允许在 `active`/`pending` 状态下更新（`update` Fact Operation 的
        安全边界，设计文档 §5.4）；管理 API 需要能手动改回/改成任意状态（比如把
        误归档的 Fact 恢复为 active），传 `allow_any_status=True` 跳过这道前置
        状态检查——这是唯一区别，仍然共用同一份"读现值 -> 局部覆盖 -> 写回 +
        outbox"逻辑，不为两个调用方各写一份。

        Returns:
            是否更新成功；目标不存在、不属于该用户，或
            （`allow_any_status=False` 时）已处于终态时返回 False。
        """
        fields: dict = {}
        if content is not None:
            fields["content"] = content
            fields["normalized_content"] = normalize_fact(content)
        if category is not None:
            fields["category"] = category
        if importance is not None:
            fields["importance"] = importance
        if confidence is not None:
            fields["confidence"] = confidence
        if status is not None:
            fields["status"] = status
        if clear_expires_at:
            fields["expires_at"] = None
        elif expires_at is not None:
            fields["expires_at"] = expires_at
        if not fields:
            return True

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                current = await conn.fetchrow(
                    "SELECT status FROM user_memory_fact WHERE fact_id = $1 AND user_id = $2 FOR UPDATE",
                    fact_id, user_id,
                )
                if current is None:
                    return False
                if not allow_any_status and current["status"] not in _MUTABLE_STATUSES:
                    return False

                columns = list(fields.keys())
                set_clause = ", ".join(f"{column} = ${index}" for index, column in enumerate(columns, start=3))
                await conn.execute(
                    f"UPDATE user_memory_fact SET {set_clause}, updated_at = NOW(), revision = revision + 1 "
                    f"WHERE fact_id = $1 AND user_id = $2",
                    fact_id, user_id, *(fields[column] for column in columns),
                )
                snapshot = await self._fetch_snapshot(conn, fact_id)
                await self._append_outbox(conn, fact_id, MemoryOutboxAction.FACT_UPDATED, snapshot)
        return True

    async def reinforce(self, fact_id: str, user_id: str, *, confidence: Optional[float] = None) -> bool:
        """`reinforce` Fact Operation：不新增 Fact，只补充确认次数/置信度/访问时间。

        Returns:
            是否命中成功；目标不存在、不属于该用户或已处于终态时返回 False。
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                current = await conn.fetchrow(
                    "SELECT status, confidence FROM user_memory_fact WHERE fact_id = $1 AND user_id = $2 FOR UPDATE",
                    fact_id, user_id,
                )
                if current is None or current["status"] not in _MUTABLE_STATUSES:
                    return False
                new_confidence = max(current["confidence"], confidence) if confidence is not None else current["confidence"]
                await conn.execute(
                    "UPDATE user_memory_fact SET access_count = access_count + 1, last_accessed_at = NOW(), "
                    "confidence = $3, updated_at = NOW(), revision = revision + 1 WHERE fact_id = $1 AND user_id = $2",
                    fact_id, user_id, new_confidence,
                )
                snapshot = await self._fetch_snapshot(conn, fact_id)
                await self._append_outbox(conn, fact_id, MemoryOutboxAction.FACT_UPDATED, snapshot)
        return True

    async def archive(self, fact_id: str, user_id: str) -> bool:
        """把一条 Fact 归档（设计文档 §5.4 archive 操作 / staleness 治理共用）。

        Returns:
            是否命中并归档成功；不存在或不属于该用户时返回 False。
        """
        return await self._transition_status(fact_id, user_id, MemoryStatus.ARCHIVED.value)

    async def confirm_pending(self, fact_id: str, user_id: str) -> bool:
        """人工确认一条 `pending` Fact，转为 `active`（管理 API 使用）。"""
        return await self._transition_status(
            fact_id, user_id, MemoryStatus.ACTIVE.value, from_status=MemoryStatus.PENDING.value,
        )

    async def reject_pending(self, fact_id: str, user_id: str) -> bool:
        """人工拒绝一条 `pending` Fact，直接归档（管理 API 使用）。"""
        return await self._transition_status(
            fact_id, user_id, MemoryStatus.ARCHIVED.value, from_status=MemoryStatus.PENDING.value,
        )

    async def delete(self, fact_id: str, user_id: str) -> bool:
        """彻底删除一条 Fact（管理 API 的硬删除，跟 `archive` 的软删除是两回事）。

        Returns:
            是否命中并删除成功；不存在或不属于该用户时返回 False。
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "DELETE FROM user_memory_fact WHERE fact_id = $1 AND user_id = $2 RETURNING fact_id",
                    fact_id, user_id,
                )
                if row is None:
                    return False
                await self._append_outbox(
                    conn, fact_id, MemoryOutboxAction.FACT_DELETED, {"fact_id": fact_id, "status": "deleted"},
                )
        return True

    async def purge_user(self, user_id: str) -> int:
        """彻底删除某用户名下的全部 Fact（设计文档 §9.2"导出和彻底删除用户记忆"）。

        逐条复用 `delete()`（含 outbox 清理），不是裸 `DELETE ... WHERE user_id`——
        保证每条被删除的 Fact 都对应一条 outbox 记录，`FactProjector` 才能把
        ES 投影里的残留一并清掉。

        Returns:
            实际删除的条数。
        """
        rows = await self._pool.fetch("SELECT fact_id FROM user_memory_fact WHERE user_id = $1", user_id)
        deleted = 0
        for row in rows:
            if await self.delete(str(row["fact_id"]), user_id):
                deleted += 1
        return deleted

    async def _transition_status(
        self, fact_id: str, user_id: str, new_status: str, *, from_status: Optional[str] = None,
    ) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if from_status:
                    row = await conn.fetchrow(
                        "UPDATE user_memory_fact SET status = $3, updated_at = NOW(), revision = revision + 1 "
                        "WHERE fact_id = $1 AND user_id = $2 AND status = $4 RETURNING fact_id",
                        fact_id, user_id, new_status, from_status,
                    )
                else:
                    row = await conn.fetchrow(
                        "UPDATE user_memory_fact SET status = $3, updated_at = NOW(), revision = revision + 1 "
                        "WHERE fact_id = $1 AND user_id = $2 RETURNING fact_id",
                        fact_id, user_id, new_status,
                    )
                if row is None:
                    return False
                snapshot = await self._fetch_snapshot(conn, fact_id)
                await self._append_outbox(conn, fact_id, MemoryOutboxAction.FACT_STATUS_CHANGED, snapshot)
        return True

    # ── 查询 ──────────────────────────────────────────────────

    async def get(self, fact_id: str, user_id: str) -> Optional[dict]:
        """按 ID 获取指定用户的一条 Fact。"""
        row = await self._pool.fetchrow(
            "SELECT * FROM user_memory_fact WHERE fact_id = $1 AND user_id = $2", fact_id, user_id,
        )
        return self._to_dict(row) if row else None

    async def get_many(self, fact_ids: list[str], user_id: str) -> list[dict]:
        """批量获取（校验归属），供 Delta Validator 检查 `target_fact_id` 合法性使用。"""
        if not fact_ids:
            return []
        rows = await self._pool.fetch(
            "SELECT * FROM user_memory_fact WHERE fact_id = ANY($1::uuid[]) AND user_id = $2",
            fact_ids, user_id,
        )
        return [self._to_dict(row) for row in rows]

    async def list_by_user(self, user_id: str, *, statuses: Optional[list[str]] = None, limit: int = 200) -> list[dict]:
        """列出某用户的 Facts（管理 API / 导出使用）。"""
        if statuses:
            rows = await self._pool.fetch(
                "SELECT * FROM user_memory_fact WHERE user_id = $1 AND status = ANY($2::text[]) "
                "ORDER BY created_at DESC LIMIT $3",
                user_id, statuses, limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM user_memory_fact WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                user_id, limit,
            )
        return [self._to_dict(row) for row in rows]

    async def list_pending(self, user_id: str, limit: int = 200) -> list[dict]:
        """列出某用户待人工确认的 Facts（Memory v2 人工闭环入口）。"""
        return await self.list_by_user(user_id, statuses=[MemoryStatus.PENDING.value], limit=limit)

    async def find_candidates_for_update(
        self, user_id: str, *, agent_name: Optional[str], limit: int,
    ) -> list[dict]:
        """给 MemoryUpdateWorker 提供更新候选 Fact（设计文档 §7.3，只取活跃/待确认）。

        按最近访问/更新时间倒序，不整表塞进 Prompt——只取 `limit` 条。

        Args:
            user_id: 归属用户 ID。
            agent_name: 限定到某个 Agent 专属记忆；为空则不限制。
            limit: 最多返回条数。
        """
        if agent_name:
            rows = await self._pool.fetch(
                "SELECT * FROM user_memory_fact WHERE user_id = $1 AND agent_name = $2 "
                "AND status IN ('active', 'pending') "
                "ORDER BY COALESCE(last_accessed_at, updated_at) DESC LIMIT $3",
                user_id, agent_name, limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM user_memory_fact WHERE user_id = $1 AND status IN ('active', 'pending') "
                "ORDER BY COALESCE(last_accessed_at, updated_at) DESC LIMIT $2",
                user_id, limit,
            )
        return [self._to_dict(row) for row in rows]

    async def sweep_stale(self, *, max_age_days: int, low_importance_threshold: int) -> int:
        """归档过期或长期无人问津的低价值 Fact（迁移自 ES `sweep_stale`，语义不变）。

        Returns:
            本次归档的条数。
        """
        rows = await self._pool.fetch(
            """
            SELECT fact_id, user_id FROM user_memory_fact
            WHERE status = 'active' AND (
                (expires_at IS NOT NULL AND expires_at < NOW())
                OR (
                    importance <= $1 AND access_count = 0
                    AND created_at < NOW() - make_interval(days => $2)
                )
            )
            """,
            low_importance_threshold, max_age_days,
        )
        archived = 0
        for row in rows:
            if await self.archive(str(row["fact_id"]), row["user_id"]):
                archived += 1
        if archived:
            logger.info(f"[UserMemoryFactStore] staleness 扫描归档 {archived} 条 Fact")
        return archived

    # ── Outbox（供 FactProjector 消费）──────────────────────────

    async def claim_outbox_batch(self, batch_size: int) -> list[dict]:
        """认领一批未处理的 Outbox 记录（`FactProjector` 使用）。"""
        rows = await self._pool.fetch(
            "SELECT outbox_id, fact_id, action, snapshot FROM memory_outbox "
            "WHERE processed_at IS NULL ORDER BY created_at LIMIT $1 FOR UPDATE SKIP LOCKED",
            batch_size,
        )
        return [
            {"outbox_id": row["outbox_id"], "fact_id": str(row["fact_id"]), "action": row["action"],
             "snapshot": json.loads(row["snapshot"])}
            for row in rows
        ]

    async def mark_outbox_processed(self, outbox_id: int) -> None:
        await self._pool.execute(
            "UPDATE memory_outbox SET processed_at = NOW() WHERE outbox_id = $1", outbox_id,
        )

    async def mark_outbox_failed(self, outbox_id: int, error_message: str) -> None:
        await self._pool.execute(
            "UPDATE memory_outbox SET attempts = attempts + 1, last_error = $2 WHERE outbox_id = $1",
            outbox_id, error_message[:2000],
        )

    # ── 内部方法 ──────────────────────────────────────────────

    async def _append_outbox(self, conn: asyncpg.Connection, fact_id, action: MemoryOutboxAction, snapshot: dict) -> None:
        # default=str 会把 datetime 序列化成 "YYYY-MM-DD HH:MM:SS+00:00"（空格分隔），
        # ES 默认日期格式要求 ISO-8601 的 "T" 分隔符，用 isoformat() 才能被
        # FactProjector 写入 ES 的 date 字段正确解析。
        await conn.execute(
            "INSERT INTO memory_outbox (fact_id, action, snapshot) VALUES ($1, $2, $3::jsonb)",
            fact_id, action.value,
            json.dumps(
                snapshot, ensure_ascii=False,
                default=lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value),
            ),
        )

    async def _fetch_snapshot(self, conn: asyncpg.Connection, fact_id) -> dict:
        row = await conn.fetchrow("SELECT * FROM user_memory_fact WHERE fact_id = $1", fact_id)
        return self._to_dict(row)

    @staticmethod
    def _to_dict(row: asyncpg.Record) -> dict:
        record = dict(row)
        record["fact_id"] = str(record["fact_id"])
        if record.get("supersedes_fact_id") is not None:
            record["supersedes_fact_id"] = str(record["supersedes_fact_id"])
        if record.get("superseded_by_fact_id") is not None:
            record["superseded_by_fact_id"] = str(record["superseded_by_fact_id"])
        evidence = record.get("evidence_message_ids")
        if isinstance(evidence, str):
            record["evidence_message_ids"] = json.loads(evidence)
        return record


_store: UserMemoryFactStore | None = None


async def init_user_memory_fact_store(pool: asyncpg.Pool) -> UserMemoryFactStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = UserMemoryFactStore(pool)
    await _store.setup()
    return _store


def get_user_memory_fact_store() -> UserMemoryFactStore:
    """返回全局唯一的 UserMemoryFactStore 实例。

    Raises:
        RuntimeError: init_user_memory_fact_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("UserMemoryFactStore 尚未初始化，请确认应用已完成启动")
    return _store
