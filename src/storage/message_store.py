"""持久化对话消息，供 `/conversations/{id}/messages` 历史接口使用。

原样迁移自 `diit-agent-server` 的 `src/storage/message_store.py`，改造为类
（原实现是模块级函数）。这张表与 LangGraph checkpointer 的关系：checkpointer
是 Lead Agent 推理/续接的事实来源，`conversation_messages` 是专门为 REST 历史
查询接口维护的、按 API 形状整理过的冗余副本——两者独立维护，互不依赖。
"""
from __future__ import annotations

import json
from typing import Optional

import asyncpg
from loguru import logger


class MessageStore:
    """`conversation_messages` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id              BIGSERIAL   PRIMARY KEY,
                conversation_id TEXT        NOT NULL,
                role            TEXT        NOT NULL,
                content         TEXT        NOT NULL DEFAULT '',
                thinking_content TEXT,
                tool_calls      TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            'ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS "references" TEXT'
        )
        await self._pool.execute(
            "ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS feedback TEXT"
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_conv ON conversation_messages(conversation_id, created_at ASC)"
        )
        logger.info("[MessageStore] 初始化完成")

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        thinking_content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        references: Optional[list] = None,
    ) -> Optional[int]:
        """新增一条消息记录。

        Args:
            conversation_id: 归属会话 ID。
            role: 消息角色（user/assistant/system）。
            content: 消息正文。
            thinking_content: 深度思考模式下的推理过程文本。
            tool_calls: 本轮工具调用记录列表，序列化为 JSON 文本存储。
            references: 本轮引用来源列表，序列化为 JSON 文本存储。

        Returns:
            新记录的自增 ID；写入失败时返回 None（不影响已生成/返回给用户的
            回复，只记 warning 日志）。前端用这个 ID 关联点赞/点踩反馈。
        """
        try:
            return await self._pool.fetchval(
                """
                INSERT INTO conversation_messages
                    (conversation_id, role, content, thinking_content, tool_calls, "references")
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                conversation_id, role, content, thinking_content,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                json.dumps(references, ensure_ascii=False) if references else None,
            )
        except Exception as exc:
            logger.warning(f"[MessageStore] 消息写入失败 conversation_id={conversation_id} role={role} error={exc}")
            return None

    async def get_messages(self, conversation_id: str) -> list[dict]:
        """按会话查询全部消息，按 created_at 升序。

        Args:
            conversation_id: 会话 ID。

        Returns:
            消息记录列表；查询失败时返回空列表，不抛出异常。
        """
        try:
            rows = await self._pool.fetch(
                'SELECT id, role, content, thinking_content, tool_calls, "references", feedback, created_at '
                "FROM conversation_messages WHERE conversation_id = $1 ORDER BY created_at ASC",
                conversation_id,
            )
        except Exception as exc:
            logger.warning(f"[MessageStore] 消息查询失败 conversation_id={conversation_id} error={exc}")
            return []

        messages: list[dict] = []
        for row in rows:
            message = dict(row)
            message["tool_calls"] = json.loads(message["tool_calls"]) if message["tool_calls"] else None
            message["references"] = json.loads(message["references"]) if message["references"] else None
            messages.append(message)
        return messages

    async def delete_messages(self, conversation_id: str) -> None:
        """删除某会话下的全部消息。

        Args:
            conversation_id: 会话 ID。
        """
        await self._pool.execute("DELETE FROM conversation_messages WHERE conversation_id = $1", conversation_id)

    async def delete_last_assistant_message(self, conversation_id: str) -> None:
        """删除某会话下最新的一条 assistant 消息，供"重新生成"覆盖旧回复使用。

        Args:
            conversation_id: 会话 ID。

        没有匹配记录时静默跳过（比如上一轮生成中途失败、从未写入过 assistant
        记录的场景），不视为异常。
        """
        await self._pool.execute(
            """
            DELETE FROM conversation_messages
            WHERE id = (
                SELECT id FROM conversation_messages
                WHERE conversation_id = $1 AND role = 'assistant'
                ORDER BY created_at DESC LIMIT 1
            )
            """,
            conversation_id,
        )

    async def set_feedback(self, message_id: int, conversation_id: str, feedback: Optional[str]) -> bool:
        """设置/清除一条消息的点赞点踩反馈。

        Args:
            message_id: 消息 ID。
            conversation_id: 归属会话 ID，用于校验消息确实属于该会话，防止跨会话越权改写。
            feedback: `"like"`/`"dislike"`，传 `None` 表示清除已有反馈（取消点赞/点踩）。

        Returns:
            是否命中并更新了一条记录；`message_id` 不存在或不属于该会话时返回 False。
        """
        result = await self._pool.execute(
            "UPDATE conversation_messages SET feedback = $1 WHERE id = $2 AND conversation_id = $3",
            feedback, message_id, conversation_id,
        )
        return result.split()[-1] != "0"


_store: MessageStore | None = None


async def init_message_store(pool: asyncpg.Pool) -> MessageStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = MessageStore(pool)
    await _store.setup()
    return _store


def get_message_store() -> MessageStore:
    """返回全局唯一的 MessageStore 实例。

    Raises:
        RuntimeError: init_message_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("MessageStore 尚未初始化，请确认应用已完成启动")
    return _store
