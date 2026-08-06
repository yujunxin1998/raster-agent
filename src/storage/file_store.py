"""持久化会话上传文件的元数据，供 `/conversations/{id}/uploads` 接口使用。

文件本体落在 `ThreadWorkspace.uploads_dir`（见 `agent_core/workspace/
thread_workspace.py`），本表只记录元数据（原始文件名、存储用文件名、大小、
MIME 类型、归属会话），职责划分对齐 `message_store.py` 与 checkpointer 的
关系——磁盘是文件内容的事实来源，这张表是便于列表查询/权限校验的索引。
"""
from __future__ import annotations

from typing import Optional

import asyncpg
from loguru import logger


class FileStore:
    """`conversation_files` 表的读写封装。"""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        """建表，幂等，应用启动时调用一次。"""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_files (
                id              TEXT        PRIMARY KEY,
                conversation_id TEXT        NOT NULL,
                user_id         TEXT        NOT NULL,
                original_name   TEXT        NOT NULL,
                stored_name     TEXT        NOT NULL,
                content_type    TEXT        NOT NULL DEFAULT 'application/octet-stream',
                size_bytes      BIGINT      NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_conv ON conversation_files(conversation_id, created_at ASC)"
        )
        logger.info("[FileStore] 初始化完成")

    async def add_file(
        self,
        file_id: str,
        conversation_id: str,
        user_id: str,
        original_name: str,
        stored_name: str,
        content_type: str,
        size_bytes: int,
    ) -> dict:
        """新增一条上传文件记录。

        Args:
            file_id: 文件 ID（同时是 uploads/ 目录下存储文件名的前缀）。
            conversation_id: 归属会话 ID。
            user_id: 归属用户 ID。
            original_name: 用户上传时的原始文件名。
            stored_name: 落盘时的实际文件名（`{file_id}_{original_name}`）。
            content_type: 客户端声明的 MIME 类型。
            size_bytes: 文件字节数。

        Returns:
            新建的记录（含全部列）。
        """
        row = await self._pool.fetchrow(
            """
            INSERT INTO conversation_files
                (id, conversation_id, user_id, original_name, stored_name, content_type, size_bytes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, conversation_id, user_id, original_name, stored_name, content_type, size_bytes, created_at
            """,
            file_id, conversation_id, user_id, original_name, stored_name, content_type, size_bytes,
        )
        return dict(row)

    async def list_files(self, conversation_id: str) -> list[dict]:
        """按会话列出全部上传文件，按 created_at 升序。

        Args:
            conversation_id: 会话 ID。

        Returns:
            文件记录列表；查询失败时返回空列表，不抛出异常。
        """
        try:
            rows = await self._pool.fetch(
                "SELECT id, conversation_id, user_id, original_name, stored_name, content_type, "
                "size_bytes, created_at FROM conversation_files "
                "WHERE conversation_id = $1 ORDER BY created_at ASC",
                conversation_id,
            )
        except Exception as exc:
            logger.warning(f"[FileStore] 文件列表查询失败 conversation_id={conversation_id} error={exc}")
            return []
        return [dict(row) for row in rows]

    async def get_file(self, file_id: str, conversation_id: str) -> Optional[dict]:
        """按 (file_id, conversation_id) 查询单条记录，归属校验内置在查询条件里。

        Args:
            file_id: 文件 ID。
            conversation_id: 会话 ID。

        Returns:
            文件记录；不存在或不属于该会话时返回 None。
        """
        row = await self._pool.fetchrow(
            "SELECT id, conversation_id, user_id, original_name, stored_name, content_type, "
            "size_bytes, created_at FROM conversation_files WHERE id = $1 AND conversation_id = $2",
            file_id, conversation_id,
        )
        return dict(row) if row else None

    async def get_files_by_ids(self, file_ids: list[str], conversation_id: str) -> list[dict]:
        """批量按 file_id 查询，用于对话轮次开始前把附件信息拼进 prompt。

        Args:
            file_ids: 文件 ID 列表。
            conversation_id: 会话 ID（归属校验）。

        Returns:
            命中的文件记录列表，顺序不保证与入参一致；不存在的 id 静默跳过。
        """
        if not file_ids:
            return []
        rows = await self._pool.fetch(
            "SELECT id, conversation_id, user_id, original_name, stored_name, content_type, "
            "size_bytes, created_at FROM conversation_files WHERE id = ANY($1::text[]) AND conversation_id = $2",
            file_ids, conversation_id,
        )
        return [dict(row) for row in rows]

    async def delete_files(self, conversation_id: str) -> None:
        """删除某会话下的全部文件记录（不负责删除磁盘文件，由调用方处理）。

        Args:
            conversation_id: 会话 ID。
        """
        await self._pool.execute("DELETE FROM conversation_files WHERE conversation_id = $1", conversation_id)


_store: FileStore | None = None


async def init_file_store(pool: asyncpg.Pool) -> FileStore:
    """应用启动时调用一次，建表并注册全局单例。"""
    global _store
    _store = FileStore(pool)
    await _store.setup()
    return _store


def get_file_store() -> FileStore:
    """返回全局唯一的 FileStore 实例。

    Raises:
        RuntimeError: init_file_store() 尚未被调用。
    """
    if _store is None:
        raise RuntimeError("FileStore 尚未初始化，请确认应用已完成启动")
    return _store
