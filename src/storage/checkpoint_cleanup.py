"""LangGraph checkpoint 孤儿数据清理（设计文档第七节路线图第三期）。

`AsyncPostgresSaver` 把每个会话（`conversation_id` 即 LangGraph 的 `thread_id`）
的推理状态存在 `checkpoints`/`checkpoint_blobs`/`checkpoint_writes` 三张表里，
这三张表由 checkpointer 自己管理，`conversation_store`/`message_store` 对它们
一无所知。此前 `conversation_router.py::delete_conversation` 删除会话时只清了
`conversations`/`conversation_messages` 两张表，从未通知 checkpointer——这正是
孤儿数据的产生点，而且这个缺口从项目一开始就在，历史上已经删过的会话在这三张
表里已经积累了孤儿数据，只堵住新增入口还不够，需要再补一次扫描。

清理动作统一走 `AsyncPostgresSaver.adelete_thread(thread_id)`（LangGraph 自带
的官方方法，内部依次对三张表做 `DELETE ... WHERE thread_id = ...`），不自己
手写 SQL 操作 LangGraph 的内部表结构，以后其内部表结构变化也不会影响这段代码。
"""
from __future__ import annotations

import asyncpg
from langgraph.checkpoint.base import BaseCheckpointSaver
from loguru import logger


class CheckpointCleanup:
    """checkpoint 孤儿数据清理的读写封装。"""

    def __init__(self, pool: asyncpg.Pool, checkpointer: BaseCheckpointSaver) -> None:
        self._pool = pool
        self._checkpointer = checkpointer

    async def delete_for_thread(self, thread_id: str) -> None:
        """会话被删除时立即调用，清掉这个 thread 在 checkpoint 三张表里的数据。

        Args:
            thread_id: 会话 ID（等同 conversation_id）。
        """
        try:
            await self._checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            logger.warning(f"[CheckpointCleanup] 清理 thread_id={thread_id} 的 checkpoint 数据失败: {exc}")

    async def cleanup_orphans(self, limit: int = 500) -> int:
        """扫描 checkpoints 表里 thread_id 已经不在 conversations 表的孤儿数据并清理。

        用于处理"这次修复上线前就已经产生"的历史孤儿数据，以及兜底其它未来
        可能遗漏调用 `delete_for_thread` 的删除路径。

        Args:
            limit: 单次最多处理的孤儿 thread 数量，避免一次扫到的量过大。

        Returns:
            本次清理的孤儿 thread 数量；查询失败返回 0。
        """
        try:
            rows = await self._pool.fetch(
                "SELECT DISTINCT thread_id FROM checkpoints c "
                "WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE id = c.thread_id) "
                "LIMIT $1",
                limit,
            )
        except Exception as exc:
            logger.warning(f"[CheckpointCleanup] 查询孤儿 checkpoint 数据失败: {exc}")
            return 0

        for row in rows:
            await self.delete_for_thread(row["thread_id"])
        return len(rows)


_cleanup: CheckpointCleanup | None = None


def init_checkpoint_cleanup(pool: asyncpg.Pool, checkpointer: BaseCheckpointSaver) -> CheckpointCleanup:
    """应用启动时调用一次，构造并登记全局单例。"""
    global _cleanup
    _cleanup = CheckpointCleanup(pool, checkpointer)
    return _cleanup


def get_checkpoint_cleanup() -> CheckpointCleanup:
    """返回全局唯一的 CheckpointCleanup 实例。

    Raises:
        RuntimeError: init_checkpoint_cleanup() 尚未被调用。
    """
    if _cleanup is None:
        raise RuntimeError("CheckpointCleanup 尚未初始化，请确认应用已完成启动")
    return _cleanup
