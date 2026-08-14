"""ES Outbox 投影器（设计文档 §7.6）。

PostgreSQL `user_memory_fact` 是 Facts 的真相源；`FactProjector` 持续消费
`memory_outbox` 表，把变更异步同步进 Elasticsearch 的可检索投影。ES 写入失败
只重试 Outbox 记录本身，不回滚已经提交的 PostgreSQL 事务——这是刻意的最终
一致性：ES 故障会让新增/更新的记忆暂时检索不到，但不会丢失数据（下次投影器
轮询会重试），符合设计文档"PostgreSQL 是主存，ES 只是投影"的定位。
"""
from __future__ import annotations

import asyncio

from loguru import logger

from src.agent_core.memory.elasticsearch_memory_store import ElasticsearchMemoryStore
from src.common.constants import MemoryStatus
from src.storage.user_memory_fact_store import UserMemoryFactStore

_ACTIVE_STATUS = MemoryStatus.ACTIVE.value


class FactProjector:
    """把 `memory_outbox` 里的 Fact 变更异步同步进 ES 投影。"""

    def __init__(
        self,
        fact_store: UserMemoryFactStore,
        es_store: ElasticsearchMemoryStore,
        *,
        poll_interval_seconds: float,
        batch_size: int,
        max_attempts: int,
    ) -> None:
        self._fact_store = fact_store
        self._es_store = es_store
        self._poll_interval_seconds = poll_interval_seconds
        self._batch_size = batch_size
        self._max_attempts = max_attempts

    async def run_forever(self) -> None:
        """持续轮询未处理的 Outbox 记录，供 `asyncio.create_task()` 常驻运行。"""
        while True:
            try:
                processed = await self.run_once()
            except Exception as exc:
                logger.warning(f"[FactProjector] 轮询异常: {exc}")
                processed = 0
            if processed == 0:
                await asyncio.sleep(self._poll_interval_seconds)

    async def run_once(self) -> int:
        """处理一批未同步的 Outbox 记录。

        Returns:
            本次实际取到的记录条数（用于决定 `run_forever()` 是否需要等待）。
        """
        batch = await self._fact_store.claim_outbox_batch(self._batch_size)
        for item in batch:
            await self._process_item(item)
        return len(batch)

    async def _process_item(self, item: dict) -> None:
        try:
            snapshot = item["snapshot"]
            if snapshot.get("status") == _ACTIVE_STATUS:
                await self._es_store._index_projection(snapshot)
            else:
                await self._es_store._remove_projection(snapshot["fact_id"])
            await self._fact_store.mark_outbox_processed(item["outbox_id"])
        except Exception as exc:
            logger.warning(f"[FactProjector] 同步失败 outbox_id={item['outbox_id']}: {exc}")
            await self._fact_store.mark_outbox_failed(item["outbox_id"], str(exc))
