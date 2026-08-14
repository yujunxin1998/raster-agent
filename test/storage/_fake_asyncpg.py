"""`asyncpg.Pool.acquire()` + `Connection.transaction()` 的测试替身。

Memory v2 的新 Store（`user_profile_store.apply_patches`、`memory_event_store`、
`memory_update_job_store`、`user_memory_fact_store`）都需要在一个事务内执行
多条 SQL，用简单的 `MagicMock()` 无法表达 `async with pool.acquire() as conn:
async with conn.transaction():` 这层嵌套的异步上下文管理协议，故提供这组最小
的测试替身，只还原被测代码实际用到的接口（`fetchrow`/`fetch`/`fetchval`/
`execute`/`acquire`/`transaction`）。
"""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import AsyncMock


class _NullAsyncContext:
    async def __aenter__(self) -> "_NullAsyncContext":
        return self

    async def __aexit__(self, *_exc_info: object) -> bool:
        return False


class FakeConnection:
    """模拟一条已获取的连接，`fetchrow`/`fetch`/`fetchval`/`execute` 均为 AsyncMock。"""

    def __init__(
        self,
        fetchrow_result: Optional[dict] = None,
        fetch_result: Optional[list[dict]] = None,
        fetchval_result: Any = None,
        execute_result: str = "UPDATE 1",
    ) -> None:
        self.fetchrow = AsyncMock(return_value=fetchrow_result)
        self.fetch = AsyncMock(return_value=fetch_result or [])
        self.fetchval = AsyncMock(return_value=fetchval_result)
        self.execute = AsyncMock(return_value=execute_result)

    def transaction(self) -> _NullAsyncContext:
        return _NullAsyncContext()


class _AcquireContext:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc_info: object) -> bool:
        return False


class FakePool:
    """模拟 `asyncpg.Pool`：`acquire()` 始终返回同一个 `FakeConnection`。"""

    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self.fetchrow = conn.fetchrow
        self.fetch = conn.fetch
        self.fetchval = conn.fetchval
        self.execute = conn.execute

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.conn)
