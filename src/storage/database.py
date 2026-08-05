"""数据库连接池的生命周期管理。

进程内所有 Store 共用同一个 asyncpg.Pool，由 DatabasePoolManager 统一持有，
避免每个 Store 各自 create_pool 造成连接数失控。
"""
from __future__ import annotations

import asyncpg
from loguru import logger

from src.common.exceptions import ConfigurationError


class DatabasePoolManager:
    """asyncpg 连接池的持有者与生命周期管理者。

    Attributes:
        pool: 底层 asyncpg 连接池实例。
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @property
    def pool(self) -> asyncpg.Pool:
        """返回底层连接池，供各 Store 执行 SQL。"""
        return self._pool

    async def close(self) -> None:
        """关闭连接池，应用关闭阶段调用一次。"""
        await self._pool.close()
        logger.info("[DatabasePoolManager] 连接池已关闭")

    @classmethod
    async def create(cls, database_url: str) -> "DatabasePoolManager":
        """创建连接池并返回管理器实例。

        Args:
            database_url: PostgreSQL 连接串，对应配置项 `DATABASE_URL`。

        Returns:
            已完成连接池创建的 DatabasePoolManager 实例。

        Raises:
            ConfigurationError: database_url 为空。
        """
        if not database_url:
            raise ConfigurationError("DATABASE_URL 未配置，无法创建数据库连接池")
        pool = await asyncpg.create_pool(database_url)
        logger.info("[DatabasePoolManager] 连接池创建完成")
        return cls(pool)


_manager: DatabasePoolManager | None = None


async def init_database_pool(database_url: str) -> DatabasePoolManager:
    """应用启动时调用一次，创建并注册全局唯一的连接池管理器。

    Args:
        database_url: PostgreSQL 连接串。

    Returns:
        创建好的 DatabasePoolManager，同时也已注册为全局单例。
    """
    global _manager
    _manager = await DatabasePoolManager.create(database_url)
    return _manager


def get_database_pool_manager() -> DatabasePoolManager:
    """返回全局唯一的连接池管理器。

    Raises:
        RuntimeError: init_database_pool() 尚未被调用。
    """
    if _manager is None:
        raise RuntimeError("DatabasePoolManager 尚未初始化，请确认应用已完成启动")
    return _manager


def get_database_pool() -> asyncpg.Pool:
    """快捷方法：直接返回底层连接池，等价于 get_database_pool_manager().pool。"""
    return get_database_pool_manager().pool


async def close_database_pool() -> None:
    """应用关闭时调用一次，释放全局连接池。未初始化时静默跳过。"""
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None
