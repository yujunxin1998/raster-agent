"""`UserProfileStore` 单元测试：mock asyncpg pool，不依赖真实 Postgres 连接。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.storage.user_profile_store import UserProfileStore


def _store(pool) -> UserProfileStore:
    return UserProfileStore(pool)


async def test_setup_creates_table() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()

    await _store(pool).setup()

    pool.execute.assert_awaited_once()
    sql = pool.execute.await_args.args[0]
    assert "CREATE TABLE IF NOT EXISTS user_memory_profile" in sql


async def test_get_returns_none_when_no_row() -> None:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)

    result = await _store(pool).get("user-1")

    assert result is None


async def test_get_returns_dict_when_row_exists() -> None:
    pool = MagicMock()
    row = {
        "work_context": "后端工程师", "personal_context": "偏好中文交流",
        "top_of_mind": "在做记忆模块重构", "recent_months": "", "earlier_context": "",
        "long_term_background": "", "updated_at": "2026-08-06T00:00:00",
    }
    pool.fetchrow = AsyncMock(return_value=row)

    result = await _store(pool).get("user-1")

    assert result["work_context"] == "后端工程师"
    pool.fetchrow.assert_awaited_once()
    args = pool.fetchrow.await_args.args
    assert args[1] == "user-1"


async def test_upsert_passes_all_fields_in_order() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()

    await _store(pool).upsert(
        "user-1",
        work_context="wc", personal_context="pc", top_of_mind="tom",
        recent_months="rm", earlier_context="ec", long_term_background="ltb",
    )

    pool.execute.assert_awaited_once()
    sql, *params = pool.execute.await_args.args
    assert "ON CONFLICT (user_id) DO UPDATE" in sql
    assert params == ["user-1", "wc", "pc", "tom", "rm", "ec", "ltb"]
