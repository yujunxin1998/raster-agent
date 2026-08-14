"""`UserProfileStore` 单元测试：mock asyncpg pool，不依赖真实 Postgres 连接。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from _fake_asyncpg import FakeConnection, FakePool

from src.storage.user_profile_store import UserProfileStore


def _store(pool) -> UserProfileStore:
    return UserProfileStore(pool)


async def test_setup_creates_table_and_field_meta_table() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()

    await _store(pool).setup()

    assert pool.execute.await_count == 3
    sqls = [call.args[0] for call in pool.execute.await_args_list]
    assert any("CREATE TABLE IF NOT EXISTS user_memory_profile (" in sql for sql in sqls)
    assert any("ALTER TABLE user_memory_profile ADD COLUMN IF NOT EXISTS revision" in sql for sql in sqls)
    assert any("CREATE TABLE IF NOT EXISTS user_memory_profile_field_meta" in sql for sql in sqls)


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


async def test_get_with_revision_defaults_to_zero_for_new_user() -> None:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)

    result = await _store(pool).get_with_revision("user-1")

    assert result["revision"] == 0
    assert result["work_context"] == ""


async def test_apply_patches_inserts_new_row_when_expected_revision_zero() -> None:
    conn = FakeConnection(fetchrow_result=None)
    pool = FakePool(conn)

    ok = await _store(pool).apply_patches(
        "user-1",
        [{"field": "work_context", "op": "set", "value": "后端工程师", "confidence": 0.9}],
        expected_revision=0,
        source_event_id="evt-1",
    )

    assert ok is True
    insert_call = conn.execute.await_args_list[0]
    assert "INSERT INTO user_memory_profile" in insert_call.args[0]
    assert insert_call.args[1] == "user-1"
    assert insert_call.args[2] == "后端工程师"  # work_context 是第一个字段


async def test_apply_patches_rejects_new_user_insert_when_row_already_exists() -> None:
    conn = FakeConnection(fetchrow_result={
        "work_context": "existing", "personal_context": "", "top_of_mind": "",
        "recent_months": "", "earlier_context": "", "long_term_background": "", "revision": 1,
    })
    pool = FakePool(conn)

    ok = await _store(pool).apply_patches(
        "user-1", [{"field": "work_context", "op": "set", "value": "x"}], expected_revision=0,
    )

    assert ok is False
    conn.execute.assert_not_awaited()


async def test_apply_patches_rejects_on_revision_mismatch() -> None:
    conn = FakeConnection(fetchrow_result={
        "work_context": "existing", "personal_context": "", "top_of_mind": "",
        "recent_months": "", "earlier_context": "", "long_term_background": "", "revision": 5,
    })
    pool = FakePool(conn)

    ok = await _store(pool).apply_patches(
        "user-1", [{"field": "work_context", "op": "set", "value": "x"}], expected_revision=3,
    )

    assert ok is False
    conn.execute.assert_not_awaited()


async def test_apply_patches_updates_and_bumps_revision_on_match() -> None:
    conn = FakeConnection(fetchrow_result={
        "work_context": "old", "personal_context": "", "top_of_mind": "",
        "recent_months": "", "earlier_context": "", "long_term_background": "", "revision": 2,
    })
    pool = FakePool(conn)

    ok = await _store(pool).apply_patches(
        "user-1",
        [{"field": "work_context", "op": "merge", "value": "new", "confidence": 0.8}],
        expected_revision=2,
        source_event_id="evt-2",
    )

    assert ok is True
    update_call = conn.execute.await_args_list[0]
    assert "UPDATE user_memory_profile SET" in update_call.args[0]
    assert update_call.args[2] == "new"  # work_context 新值
    assert update_call.args[-1] == 2  # WHERE revision = $8（expected_revision）
    field_meta_call = conn.execute.await_args_list[1]
    assert "user_memory_profile_field_meta" in field_meta_call.args[0]


async def test_apply_patches_clear_sets_field_empty() -> None:
    conn = FakeConnection(fetchrow_result={
        "work_context": "old", "personal_context": "", "top_of_mind": "",
        "recent_months": "", "earlier_context": "", "long_term_background": "", "revision": 0,
    })
    pool = FakePool(conn)

    await _store(pool).apply_patches(
        "user-1", [{"field": "work_context", "op": "clear"}], expected_revision=0,
    )

    update_call = conn.execute.await_args_list[0]
    assert update_call.args[2] == ""


async def test_apply_patches_empty_list_is_noop() -> None:
    pool = MagicMock()
    pool.acquire = MagicMock()

    ok = await _store(pool).apply_patches("user-1", [], expected_revision=0)

    assert ok is True
    pool.acquire.assert_not_called()


async def test_get_field_sources_returns_rows() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[
        {"field_name": "work_context", "updated_at": "2026-08-06T00:00:00", "source_event_id": "evt-1", "confidence": 0.9},
    ])

    result = await _store(pool).get_field_sources("user-1")

    assert result[0]["field_name"] == "work_context"


async def test_delete_removes_profile_and_field_meta_rows() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()

    await _store(pool).delete("user-1")

    assert pool.execute.await_count == 2
    sqls = [call.args[0] for call in pool.execute.await_args_list]
    assert any("DELETE FROM user_memory_profile WHERE" in sql for sql in sqls)
    assert any("DELETE FROM user_memory_profile_field_meta" in sql for sql in sqls)
