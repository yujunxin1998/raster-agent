"""`MessageStore` 新增能力的单元测试：`add_message` 返回自增 ID、
`delete_last_assistant_message`、`set_feedback`。mock asyncpg pool，不依赖
真实 Postgres 连接。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.storage.message_store import MessageStore


def _store(pool) -> MessageStore:
    return MessageStore(pool)


async def test_add_message_returns_new_id() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=7)

    message_id = await _store(pool).add_message("conv-1", "assistant", "你好")

    assert message_id == 7
    pool.fetchval.assert_awaited_once()
    sql = pool.fetchval.await_args.args[0]
    assert "RETURNING id" in sql


async def test_add_message_returns_none_on_failure() -> None:
    pool = MagicMock()
    pool.fetchval = AsyncMock(side_effect=RuntimeError("db down"))

    message_id = await _store(pool).add_message("conv-1", "user", "你好")

    assert message_id is None


async def test_get_messages_includes_id_and_feedback() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[
        {
            "id": 1, "role": "assistant", "content": "回复", "thinking_content": None,
            "tool_calls": None, "references": None, "feedback": "like", "created_at": "2026-08-06T00:00:00",
        }
    ])

    rows = await _store(pool).get_messages("conv-1")

    assert rows[0]["id"] == 1
    assert rows[0]["feedback"] == "like"


async def test_delete_last_assistant_message_scopes_by_role_and_conversation() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock()

    await _store(pool).delete_last_assistant_message("conv-1")

    pool.execute.assert_awaited_once()
    sql = pool.execute.await_args.args[0]
    assert "role = 'assistant'" in sql
    assert pool.execute.await_args.args[1] == "conv-1"


async def test_set_feedback_returns_true_when_row_updated() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock(return_value="UPDATE 1")

    ok = await _store(pool).set_feedback(42, "conv-1", "like")

    assert ok is True
    pool.execute.assert_awaited_once()
    args = pool.execute.await_args.args
    assert args[1:] == ("like", 42, "conv-1")


async def test_set_feedback_returns_false_when_no_row_matched() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock(return_value="UPDATE 0")

    ok = await _store(pool).set_feedback(999, "conv-1", "dislike")

    assert ok is False


async def test_set_feedback_can_clear_with_none() -> None:
    pool = MagicMock()
    pool.execute = AsyncMock(return_value="UPDATE 1")

    ok = await _store(pool).set_feedback(42, "conv-1", None)

    assert ok is True
    args = pool.execute.await_args.args
    assert args[1] is None
