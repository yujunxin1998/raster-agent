"""`MemoryManager` 单元测试（Memory v2 收敛版）：mock fact_store/profile_store/compressor。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.agent_core.memory.memory_manager import MemoryManager


def _manager(fact_store=None, compressor=None, profile_store=None, **overrides):
    fact_store = fact_store or MagicMock()
    compressor = compressor or MagicMock()
    profile_store = profile_store or MagicMock()
    defaults = {"importance_threshold": 5, "sensitive_filter_enabled": True}
    defaults.update(overrides)
    return MemoryManager(fact_store, compressor, profile_store, **defaults), fact_store, compressor, profile_store


async def test_get_profile_delegates_to_profile_store() -> None:
    manager, _, _, profile_store = _manager()
    profile_store.get = AsyncMock(return_value={"work_context": "x"})

    result = await manager.get_profile("user-1")

    assert result == {"work_context": "x"}
    profile_store.get.assert_awaited_once_with("user-1")


def test_is_sensitive_content_respects_flag() -> None:
    manager, *_ = _manager(sensitive_filter_enabled=False)

    assert manager.is_sensitive_content("api_key=sk-abcdefghijklmnop1234") is False


def test_is_sensitive_content_detects_when_enabled() -> None:
    manager, *_ = _manager(sensitive_filter_enabled=True)

    assert manager.is_sensitive_content("api_key=sk-abcdefghijklmnop1234") is True


def test_is_sensitive_content_false_for_empty() -> None:
    manager, *_ = _manager()

    assert manager.is_sensitive_content(None) is False
    assert manager.is_sensitive_content("") is False


async def test_compress_after_chat_writes_summary_as_fact() -> None:
    manager, fact_store, compressor, _ = _manager()
    compressor.compress_if_needed = AsyncMock(return_value={"summary": "本轮摘要", "compressed_count": 12})
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "active"})

    await manager.compress_after_chat(graph=MagicMock(), config={}, user_id="user-1", conversation_id="conv-1")

    fact_store.add_or_reinforce.assert_awaited_once()
    kwargs = fact_store.add_or_reinforce.await_args.kwargs
    assert kwargs["content"] == "本轮摘要"
    assert kwargs["category"] == "summary"
    assert kwargs["status"] == "active"
    assert kwargs["confidence"] == 1.0


async def test_compress_after_chat_noop_when_not_triggered() -> None:
    manager, fact_store, compressor, _ = _manager()
    compressor.compress_if_needed = AsyncMock(return_value=None)

    await manager.compress_after_chat(graph=MagicMock(), config={}, user_id="user-1", conversation_id="conv-1")

    fact_store.add_or_reinforce.assert_not_called()


async def test_compress_after_chat_swallows_compressor_exception() -> None:
    manager, fact_store, compressor, _ = _manager()
    compressor.compress_if_needed = AsyncMock(side_effect=Exception("boom"))

    await manager.compress_after_chat(graph=MagicMock(), config={}, user_id="user-1", conversation_id="conv-1")

    fact_store.add_or_reinforce.assert_not_called()


async def test_compress_after_chat_swallows_fact_store_write_failure() -> None:
    manager, fact_store, compressor, _ = _manager()
    compressor.compress_if_needed = AsyncMock(return_value={"summary": "摘要", "compressed_count": 5})
    fact_store.add_or_reinforce = AsyncMock(side_effect=Exception("db down"))

    await manager.compress_after_chat(graph=MagicMock(), config={}, user_id="user-1", conversation_id="conv-1")  # 不应抛出


async def test_compress_messages_after_chat_returns_state_update_on_success() -> None:
    manager, fact_store, compressor, _ = _manager()
    outcome = SimpleNamespace(summary="摘要", compressed_count=8, state_update={"messages": []})
    compressor.compress_messages = AsyncMock(return_value=outcome)
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "active"})

    result = await manager.compress_messages_after_chat(messages=[], user_id="user-1", conversation_id="conv-1")

    assert result == {"messages": []}


async def test_compress_messages_after_chat_returns_none_when_not_triggered() -> None:
    manager, _, compressor, _ = _manager()
    compressor.compress_messages = AsyncMock(return_value=None)

    result = await manager.compress_messages_after_chat(messages=[], user_id="user-1", conversation_id="conv-1")

    assert result is None


async def test_compress_messages_after_chat_returns_none_when_save_fails() -> None:
    manager, fact_store, compressor, _ = _manager()
    outcome = SimpleNamespace(summary="摘要", compressed_count=8, state_update={"messages": []})
    compressor.compress_messages = AsyncMock(return_value=outcome)
    fact_store.add_or_reinforce = AsyncMock(side_effect=Exception("db down"))

    result = await manager.compress_messages_after_chat(messages=[], user_id="user-1", conversation_id="conv-1")

    assert result is None
