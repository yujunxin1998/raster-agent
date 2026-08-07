"""`MemoryExtractor` 单元测试：mock LLM 调用 + `BaseMemoryStore`，验证五分类
提取、"记住"类关键词兜底、重复/冲突检测。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.agent_core.memory.memory_extractor import MemoryExtractor


def _extractor(**overrides) -> MemoryExtractor:
    defaults = dict(
        model_name="test-model", provider="deepseek", api_key="k", base_url="u",
        default_importance_threshold=5, duplicate_score_threshold=0.92,
        conflict_score_threshold=0.75, low_confidence_margin=2,
    )
    defaults.update(overrides)
    return MemoryExtractor(**defaults)


def _fake_store(search_result=None) -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=search_result or [])
    store.save = AsyncMock(return_value="new-id")
    store.update = AsyncMock()
    return store


async def test_extracts_and_saves_new_five_category() -> None:
    extractor = _extractor()
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "偏好使用 Vim", "memory_type": "preference", "importance": 7},
    ])
    store = _fake_store()

    await extractor.extract_and_save(
        store, user_message="我喜欢用 Vim", ai_response="好的", user_id="u1", conversation_id="c1",
    )

    store.save.assert_awaited_once()
    kwargs = store.save.await_args.kwargs
    assert kwargs["memory_type"] == "preference"
    assert kwargs["content"] == "偏好使用 Vim"
    assert kwargs["status"] == "active"


async def test_low_importance_result_saved_as_pending() -> None:
    """重要度刚过滤阈值不远（在 low_confidence_margin 容差内）时应存为 pending。"""
    extractor = _extractor(default_importance_threshold=5, low_confidence_margin=2)
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "精通 Rust", "memory_type": "knowledge", "importance": 6},
    ])
    store = _fake_store()

    await extractor.extract_and_save(
        store, user_message="我精通 Rust", ai_response="厉害", user_id="u1", conversation_id="c1",
    )

    assert store.save.await_args.kwargs["status"] == "pending"


async def test_below_threshold_result_is_skipped() -> None:
    extractor = _extractor(default_importance_threshold=5)
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "随口一提", "memory_type": "context", "importance": 2},
    ])
    store = _fake_store()

    await extractor.extract_and_save(
        store, user_message="随口一提", ai_response="嗯", user_id="u1", conversation_id="c1",
    )

    store.save.assert_not_awaited()


async def test_missing_memory_type_defaults_to_context() -> None:
    extractor = _extractor()
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "在字节跳动工作", "importance": 7},  # 缺 memory_type
    ])
    store = _fake_store()

    await extractor.extract_and_save(
        store, user_message="我在字节跳动工作", ai_response="了解", user_id="u1", conversation_id="c1",
    )

    assert store.save.await_args.kwargs["memory_type"] == "context"


async def test_remember_keyword_fallback_saves_as_goal_with_forced_importance() -> None:
    """LLM 没提取到内容，但命中"记住"类关键词时强制保存原文，
    分类固定为 goal（不再依赖已删除的 instruction/correction 分类）。
    """
    extractor = _extractor()
    extractor._call_llm_extract = AsyncMock(return_value=[])
    store = _fake_store()

    await extractor.extract_and_save(
        store, user_message="以后都用中文回答我", ai_response="好的", user_id="u1", conversation_id="c1",
    )

    store.save.assert_awaited_once()
    kwargs = store.save.await_args.kwargs
    assert kwargs["memory_type"] == "goal"
    assert kwargs["importance"] == 8
    assert kwargs["content"] == "以后都用中文回答我"


async def test_duplicate_content_is_skipped() -> None:
    extractor = _extractor(duplicate_score_threshold=0.9)
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "偏好使用 Vim", "memory_type": "preference", "importance": 7},
    ])
    store = _fake_store(search_result=[{"id": "old-1", "content": "偏好使用 Vim", "score": 0.99}])

    await extractor.extract_and_save(
        store, user_message="我喜欢用 Vim", ai_response="好的", user_id="u1", conversation_id="c1",
    )

    store.save.assert_not_awaited()


async def test_conflicting_content_archives_old_memory() -> None:
    extractor = _extractor(duplicate_score_threshold=0.92, conflict_score_threshold=0.75)
    extractor._call_llm_extract = AsyncMock(return_value=[
        {"content": "现在偏好使用 Neovim", "memory_type": "preference", "importance": 7},
    ])
    store = _fake_store(search_result=[{"id": "old-1", "content": "偏好使用 Vim", "score": 0.8}])

    await extractor.extract_and_save(
        store, user_message="我改用 Neovim 了", ai_response="好的", user_id="u1", conversation_id="c1",
    )

    store.save.assert_awaited_once()
    store.update.assert_awaited_once_with(
        "old-1", user_id="u1", status="archived", superseded_by="new-id",
        source="extractor", trace_id=None,
    )
