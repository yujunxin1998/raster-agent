"""`MemoryContextBuilder` 单元测试：mock `BaseMemoryStore`/`UserProfileStore`。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.agent_core.memory.memory_context_builder import MemoryContextBuilder

_EMPTY_PROFILE = {
    "work_context": "", "personal_context": "", "top_of_mind": "",
    "recent_months": "", "earlier_context": "", "long_term_background": "",
}


def _builder(memory_store=None, profile_store=None, max_context_tokens=1000):
    memory_store = memory_store or MagicMock()
    profile_store = profile_store or MagicMock()
    return MemoryContextBuilder(
        memory_store, profile_store,
        recall_candidate_k=20, max_recall=5, min_recall_score=0.3, max_context_tokens=max_context_tokens,
    ), memory_store, profile_store


async def test_build_returns_empty_when_nothing_to_show() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(return_value=[])
    profile_store.get = AsyncMock(return_value=None)

    result = await builder.build(user_id="user-1", query="你好")

    assert result == ""


async def test_build_skips_recall_when_query_empty() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock()
    profile_store.get = AsyncMock(return_value=None)

    await builder.build(user_id="user-1", query="")

    memory_store.search.assert_not_awaited()


async def test_build_renders_facts_and_profile_sections() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(return_value=[
        {"memory_type": "goal", "content": "用户计划重构记忆模块", "score": 0.9},
    ])
    profile_store.get = AsyncMock(return_value={**_EMPTY_PROFILE, "top_of_mind": "正在重构 Agent 记忆架构"})

    result = await builder.build(user_id="user-1", query="记忆模块进展如何")

    assert "<relevant_facts>" in result
    assert "[goal] 用户计划重构记忆模块" in result
    assert "<profile>" in result
    assert "当前关注：正在重构 Agent 记忆架构" in result
    assert result.startswith("<memory>")
    assert result.rstrip().endswith("</memory>")


async def test_build_filters_low_score_facts() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(return_value=[
        {"memory_type": "goal", "content": "低相关内容", "score": 0.1},
    ])
    profile_store.get = AsyncMock(return_value=None)

    result = await builder.build(user_id="user-1", query="随便问问")

    assert result == ""


async def test_build_drops_low_priority_profile_fields_when_over_budget() -> None:
    builder, memory_store, profile_store = _builder(max_context_tokens=15)
    memory_store.search = AsyncMock(return_value=[])
    profile_store.get = AsyncMock(return_value={
        **_EMPTY_PROFILE,
        "top_of_mind": "x" * 20,  # 优先级最高，即使超预算也保留（fact_lines/首个 profile 行不因为超预算被拒绝）
        "long_term_background": "y" * 20,  # 优先级最低，预算耗尽后应被丢弃
    })

    result = await builder.build(user_id="user-1", query="")

    assert "当前关注" in result
    assert "长期背景" not in result


async def test_build_caches_within_same_run() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(return_value=[
        {"memory_type": "goal", "content": "用户计划重构记忆模块", "score": 0.9},
    ])
    profile_store.get = AsyncMock(return_value=None)
    cache: dict = {}

    first = await builder.build(user_id="user-1", query="记忆模块进展如何", cache=cache)
    second = await builder.build(user_id="user-1", query="换一个问题也应该命中缓存", cache=cache)

    assert first == second
    memory_store.search.assert_awaited_once()  # 第二次调用直接命中缓存，没有真的再查一次


async def test_build_swallows_search_exception() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(side_effect=Exception("es down"))
    profile_store.get = AsyncMock(return_value=None)

    result = await builder.build(user_id="user-1", query="记忆模块进展如何")

    assert result == ""


async def test_build_swallows_profile_lookup_exception() -> None:
    builder, memory_store, profile_store = _builder()
    memory_store.search = AsyncMock(return_value=[
        {"memory_type": "goal", "content": "用户计划重构记忆模块", "score": 0.9},
    ])
    profile_store.get = AsyncMock(side_effect=Exception("db down"))

    result = await builder.build(user_id="user-1", query="记忆模块进展如何")

    assert "[goal] 用户计划重构记忆模块" in result
