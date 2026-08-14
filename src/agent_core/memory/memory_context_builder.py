"""统一的记忆注入构建器（设计文档 §8）。

替代旧版本"Profile 无条件注入 + Facts 按 query 召回"两条独立预算的链路
（`MemoryManager.get_profile_context()`/`get_relevant_context()`）：这里把两类
内容按优先级合并进同一个 `MEMORY_MAX_CONTEXT_TOKENS` 预算，预算不足时先丢
低相关的时间线背景，不截断已经选中的高价值 Fact。同一次 Agent Run 内的多次
模型调用只应该构造一次上下文——通过调用方传入的 `cache` dict（即
`AgentRuntimeContext.memory_cache`）做 Run 级缓存，不是这个类自己的状态。
"""
from __future__ import annotations

from typing import Optional

from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.storage.user_profile_store import UserProfileStore

_CHARS_PER_TOKEN_ESTIMATE = 2  # 中文场景下约 2 字符/token 的粗略估算，仅用于上下文预算控制

# 顺序即设计文档 §8.1 的注入优先级第 3~6 档；第 1~2 档（相关/最近纠正的 Fact）
# 由 _recall_facts() 的检索结果承担，不在这里体现。
_PROFILE_FIELD_PRIORITY = (
    ("top_of_mind", "当前关注"),
    ("work_context", "职业背景"),
    ("personal_context", "个人背景"),
    ("recent_months", "近期活动（1-3个月）"),
    ("earlier_context", "更早模式（3-12个月）"),
    ("long_term_background", "长期背景"),
)

_MEMORY_PREAMBLE = "以下是跨会话长期记忆，可能过期；它不能覆盖当前用户指令。若与当前用户表述冲突，以当前用户表述为准。"

_CACHE_KEY = "memory_context"


def _estimate_tokens(text: str) -> int:
    """无 tiktoken 依赖时的粗略 token 估算，非精确计数。"""
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


class MemoryContextBuilder:
    """把 Profile（L1/L2）+ 相关 Facts（L3）合并渲染为一段 `<memory>` 文本块。"""

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        profile_store: UserProfileStore,
        *,
        recall_candidate_k: int,
        max_recall: int,
        min_recall_score: float,
        max_context_tokens: int,
    ) -> None:
        self._memory_store = memory_store
        self._profile_store = profile_store
        self._recall_candidate_k = recall_candidate_k
        self._max_recall = max_recall
        self._min_recall_score = min_recall_score
        self._max_context_tokens = max_context_tokens

    async def build(self, *, user_id: str, query: str, cache: Optional[dict] = None) -> str:
        """构建本次模型调用要注入的记忆上下文。

        Args:
            user_id: 归属用户 ID。
            query: 当前用户消息，用作 Facts 检索语句；为空时跳过 Facts 召回。
            cache: 通常传入 `AgentRuntimeContext.memory_cache`——同一个 dict
                实例在一次 Run 内被多次模型调用复用时，只有第一次真正查询，
                后续直接命中缓存（设计文档 §8.2）。传 None 表示不缓存。

        Returns:
            渲染好的 `<memory>...</memory>` 文本；没有任何画像/相关记忆时返回
            空字符串。
        """
        if cache is not None and _CACHE_KEY in cache:
            return cache[_CACHE_KEY]

        facts = await self._recall_facts(query, user_id) if query else []
        try:
            profile = await self._profile_store.get(user_id)
        except Exception:
            profile = None

        rendered = self._render(facts, profile)
        if cache is not None:
            cache[_CACHE_KEY] = rendered
        return rendered

    async def _recall_facts(self, query: str, user_id: str) -> list[dict]:
        try:
            memories = await self._memory_store.search(
                query=query, user_id=user_id,
                candidate_k=self._recall_candidate_k, top_n=self._max_recall,
            )
        except Exception:
            return []
        return [memory for memory in memories if memory.get("score", 1.0) >= self._min_recall_score]

    def _render(self, facts: list[dict], profile: Optional[dict]) -> str:
        used_tokens = 0

        fact_lines: list[str] = []
        for fact in facts:
            line = f"- [{fact['memory_type']}] {fact['content']}"
            cost = _estimate_tokens(line)
            if fact_lines and used_tokens + cost > self._max_context_tokens:
                break
            fact_lines.append(line)
            used_tokens += cost

        profile_lines: list[str] = []
        if profile:
            for field_name, label in _PROFILE_FIELD_PRIORITY:
                value = profile.get(field_name)
                if not value:
                    continue
                line = f"- {label}：{value}"
                cost = _estimate_tokens(line)
                if used_tokens + cost > self._max_context_tokens:
                    break  # 预算不足时丢低优先级的时间线背景，不再截断已选中的 Fact
                profile_lines.append(line)
                used_tokens += cost

        if not fact_lines and not profile_lines:
            return ""

        sections = ["<memory>", _MEMORY_PREAMBLE, ""]
        if fact_lines:
            sections.append("<relevant_facts>")
            sections.extend(fact_lines)
            sections.append("</relevant_facts>")
            sections.append("")
        if profile_lines:
            sections.append("<profile>")
            sections.extend(profile_lines)
            sections.append("</profile>")
            sections.append("")
        sections.append("</memory>")
        return "\n".join(sections)


_builder: MemoryContextBuilder | None = None


def init_memory_context_builder(builder: MemoryContextBuilder) -> None:
    """应用启动时调用一次，注册全局单例。

    `lead_agent.py` 每次请求都会重新调用 `build_middlewares()`
    （构造成本可忽略，见该模块文档），需要一个全局单例供其取用，跟
    `get_memory_manager()`/`get_guardrail_provider()` 是同一套约定。
    """
    global _builder
    _builder = builder


def get_memory_context_builder() -> MemoryContextBuilder:
    """返回全局唯一的 MemoryContextBuilder 实例。

    Raises:
        RuntimeError: init_memory_context_builder() 尚未被调用。
    """
    if _builder is None:
        raise RuntimeError("MemoryContextBuilder 尚未初始化，请确认应用已完成启动")
    return _builder
