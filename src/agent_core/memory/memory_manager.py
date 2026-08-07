"""记忆机制对外的统一门面。

原样迁移自 `src/core/memory/manager.py`：组合记忆存储、提取器、压缩器，
对外提供 `get_relevant_context` / `extract_after_chat` / `compress_after_chat`
三个高层接口。与原实现的差异仅在于 extractor/compressor 现在是显式注入的
`MemoryExtractor`/`MemoryCompressor` 实例，而不是模块级函数。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.agent_core.memory.memory_compressor import MemoryCompressor
from src.agent_core.memory.memory_extractor import MemoryExtractor
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.agent_core.memory.user_profile_updater import UserProfileUpdater
from src.common.constants import MemoryAuditAction, MemorySource
from src.storage.memory_audit_store import get_memory_audit_store
from src.storage.user_profile_store import UserProfileStore

_CHARS_PER_TOKEN_ESTIMATE = 2  # 中文场景下约 2 字符/token 的粗略估算，仅用于上下文预算控制


def _estimate_tokens(text: str) -> int:
    """无 tiktoken 依赖时的粗略 token 估算，非精确计数。"""
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


class MemoryManager:
    """记忆机制的统一门面，供上层业务代码调用。"""

    def __init__(
        self,
        store: BaseMemoryStore,
        extractor: MemoryExtractor,
        compressor: MemoryCompressor,
        profile_store: UserProfileStore,
        profile_updater: UserProfileUpdater,
        *,
        memory_enabled: bool,
        injection_enabled: bool,
        auto_extract_enabled: bool,
        profile_update_enabled: bool,
        recall_candidate_k: int,
        max_recall: int,
        min_recall_score: float,
        max_context_tokens: int,
        importance_threshold: int,
        sensitive_filter_enabled: bool,
    ) -> None:
        """初始化记忆管理器。

        Args:
            store: 长期记忆存储实现（L3 Facts，可检索）。
            extractor: 后处理提取器（L3 Facts）。
            compressor: 会话压缩器。
            profile_store: 用户画像/时间线存储（L1/L2，不可检索，按 user_id 整体读写）。
            profile_updater: 画像/时间线更新器（L1/L2）。
            memory_enabled: 长期记忆总开关。
            injection_enabled: 是否在对话前注入相关长期记忆。
            auto_extract_enabled: 是否启用后处理自动提取（L3 Facts）。
            profile_update_enabled: 是否启用画像/时间线实时更新（L1/L2），
                与 Facts 提取各自独立的 LLM 调用，可单独关闭。
            recall_candidate_k: 检索粗召回候选数量。
            max_recall: 精排后最终召回条数。
            min_recall_score: 召回结果的最低相关性分数，低于此值被过滤。
            max_context_tokens: 注入上下文的 token 预算。
            importance_threshold: 压缩摘要写入长期记忆时使用的重要度。
            sensitive_filter_enabled: 是否启用敏感信息检测。
        """
        self._store = store
        self._extractor = extractor
        self._compressor = compressor
        self._profile_store = profile_store
        self._profile_updater = profile_updater
        self._memory_enabled = memory_enabled
        self._injection_enabled = injection_enabled
        self._auto_extract_enabled = auto_extract_enabled
        self._profile_update_enabled = profile_update_enabled
        self._recall_candidate_k = recall_candidate_k
        self._max_recall = max_recall
        self._min_recall_score = min_recall_score
        self._max_context_tokens = max_context_tokens
        self._importance_threshold = importance_threshold
        self._sensitive_filter_enabled = sensitive_filter_enabled

    @property
    def store(self) -> BaseMemoryStore:
        """暴露底层存储，供管理 API（增删改查）直接调用。"""
        return self._store

    async def get_relevant_context(
        self,
        query: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """检索与当前消息相关的长期记忆，格式化后返回。

        Args:
            query: 当前用户消息，用作检索语句。
            user_id: 归属用户 ID。
            conversation_id: 当前会话 ID，用于审计记录。
            trace_id: 链路追踪 ID。

        Returns:
            格式化后的记忆上下文文本；无相关记忆或功能关闭时返回空字符串。
        """
        if not self._memory_enabled or not self._injection_enabled:
            return ""

        try:
            memories = await self._store.search(
                query=query, user_id=user_id,
                candidate_k=self._recall_candidate_k, top_n=self._max_recall,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 记忆检索失败，跳过注入: {exc}")
            return ""

        memories = [memory for memory in memories if memory.get("score", 1.0) >= self._min_recall_score]
        if not memories:
            return ""

        kept = self._fit_within_token_budget(memories)
        if not kept:
            return ""

        await self._record_audit(
            action=MemoryAuditAction.INJECT, user_id=user_id, source=MemorySource.API,
            conversation_id=conversation_id, trace_id=trace_id,
            detail=f"memory_ids={[memory['id'] for memory in kept]}",
        )

        return "\n".join(f"- [{memory['memory_type']}] {memory['content']}" for memory in kept)

    def _fit_within_token_budget(self, memories: list[dict]) -> list[dict]:
        kept: list[dict] = []
        used_tokens = 0
        for memory in memories:
            line = f"- [{memory['memory_type']}] {memory['content']}"
            cost = _estimate_tokens(line)
            if kept and used_tokens + cost > self._max_context_tokens:
                break
            kept.append(memory)
            used_tokens += cost
        return kept

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """查询用户画像与时间线原始记录，供管理 API 直接返回。

        Args:
            user_id: 归属用户 ID。

        Returns:
            六个字段 + `updated_at` 的字典；用户还没生成过画像时返回 None。
        """
        return await self._profile_store.get(user_id)

    async def get_profile_context(self, user_id: str) -> str:
        """取用户画像与时间线，格式化为拼进 system_prompt 的文本块。

        跟 `get_relevant_context`（L3 Facts，按检索相关性决定是否注入）不同，
        这里是无条件注入——画像/时间线代表"这个用户是谁、最近在做什么"这类
        稳定身份背景，不依赖当前这句话具体问了什么。

        Args:
            user_id: 归属用户 ID。

        Returns:
            格式化后的文本；用户还没有画像记录、功能关闭时返回空字符串。
        """
        if not self._memory_enabled or not self._injection_enabled:
            return ""

        try:
            profile = await self._profile_store.get(user_id)
        except Exception as exc:
            logger.warning(f"[MemoryManager] 画像查询失败，跳过注入: {exc}")
            return ""

        if not profile:
            return ""

        lines = ["## 用户画像与时间线"]
        profile_lines = [
            ("职业背景", profile.get("work_context", "")),
            ("个人背景", profile.get("personal_context", "")),
            ("当前关注", profile.get("top_of_mind", "")),
            ("近期活动（1-3个月）", profile.get("recent_months", "")),
            ("更早模式（3-12个月）", profile.get("earlier_context", "")),
            ("长期背景", profile.get("long_term_background", "")),
        ]
        for label, value in profile_lines:
            if value:
                lines.append(f"- {label}：{value}")

        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    async def update_profile_after_chat(self, user_message: str, ai_response: str, user_id: str) -> None:
        """对话结束后，合并本轮对话更新用户画像与时间线。

        Args:
            user_message: 用户本轮发言。
            ai_response: AI 本轮回复。
            user_id: 归属用户 ID。
        """
        if not self._memory_enabled or not self._profile_update_enabled:
            return
        try:
            await self._profile_updater.update_after_chat(
                store=self._profile_store, user_message=user_message,
                ai_response=ai_response, user_id=user_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 画像更新异常，已跳过: {exc}")

    async def extract_after_chat(
        self,
        user_message: str,
        ai_response: str,
        user_id: str,
        conversation_id: str,
        prev_ai_response: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """对话结束后，异步提取并保存值得长期记忆的内容。

        Args:
            user_message: 用户本轮发言。
            ai_response: AI 本轮回复。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            prev_ai_response: 上一轮 AI 回复，用于理解确认性发言。
            trace_id: 链路追踪 ID。
        """
        if not self._memory_enabled or not self._auto_extract_enabled:
            return
        try:
            await self._extractor.extract_and_save(
                store=self._store, user_message=user_message, ai_response=ai_response,
                user_id=user_id, conversation_id=conversation_id,
                prev_ai_response=prev_ai_response, trace_id=trace_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 记忆提取异常，已跳过: {exc}")

    async def compress_after_chat(
        self, graph, config: dict, user_id: str, conversation_id: str, trace_id: Optional[str] = None,
    ) -> None:
        """对话结束后，检查并按需压缩会话历史；压缩产生的摘要追加写入长期记忆。

        Args:
            graph: LangGraph 编译后的图实例。
            config: 传给压缩器的 RunnableConfig。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            trace_id: 链路追踪 ID。
        """
        try:
            result = await self._compressor.compress_if_needed(graph, config)
        except Exception as exc:
            logger.warning(f"[MemoryManager] 压缩检查异常，已跳过: {exc}")
            return

        if not result:
            return

        summary = result["summary"]
        compressed_count = result["compressed_count"]

        try:
            await self._store.save(
                content=summary, user_id=user_id, conversation_id=conversation_id,
                memory_type="summary", importance=self._importance_threshold,
                source="compressor", trace_id=trace_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 压缩摘要写入长期记忆失败，已跳过: {exc}")
            return

        await self._record_audit(
            action=MemoryAuditAction.COMPRESS, user_id=user_id, source=MemorySource.COMPRESSOR,
            conversation_id=conversation_id, trace_id=trace_id,
            detail=f"compressed_count={compressed_count}；原始消息可在会话历史表按 conversation_id 查询",
        )

    async def compress_messages_after_chat(
        self,
        messages: list,
        user_id: str,
        conversation_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[dict]:
        """`compress_after_chat` 的 graph-free 版本，供中间件流水线使用。

        对应设计文档 4.2 节 `SummarizationMiddleware`：中间件的 `aafter_agent`
        钩子已经直接拿到本轮最终的 `messages`，不需要（也拿不到）一个真实的
        LangGraph 编译图去做 `aget_state`/`aupdate_state`。

        Args:
            messages: 当前会话的完整消息列表。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            trace_id: 链路追踪 ID。

        Returns:
            触发了压缩时返回压缩产出的 `state_update`（供调用方作为中间件
            钩子的返回值，交给框架自动合并进 checkpoint）；未触发压缩、
            摘要生成失败或长期记忆写入失败时返回 None。
        """
        try:
            outcome = await self._compressor.compress_messages(messages)
        except Exception as exc:
            logger.warning(f"[MemoryManager] 压缩检查异常，已跳过: {exc}")
            return None

        if outcome is None:
            return None

        try:
            await self._store.save(
                content=outcome.summary, user_id=user_id, conversation_id=conversation_id,
                memory_type="summary", importance=self._importance_threshold,
                source="compressor", trace_id=trace_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 压缩摘要写入长期记忆失败，已跳过: {exc}")
            return None

        await self._record_audit(
            action=MemoryAuditAction.COMPRESS, user_id=user_id, source=MemorySource.COMPRESSOR,
            conversation_id=conversation_id, trace_id=trace_id,
            detail=f"compressed_count={outcome.compressed_count}；原始消息可在会话历史表按 conversation_id 查询",
        )

        return outcome.state_update

    def is_sensitive_content(self, content: Optional[str]) -> bool:
        """判断内容是否命中敏感信息规则（供管理 API 在写入前做前置校验）。

        Args:
            content: 待检测内容。

        Returns:
            是否命中敏感信息规则；功能关闭或内容为空时返回 False。
        """
        return bool(content and self._sensitive_filter_enabled and contains_sensitive_info(content))

    async def _record_audit(
        self,
        action: MemoryAuditAction,
        user_id: str,
        source: MemorySource,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        audit_store = get_memory_audit_store()
        if audit_store is None:
            return
        await audit_store.record(
            action=action, user_id=user_id, source=source,
            conversation_id=conversation_id, trace_id=trace_id, detail=detail,
        )


async def get_memory_with_retry(
    store: BaseMemoryStore,
    memory_id: str,
    user_id: str,
    attempts: int = 3,
    delay_seconds: float = 0.5,
) -> Optional[dict]:
    """带重试的最终一致性读取：ES 等存储写入后短时间内可能读不到，按固定间隔重试。

    Args:
        store: 目标记忆存储。
        memory_id: 记忆 ID。
        user_id: 归属用户 ID。
        attempts: 最大重试次数。
        delay_seconds: 每次重试之间的等待秒数。

    Returns:
        读取到的记忆详情；多次重试后仍未读到则返回 None。

    Raises:
        Exception: 最后一次尝试仍然异常时，把该异常重新抛出。
    """
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            row = await store.get(memory_id, user_id)
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[MemoryManager] get memory attempt failed id={memory_id} "
                f"attempt={attempt + 1}/{attempts}: {exc}"
            )
            row = None
        if row is not None:
            return row
        if attempt < attempts - 1:
            await asyncio.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    return None


_manager: MemoryManager | None = None


def init_memory_manager(manager: MemoryManager) -> None:
    """应用启动时调用一次，注册全局单例。

    Args:
        manager: 已完成装配的 MemoryManager 实例。
    """
    global _manager
    _manager = manager


def get_memory_manager() -> MemoryManager:
    """返回全局唯一的 MemoryManager 实例。

    Raises:
        RuntimeError: init_memory_manager() 尚未被调用。
    """
    if _manager is None:
        raise RuntimeError("MemoryManager 尚未初始化，请确认应用已完成启动")
    return _manager
