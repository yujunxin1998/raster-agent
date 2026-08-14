"""记忆机制对外的统一门面（Memory v2 收敛版）。

跟旧版本相比大幅收窄：Facts/Profile 的读写主链路已经拆分到各自专门的组件——
背景抽取/更新是 `MemoryCaptureMiddleware` + `MemoryUpdateWorker` +
`MemoryApplyEngine`；注入是 `MemoryInjectionMiddleware` + `MemoryContextBuilder`；
人工/工具直接写入是 `UserMemoryFactStore` 单例。`MemoryManager` 现在只保留三件
跨这些组件、没有更自然归属的事情：会话压缩摘要写入长期记忆、画像只读查询（供
REST API）、敏感信息前置校验（供 REST API 在写入前给出更友好的错误提示）。
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.agent_core.memory.memory_compressor import MemoryCompressor
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.common.constants import MemoryAuditAction, MemorySource
from src.storage.memory_audit_store import get_memory_audit_store
from src.storage.user_memory_fact_store import UserMemoryFactStore
from src.storage.user_profile_store import UserProfileStore


class MemoryManager:
    """记忆机制的统一门面，供上层业务代码调用。"""

    def __init__(
        self,
        fact_store: UserMemoryFactStore,
        compressor: MemoryCompressor,
        profile_store: UserProfileStore,
        *,
        importance_threshold: int,
        sensitive_filter_enabled: bool,
    ) -> None:
        """初始化记忆管理器。

        Args:
            fact_store: L3 Facts 规范化主存，压缩摘要写入的落点。
            compressor: 会话压缩器。
            profile_store: 用户画像/时间线存储（L1/L2），供只读查询。
            importance_threshold: 压缩摘要写入长期记忆时使用的重要度。
            sensitive_filter_enabled: 是否启用敏感信息检测。
        """
        self._fact_store = fact_store
        self._compressor = compressor
        self._profile_store = profile_store
        self._importance_threshold = importance_threshold
        self._sensitive_filter_enabled = sensitive_filter_enabled

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """查询用户画像与时间线原始记录，供管理 API 直接返回。

        Args:
            user_id: 归属用户 ID。

        Returns:
            六个字段 + `updated_at` 的字典；用户还没生成过画像时返回 None。
        """
        return await self._profile_store.get(user_id)

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

        await self._save_summary(
            content=result["summary"], user_id=user_id, conversation_id=conversation_id,
            compressed_count=result["compressed_count"], trace_id=trace_id,
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

        saved = await self._save_summary(
            content=outcome.summary, user_id=user_id, conversation_id=conversation_id,
            compressed_count=outcome.compressed_count, trace_id=trace_id,
        )
        return outcome.state_update if saved else None

    async def _save_summary(
        self, *, content: str, user_id: str, conversation_id: str, compressed_count: int, trace_id: Optional[str],
    ) -> bool:
        """把压缩摘要写入 `user_memory_fact`（category="summary"，系统生成，直接 active）。"""
        try:
            result = await self._fact_store.add_or_reinforce(
                user_id=user_id, content=content, category="summary",
                importance=self._importance_threshold, confidence=1.0, status="active",
                source_conversation_id=conversation_id,
            )
        except Exception as exc:
            logger.warning(f"[MemoryManager] 压缩摘要写入长期记忆失败，已跳过: {exc}")
            return False

        await self._record_audit(
            action=MemoryAuditAction.COMPRESS, user_id=user_id, source=MemorySource.COMPRESSOR,
            memory_id=result["fact_id"], conversation_id=conversation_id, trace_id=trace_id,
            detail=f"compressed_count={compressed_count}；原始消息可在会话历史表按 conversation_id 查询",
        )
        return True

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
        memory_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        audit_store = get_memory_audit_store()
        if audit_store is None:
            return
        await audit_store.record(
            action=action, user_id=user_id, source=source, memory_id=memory_id,
            conversation_id=conversation_id, trace_id=trace_id, detail=detail,
        )


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
