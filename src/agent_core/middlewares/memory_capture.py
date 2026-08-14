"""记忆捕获中间件（Memory v2，设计文档 §7.1，替代旧的 `MemoryExtractionMiddleware`）。

跟旧版本最大的区别：这里只做"清洗 + 落库"两件事，不调用 LLM、不直接改
Profile/Facts——把"这一轮对话里有没有值得记住的东西"这个语义判断完全交给
`MemoryUpdateWorker` 里的一次结构化 LLM 调用去做。`aafter_agent` 只负责：

1. 按 watermark（上一次已捕获事件的 `to_message_id`）取本轮新增消息；
2. 只保留 User 消息与最终、无悬挂 Tool Call 的 AI 回复，剥离附件说明块；
3. 明显低价值（单轮问候/空内容）的窗口不入队，watermark 不推进——下一轮会
   带着这些消息一起重新判断，不会丢；
4. 在同一数据库事务内写 `memory_event` + `memory_update_job`
   （`MemoryEventStore.create_event_and_job`），不是 fire-and-forget 的
   `asyncio.create_task`。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.config.settings import get_settings
from src.storage.memory_event_store import MemoryEventStore
from src.storage.memory_update_job_store import MemoryUpdateJobStore

_ATTACHMENT_BLOCK_MARKER = "【本轮附件】"

_LOW_VALUE_MESSAGES = frozenset({
    "你好", "hi", "hello", "hey", "嗨", "在吗", "谢谢", "谢谢你", "thanks", "thank you",
    "ok", "okay", "好的", "嗯", "嗯嗯", "收到", "test", "测试",
})


def _strip_attachment_block(text: str) -> str:
    """剥离 `_build_message_with_attachments()` 追加的附件说明块（设计文档 §4.1 清洗规则）。"""
    marker_index = text.find(_ATTACHMENT_BLOCK_MARKER)
    return text[:marker_index].rstrip() if marker_index != -1 else text


def _looks_low_value(conversation: list[dict]) -> bool:
    """单轮问候/空内容的粗粒度过滤，避免每句"你好"都触发一次 LLM 调用。"""
    user_texts = [item["content"].strip() for item in conversation if item["role"] == "user"]
    if len(user_texts) != 1:
        return False
    normalized = user_texts[0].lower().strip("!?。！？~.， ")
    return len(normalized) <= 1 or normalized in _LOW_VALUE_MESSAGES


class MemoryCaptureMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """会话结束后，把清洗过的增量对话落库为 `memory_event` + `memory_update_job`。"""

    def __init__(self, event_store: MemoryEventStore, job_store: MemoryUpdateJobStore) -> None:
        """初始化中间件。

        Args:
            event_store: 用于读取 watermark、写入捕获事件。
            job_store: 事件写入同一事务内用于创建/合并更新任务。
        """
        super().__init__()
        self._event_store = event_store
        self._job_store = job_store

    async def aafter_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """取 watermark 之后的增量消息，清洗后落库；本中间件不修改 state。"""
        context = runtime.context
        if context is None or not context.user_id or not context.conversation_id:
            return None

        settings = get_settings()
        if not settings.MEMORY_ENABLED or not settings.MEMORY_AUTO_UPDATE_ENABLED:
            return None

        messages = state.get("messages") or []
        if not messages:
            return None

        try:
            watermark = await self._event_store.get_watermark(context.user_id, context.conversation_id)
        except Exception as exc:
            logger.warning(f"[MemoryCaptureMiddleware] 读取 watermark 失败，本轮跳过捕获: {exc}")
            return None

        window = self._incremental_window(messages, watermark)
        if not window:
            return None

        to_message_id = getattr(window[-1], "id", None)
        if not to_message_id:
            logger.warning("[MemoryCaptureMiddleware] 窗口末尾消息缺少 id，无法推进 watermark，本轮跳过捕获")
            return None

        conversation = self._extract_conversation(window)
        if not conversation or _looks_low_value(conversation):
            return None

        content_hash = hashlib.sha256(
            json.dumps(conversation, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        try:
            await self._event_store.create_event_and_job(
                self._job_store,
                user_id=context.user_id, conversation_id=context.conversation_id,
                from_message_id=watermark or "", to_message_id=to_message_id,
                conversation=conversation, content_hash=content_hash,
                agent_name=None, trace_id=None,
                debounce_seconds=settings.MEMORY_UPDATE_DEBOUNCE_SECONDS,
            )
        except Exception as exc:
            logger.warning(f"[MemoryCaptureMiddleware] 捕获事件写入失败，已跳过: {exc}")
        return None

    @staticmethod
    def _incremental_window(messages: list[BaseMessage], watermark: Optional[str]) -> list[BaseMessage]:
        """截取 watermark 之后的新增消息；watermark 为空或已不在历史中时取全量。"""
        if watermark is None:
            return messages
        for index, message in enumerate(messages):
            if getattr(message, "id", None) == watermark:
                return messages[index + 1:]
        return messages

    @staticmethod
    def _extract_conversation(window: list[BaseMessage]) -> list[dict]:
        """只保留 User 消息与最终（无悬挂 Tool Call）的 AI 回复。

        每条保留下来的消息都带上原始 `id`——`MemoryUpdateWorker` 把这份对话喂给
        LLM 时会展示这些 ID，`evidence_message_ids` 引用的就是它们；
        `memory_delta_validator.validate_delta()` 也靠这份 ID 集合校验"证据是否
        确实来自本次可见对话"。
        """
        conversation: list[dict] = []
        for message in window:
            message_id = getattr(message, "id", None)
            if not message_id:
                continue
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                text = _strip_attachment_block(message.content).strip()
                if text:
                    conversation.append({"role": "user", "content": text, "id": message_id})
            elif isinstance(message, AIMessage) and isinstance(message.content, str) and not message.tool_calls:
                text = message.content.strip()
                if text:
                    conversation.append({"role": "assistant", "content": text, "id": message_id})
        return conversation
