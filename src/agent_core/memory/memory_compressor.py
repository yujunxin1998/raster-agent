"""会话历史压缩。

原样迁移自 `src/core/memory/compressor.py`，改造为 `MemoryCompressor` 类。
压缩触发条件、结构化摘要渲染、失败降级文案均未改动。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import RemoveMessage
from loguru import logger

from src.agent_core.model import create_chat_model

_SUMMARIZE_PROMPT_TEMPLATE = """请将以下对话历史压缩为结构化摘要，输出 JSON，字段说明：
- goal: 用户本轮目标（字符串，没有则为空字符串）
- key_facts: 关键事实（字符串数组）
- decisions: 技术/业务决策（字符串数组）
- todos: 待办事项（字符串数组）
- open_questions: 未解决问题（字符串数组）
- tool_results: 重要工具调用结果摘要（字符串数组）

要求：忽略闲聊和重复内容，用中文输出，只输出 JSON，不要任何其他文字。

{messages}"""

_SUMMARY_FAILURE_PREFIX = "[摘要生成失败，"
_MESSAGE_TRUNCATE_CHARS = 300

_SECTION_LABELS = (
    ("key_facts", "关键事实"),
    ("decisions", "技术决策"),
    ("todos", "待办事项"),
    ("open_questions", "未解决问题"),
    ("tool_results", "工具结果"),
)


@dataclass(frozen=True)
class CompressionOutcome:
    """一次压缩的结果（graph-free，供中间件流水线直接消费）。

    Attributes:
        summary: 生成的结构化摘要文本。
        compressed_count: 被压缩掉的原始消息条数。
        state_update: 可直接作为 LangGraph 中间件钩子返回值的 state 更新
            （`{"messages": [RemoveMessage(...), ..., AIMessage(...)]}`），
            框架会把它自动合并进 checkpoint，不需要调用方手动
            `graph.aupdate_state`。
    """

    summary: str
    compressed_count: int
    state_update: dict


class MemoryCompressor:
    """检查会话消息数量，超阈值时压缩旧消息为结构化摘要。"""

    def __init__(
        self,
        model_name: str,
        provider: str,
        api_key: str,
        base_url: str,
        default_threshold: int,
        default_keep_recent: int,
    ) -> None:
        """初始化压缩器。

        Args:
            model_name: 摘要生成用的模型名称，为空时由调用方回退到主模型。
            provider: 模型供应商标识。
            api_key: 模型服务 API Key。
            base_url: 模型服务 Base URL。
            default_threshold: 默认压缩阈值，消息总数超过此值触发压缩。
            default_keep_recent: 默认保留最近 N 条不压缩。
        """
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._default_threshold = default_threshold
        self._default_keep_recent = default_keep_recent

    async def compress_messages(
        self,
        messages: list,
        threshold: Optional[int] = None,
        keep_recent: Optional[int] = None,
    ) -> Optional[CompressionOutcome]:
        """检查消息数量，超过阈值时把旧消息压缩为结构化摘要。

        graph-free 版本：直接接受消息列表，不依赖 LangGraph 编译图的
        `aget_state`/`aupdate_state`（供中间件流水线的 `aafter_agent` 钩子
        使用——钩子拿到的 state 里已经有现成的 `messages`，返回值会被框架
        自动合并进 checkpoint）。

        Args:
            messages: 当前会话的完整消息列表。
            threshold: 本次检查使用的阈值，为空则使用默认值。
            keep_recent: 本次压缩保留的最近消息条数，为空则使用默认值。

        Returns:
            触发压缩时返回 `CompressionOutcome`；未超过阈值或摘要生成彻底
            失败时返回 None。
        """
        resolved_threshold = threshold or self._default_threshold
        resolved_keep_recent = keep_recent or self._default_keep_recent

        if len(messages) <= resolved_threshold:
            return None

        old_messages = messages[:-resolved_keep_recent]
        logger.info(f"[MemoryCompressor] 触发压缩，共 {len(messages)} 条，压缩 {len(old_messages)} 条")

        summary_text = await self._summarize(old_messages)
        if summary_text.startswith(_SUMMARY_FAILURE_PREFIX):
            return None

        removes = [RemoveMessage(id=message.id) for message in old_messages]
        # 摘要以 AIMessage（而非 SystemMessage）形式插入历史中间：Qwen/vLLM 等
        # 模型的 chat template 强制要求 system 消息必须位于消息列表最前面，
        # 而这条摘要在压缩后会长期停留在"保留的最近消息"之前——用 SystemMessage
        # 会导致后续每一轮请求都携带一条非开头位置的 system 消息，触发
        # `System message must be at the beginning` 400 错误。
        summary_message = AIMessage(content=f"[历史摘要]\n{summary_text}")
        logger.info(f"[MemoryCompressor] 压缩完成，保留最近 {resolved_keep_recent} 条 + 摘要")

        return CompressionOutcome(
            summary=summary_text,
            compressed_count=len(old_messages),
            state_update={"messages": removes + [summary_message]},
        )

    async def compress_if_needed(
        self,
        graph,
        config: dict,
        threshold: Optional[int] = None,
        keep_recent: Optional[int] = None,
    ) -> Optional[dict]:
        """检查会话消息数，超过阈值时压缩旧消息（LangGraph 编译图版本）。

        保留该方法是为了兼容"调用方持有一个真实编译图"的场景（如未来某个
        入口仍然直接操作 LangGraph checkpointer）；核心压缩逻辑已经收敛到
        graph-free 的 `compress_messages`，本方法只是加了一层
        `aget_state`/`aupdate_state` 的读写。

        Args:
            graph: LangGraph 编译后的图实例，用于读写会话状态。
            config: 传给 graph.aget_state/aupdate_state 的 RunnableConfig。
            threshold: 本次检查使用的阈值，为空则使用默认值。
            keep_recent: 本次压缩保留的最近消息条数，为空则使用默认值。

        Returns:
            执行了压缩时返回 `{"summary": str, "compressed_count": int}`；
            未触发压缩或摘要生成彻底失败时返回 None。
        """
        try:
            state = await graph.aget_state(config)
        except Exception as exc:
            logger.warning(f"[MemoryCompressor] 获取会话状态失败，跳过压缩: {exc}")
            return None

        outcome = await self.compress_messages(state.values.get("messages", []), threshold, keep_recent)
        if outcome is None:
            return None

        await graph.aupdate_state(config, outcome.state_update)
        return {"summary": outcome.summary, "compressed_count": outcome.compressed_count}

    def _render_structured_summary(self, data: dict) -> str:
        """把结构化摘要 JSON 渲染为可检索、可读的 Markdown 文本。"""
        sections: list[str] = []
        goal = str(data.get("goal") or "").strip()
        if goal:
            sections.append(f"目标：{goal}")
        for key, label in _SECTION_LABELS:
            items = [str(item).strip() for item in (data.get(key) or []) if str(item).strip()]
            if items:
                lines = "\n".join(f"- {item}" for item in items)
                sections.append(f"{label}：\n{lines}")
        return "\n".join(sections)

    async def _summarize(self, messages: list) -> str:
        text_parts = []
        for message in messages:
            role = "用户" if message.__class__.__name__ == "HumanMessage" else "AI"
            content = message.content if isinstance(message.content, str) else str(message.content)
            if content.strip():
                text_parts.append(f"{role}：{content[:_MESSAGE_TRUNCATE_CHARS]}")

        conversation_text = "\n".join(text_parts)
        prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(messages=conversation_text)

        try:
            llm = create_chat_model(
                model=self._model_name, provider=self._provider,
                api_key=self._api_key, base_url=self._base_url,
            )
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()
        except Exception as exc:
            logger.error(f"[MemoryCompressor] 摘要生成失败: {exc}")
            return f"{_SUMMARY_FAILURE_PREFIX}原始消息 {len(messages)} 条已归档]"

        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

        try:
            data = json.loads(cleaned)
            rendered = self._render_structured_summary(data) if isinstance(data, dict) else ""
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"[MemoryCompressor] 结构化摘要解析失败，降级为模型原始输出: {exc}")
            rendered = ""

        return rendered or raw
