"""长期记忆后处理提取器（三层记忆架构的 L3 Facts）。

原样迁移自 `src/core/memory/extractor.py`，改造为 `MemoryExtractor` 类：
原模块级函数 `extract_and_save`/`_find_similar`/`_has_remember_keyword`
分别变为类的公开方法与私有方法，构造参数取代了原来直接读取全局
`get_settings()` 的做法，便于单测时注入不同配置。重复/冲突检测、"记住"类
关键词兜底逻辑均未改动；提取分类从 fact/preference/decision/instruction/
correction 改造为 preference/knowledge/context/behavior/goal 五分类（原
instruction/correction 命中即强制 importance>=8 的特例已删除，改由跟分类
无关的"记住"类关键词兜底承担"必须记住"语义）。

跟 `user_profile_updater.py`（L1/L2 画像/时间线，随对话持续演进的整段摘要）
是两套独立的 LLM 调用：这里做的是"从这一轮里挑出值得单独存一条的离散知识
点"，不做画像/时间线的合并重写。
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage
from loguru import logger

from src.agent_core.memory.base_memory_store import BaseMemoryStore
from src.agent_core.model import create_chat_model

_EXTRACT_PROMPT_TEMPLATE = """你是一个记忆提取助手。分析下方对话，从用户发言中提取值得长期保留的离散知识点。

【提取标准】满足以下任一条件才提取：
1. 用户明确的偏好或习惯（工具选择、代码风格、沟通方式等）→ preference，例："偏好使用 Vim 而非 VS Code"
2. 用户具备的专业知识/技能（擅长的语言、框架、领域）→ knowledge，例："精通 Rust 和 WebAssembly"
3. 用户的客观背景事实（姓名、职业、所在地、公司、年龄等）→ context，例："在字节跳动担任高级工程师"
4. 用户展现出的行为模式（做事习惯、工作流程）→ behavior，例："习惯先写测试再写实现"
5. 用户明确的目标或意图（计划、规则、要求长期遵守的约定）→ goal，例："计划在 Q2 发布 v2.0"

如果用户发言是确认性的（如"就按你刚才说的方案做"、"可以，按这个来"），请结合"上一轮 AI 回复"理解用户确认/选择的具体内容，再提取。

【不提取】闲聊、问候、一次性临时问题、通用知识问答、AI 回答的内容。

【输出格式】JSON 数组，无内容则返回 []：
[
  {{"content": "提取的记忆内容（简洁陈述句）", "memory_type": "preference|knowledge|context|behavior|goal", "importance": 1到10的整数}}
]

只输出 JSON，不要有任何其他文字。

【对话内容】
{prev_section}用户：{user_message}
AI：{ai_response}"""

_DEFAULT_REMEMBER_KEYWORDS = ("记住", "记下", "务必记住", "请牢记", "牢记", "以后都", "以后请", "请务必")

_PREV_AI_TRUNCATE_CHARS = 300
_AI_RESPONSE_TRUNCATE_CHARS = 500


class MemoryExtractor:
    """分析一轮对话内容，提取值得长期保留的记忆并写入存储。"""

    def __init__(
        self,
        model_name: str,
        provider: str,
        api_key: str,
        base_url: str,
        default_importance_threshold: int,
        duplicate_score_threshold: float,
        conflict_score_threshold: float,
        low_confidence_margin: int,
        remember_keywords: tuple[str, ...] = _DEFAULT_REMEMBER_KEYWORDS,
    ) -> None:
        """初始化提取器。

        Args:
            model_name: 提取用的模型名称，为空时由调用方回退到主模型。
            provider: 模型供应商标识。
            api_key: 模型服务 API Key。
            base_url: 模型服务 Base URL。
            default_importance_threshold: 默认重要度阈值，低于此值的提取结果不写入。
            duplicate_score_threshold: 判定为"重复记忆"的相似度阈值。
            conflict_score_threshold: 判定为"冲突记忆"的相似度阈值下限。
            low_confidence_margin: 重要度刚过滤阈值不远时判定为低置信度的容差。
            remember_keywords: "记住"类兜底关键词元组，命中时即使 LLM 未提取到
                内容也强制保存原文。
        """
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._default_importance_threshold = default_importance_threshold
        self._duplicate_score_threshold = duplicate_score_threshold
        self._conflict_score_threshold = conflict_score_threshold
        self._low_confidence_margin = low_confidence_margin
        self._remember_keywords = remember_keywords

    async def extract_and_save(
        self,
        store: BaseMemoryStore,
        user_message: str,
        ai_response: str,
        user_id: str,
        conversation_id: str,
        importance_threshold: Optional[int] = None,
        prev_ai_response: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """分析本轮对话，提取值得长期保留的记忆并写入存储。

        Args:
            store: 目标记忆存储。
            user_message: 用户本轮发言。
            ai_response: AI 本轮回复。
            user_id: 归属用户 ID。
            conversation_id: 归属会话 ID。
            importance_threshold: 本次提取使用的重要度阈值，为空则使用默认值。
            prev_ai_response: 上一轮 AI 回复，用于理解确认性发言的具体指代内容。
            trace_id: 链路追踪 ID。
        """
        threshold = importance_threshold or self._default_importance_threshold
        prompt = self._build_prompt(user_message, ai_response, prev_ai_response)

        memories = await self._call_llm_extract(prompt)

        if not memories and self._has_remember_keyword(user_message):
            logger.info(f"[MemoryExtractor] LLM 未提取到内容，但命中'记住'类关键词，规则兜底保存原文 user={user_id}")
            await store.save(
                content=user_message, user_id=user_id, conversation_id=conversation_id,
                memory_type="goal", importance=8, source="extractor",
                status="active", trace_id=trace_id,
            )
            return

        await self._save_extracted_memories(
            store, memories, user_id, conversation_id, threshold, trace_id,
        )

    def _build_prompt(self, user_message: str, ai_response: str, prev_ai_response: Optional[str]) -> str:
        prev_section = f"上一轮 AI：{prev_ai_response[:_PREV_AI_TRUNCATE_CHARS]}\n" if prev_ai_response else ""
        return _EXTRACT_PROMPT_TEMPLATE.format(
            prev_section=prev_section,
            user_message=user_message,
            ai_response=ai_response[:_AI_RESPONSE_TRUNCATE_CHARS],
        )

    async def _call_llm_extract(self, prompt: str) -> list:
        try:
            llm = create_chat_model(
                model=self._model_name, provider=self._provider,
                api_key=self._api_key, base_url=self._base_url,
            )
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"[MemoryExtractor] 记忆提取失败: {exc}")
            return []

    async def _save_extracted_memories(
        self,
        store: BaseMemoryStore,
        memories: list,
        user_id: str,
        conversation_id: str,
        threshold: int,
        trace_id: Optional[str],
    ) -> None:
        saved_count = 0
        seen_in_batch: set[str] = set()

        for memory in memories:
            if not isinstance(memory, dict):
                continue
            content = str(memory.get("content", "")).strip()
            memory_type = memory.get("memory_type", "context")
            importance = int(memory.get("importance", 5))

            if not content or importance < threshold:
                continue
            if content in seen_in_batch:
                continue
            seen_in_batch.add(content)

            similar = await self._find_similar(store, content, user_id, memory_type)
            conflict_old_id = self._detect_conflict(similar, content)
            if conflict_old_id is _DUPLICATE_SENTINEL:
                logger.info(f"[MemoryExtractor] 与已有记忆高度相似，跳过保存 user={user_id} content={content[:50]}")
                continue

            status = "pending" if importance < threshold + self._low_confidence_margin else "active"

            new_id = await store.save(
                content=content, user_id=user_id, conversation_id=conversation_id,
                memory_type=memory_type, importance=importance, source="extractor",
                status=status, trace_id=trace_id,
            )
            if not new_id:
                continue
            saved_count += 1

            if conflict_old_id:
                logger.info(f"[MemoryExtractor] 检测到冲突记忆，归档旧记忆 old={conflict_old_id} new={new_id}")
                await store.update(
                    conflict_old_id, user_id=user_id, status="archived",
                    superseded_by=new_id, source="extractor", trace_id=trace_id,
                )

        if saved_count:
            logger.info(f"[MemoryExtractor] 提取并保存 {saved_count} 条记忆 user={user_id}")

    def _detect_conflict(self, similar: Optional[dict], content: str) -> Optional[str]:
        """返回冲突旧记忆的 ID；返回 `_DUPLICATE_SENTINEL` 表示应视为重复直接跳过；返回 None 表示无冲突。"""
        if not similar:
            return None
        score = similar.get("score", 0)
        if similar.get("content") == content or score >= self._duplicate_score_threshold:
            return _DUPLICATE_SENTINEL
        if self._conflict_score_threshold <= score < self._duplicate_score_threshold:
            return similar["id"]
        return None

    async def _find_similar(
        self, store: BaseMemoryStore, content: str, user_id: str, memory_type: str
    ) -> Optional[dict]:
        """同类型内检索语义最相似的已有记忆，用于重复/冲突判断。

        检索失败时返回 None（视为无相似项），不影响本轮提取流程。
        """
        try:
            existing = await store.search(
                query=content, user_id=user_id, memory_type=memory_type, candidate_k=5, top_n=1
            )
        except Exception as exc:
            logger.warning(f"[MemoryExtractor] 相似记忆检索失败，跳过重复/冲突检测: {exc}")
            return None
        return existing[0] if existing else None

    def _has_remember_keyword(self, text: str) -> bool:
        return any(keyword in text for keyword in self._remember_keywords)


# 哨兵值：区分"应视为重复跳过"与"存在冲突旧记忆需要归档"两种语义，
# 避免用 -1/空字符串等容易被误判的值表达"特殊情况"。
_DUPLICATE_SENTINEL = "__duplicate__"
