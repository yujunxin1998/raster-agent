"""记忆更新 Worker（Memory v2 更新流水线的核心，设计文档 §7.2-§7.4）。

跟旧版本 `MemoryExtractor`/`UserProfileUpdater` 最大的区别：这里只有一次
LLM 调用，产出一个跨 Profile + Facts 的统一 `MemoryDelta`；语义判断全部在
Prompt 里交给模型，代码只负责拼 Prompt、解析、校验（`validate_delta`）、
按置信度分档、然后转交 `MemoryApplyEngine` 落库。

进程内 `run_forever()` 是一个持续轮询 `memory_update_job` 表的循环——可靠性
由数据库的 `FOR UPDATE SKIP LOCKED` + 租约 + 重试退避提供，不依赖 Worker
进程本身不崩溃（这正是旧版本 `asyncio.create_task()` 的问题）。
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import ValidationError

from src.agent_core.memory.memory_apply_engine import MemoryApplyEngine, ProfilePatchConflictError
from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta
from src.agent_core.memory.memory_delta_validator import validate_delta
from src.agent_core.model import create_chat_model
from src.common.constants import FactOperationType, MemoryAuditAction, MemorySource
from src.storage.memory_audit_store import MemoryAuditStore
from src.storage.memory_event_store import MemoryEventStore
from src.storage.memory_update_job_store import MemoryUpdateJobStore
from src.storage.user_memory_fact_store import UserMemoryFactStore
from src.storage.user_profile_store import UserProfileStore

_PROFILE_FIELD_LABELS = {
    "work_context": "职业背景", "personal_context": "个人背景", "top_of_mind": "当前关注",
    "recent_months": "近期活动（1-3个月）", "earlier_context": "更早模式（3-12个月）",
    "long_term_background": "长期背景",
}

_PROMPT_TEMPLATE = """你在维护一个用户的长期记忆（画像 + 离散事实）。只输出建议发生哪些
变更的 JSON，不要输出整份记忆——你的判断会先经过程序校验再落库，不合规则的
变更会被直接丢弃，所以宁可少输出，不要编造证据或引用不存在的消息/记录 ID。

【现有画像】（revision={revision}）
{profile_lines}

【现有相关事实候选】（只有这些 ID 可以被 update/supersede/archive/reinforce 引用，
不在列表里的 ID 一律非法）
{fact_lines}

【本轮新增对话】（每条消息前的 [ID] 是 evidence_message_ids 必须引用的真实 ID，
不要编造不存在的 ID）
{conversation_lines}

【任务】
1. profile_patches：如果有值得更新画像/时间线的信息，给出 field/op/value/confidence/
   evidence_message_ids。op=set 或 merge 时都要给出该字段更新后的完整文本（不是增量）；
   op=clear 仅限用户明确要求删除/纠正某个字段时使用。没有新信息就不要输出这个字段的 patch。
2. fact_operations：
   - add：全新的离散事实，需要 content/category/confidence/evidence_message_ids。
     category 取 preference/knowledge/context/behavior/goal 之一。
   - update：只是补充/修正细节，需要 target_fact_id。
   - supersede：用户明确纠正/推翻了某条已有事实，需要 target_fact_id、新 content、
     category、reason、evidence_message_ids。
   - archive：某条已有事实不再成立且没有替代内容，需要 target_fact_id、evidence_message_ids。
   - reinforce：用户再次确认了某条已有事实但没有新信息，只需要 target_fact_id。
3. 纯闲聊、问候、临时性问题不产生任何 patch/operation，两个数组都可以为空。
4. confidence 范围 0~1，只有你确信的信息才给高分。

只输出 JSON，不要有任何其他文字。profile_patches/fact_operations 数组里的每个元素
都是一个扁平对象，操作类型放在 op 字段里，不要把 op 的取值当成外层 key 再把其余
字段嵌套进去（错误示例：{{"add": {{"content": "..."}}}}；正确示例见下）：
{{"schema_version": 1, "source_event_ids": {source_event_ids},
  "profile_patches": [
    {{"field": "work_context", "op": "set", "value": "...", "confidence": 0.9, "evidence_message_ids": ["..."]}}
  ],
  "fact_operations": [
    {{"op": "add", "content": "...", "category": "preference", "confidence": 0.9, "evidence_message_ids": ["..."]}},
    {{"op": "update", "target_fact_id": "...", "content": "...", "evidence_message_ids": ["..."]}}
  ]}}
没有变更时两个数组给空列表 []，不要省略 op 字段，也不要输出示例之外的操作类型。"""


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class MemoryUpdateWorker:
    """持久化任务队列的消费者：claim -> 拼 Prompt -> 调用 LLM -> 校验 -> Apply。"""

    def __init__(
        self,
        event_store: MemoryEventStore,
        job_store: MemoryUpdateJobStore,
        profile_store: UserProfileStore,
        fact_store: UserMemoryFactStore,
        apply_engine: MemoryApplyEngine,
        audit_store: Optional[MemoryAuditStore],
        *,
        model_name: str,
        provider: str,
        api_key: str,
        base_url: str,
        debounce_seconds: int,
        lease_seconds: int,
        poll_interval_seconds: float,
        max_attempts: int,
        retry_backoff_seconds: list[int],
        candidate_facts_limit: int,
        max_conversation_chars: int,
        discard_confidence_threshold: float,
        max_profile_patches: int,
        max_fact_operations: int,
        max_text_length: int,
        sensitive_filter_enabled: bool,
    ) -> None:
        self._event_store = event_store
        self._job_store = job_store
        self._profile_store = profile_store
        self._fact_store = fact_store
        self._apply_engine = apply_engine
        self._audit_store = audit_store
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url
        self._debounce_seconds = debounce_seconds
        self._lease_seconds = lease_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._candidate_facts_limit = candidate_facts_limit
        self._max_conversation_chars = max_conversation_chars
        self._discard_confidence_threshold = discard_confidence_threshold
        self._max_profile_patches = max_profile_patches
        self._max_fact_operations = max_fact_operations
        self._max_text_length = max_text_length
        self._sensitive_filter_enabled = sensitive_filter_enabled

    async def run_forever(self) -> None:
        """持续轮询并处理任务，供 `asyncio.create_task()` 常驻运行。"""
        while True:
            try:
                job = await self._job_store.claim_next(
                    debounce_seconds=self._debounce_seconds, lease_seconds=self._lease_seconds,
                )
            except Exception as exc:
                logger.warning(f"[MemoryUpdateWorker] claim_next 失败: {exc}")
                job = None

            if job is None:
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            await self._process_job_safely(job)

    async def _process_job_safely(self, job: dict) -> None:
        try:
            await self._process(job)
        except Exception as exc:
            logger.warning(f"[MemoryUpdateWorker] 任务处理异常 job={job['job_id']}: {exc}")
            await self._job_store.mark_retry(
                job["job_id"], str(exc), max_attempts=self._max_attempts,
                backoff_seconds=self._retry_backoff_seconds,
            )

    async def _process(self, job: dict) -> None:
        conversation = await self._event_store.get_conversation_for_job(
            job["user_id"], job["conversation_id"], job["first_event_id"], job["last_event_id"],
        )
        if not conversation:
            await self._job_store.mark_succeeded(job["job_id"])
            return
        conversation = self._truncate_conversation(conversation)

        candidates = await self._fact_store.find_candidates_for_update(
            job["user_id"], agent_name=job.get("agent_name"), limit=self._candidate_facts_limit,
        )
        profile_snapshot = await self._profile_store.get_with_revision(job["user_id"])

        prompt = self._build_prompt(conversation, profile_snapshot, candidates, job)
        raw = await self._call_llm(prompt)
        if raw is None:
            await self._job_store.mark_retry(
                job["job_id"], "LLM 调用失败或返回空", max_attempts=self._max_attempts,
                backoff_seconds=self._retry_backoff_seconds,
            )
            return

        delta = self._parse_delta(raw, job)
        if delta is None:
            await self._job_store.mark_retry(
                job["job_id"], "Delta JSON/Schema 解析失败", max_attempts=self._max_attempts,
                backoff_seconds=self._retry_backoff_seconds,
            )
            return

        visible_message_ids = {message["id"] for message in conversation if "id" in message}
        visible_facts = {candidate["fact_id"]: candidate for candidate in candidates}
        rejection = validate_delta(
            delta, visible_message_ids=visible_message_ids, visible_facts=visible_facts,
            max_profile_patches=self._max_profile_patches, max_fact_operations=self._max_fact_operations,
            max_text_length=self._max_text_length, sensitive_filter_enabled=self._sensitive_filter_enabled,
        )
        if rejection is not None:
            logger.info(f"[MemoryUpdateWorker] Delta 被拒绝 job={job['job_id']} reason={rejection.reason}")
            await self._record_audit(MemoryAuditAction.DELTA_REJECTED, job, detail=rejection.reason)
            await self._job_store.mark_retry(
                job["job_id"], rejection.reason, max_attempts=self._max_attempts,
                backoff_seconds=self._retry_backoff_seconds,
            )
            return

        delta = self._drop_low_confidence_adds(delta)
        await self._record_audit(
            MemoryAuditAction.DELTA_GENERATED, job,
            detail=f"profile_patches={len(delta.profile_patches)} fact_operations={len(delta.fact_operations)}",
        )

        try:
            await self._apply_engine.apply(delta, user_id=job["user_id"], source_event_id=job["last_event_id"])
        except ProfilePatchConflictError as exc:
            await self._job_store.mark_retry(
                job["job_id"], str(exc), max_attempts=self._max_attempts,
                backoff_seconds=self._retry_backoff_seconds,
            )
            return

        await self._job_store.mark_succeeded(job["job_id"])

    def _truncate_conversation(self, conversation: list[dict]) -> list[dict]:
        """超过 `max_conversation_chars` 时只保留末尾部分（最近对话更重要）。"""
        total = 0
        kept: list[dict] = []
        for message in reversed(conversation):
            cost = len(message.get("content", ""))
            if kept and total + cost > self._max_conversation_chars:
                break
            kept.append(message)
            total += cost
        kept.reverse()
        return kept

    def _build_prompt(self, conversation: list[dict], profile_snapshot: dict, candidates: list[dict], job: dict) -> str:
        profile_lines = "\n".join(
            f"- {_PROFILE_FIELD_LABELS[field]}（{field}）：{profile_snapshot.get(field, '')}"
            for field in _PROFILE_FIELD_LABELS
        )
        fact_lines = "\n".join(
            f"- [{candidate['fact_id']}] ({candidate['category']}/{candidate['status']}) {candidate['content']}"
            for candidate in candidates
        ) or "（无）"
        conversation_lines = "\n".join(
            f"[{message.get('id', '')}] {'用户' if message['role'] == 'user' else 'AI'}：{message['content']}"
            for message in conversation
        )
        source_event_ids = json.dumps(
            sorted({str(job["first_event_id"]), str(job["last_event_id"])}), ensure_ascii=False,
        )
        return _PROMPT_TEMPLATE.format(
            revision=profile_snapshot.get("revision", 0), profile_lines=profile_lines,
            fact_lines=fact_lines, conversation_lines=conversation_lines,
            source_event_ids=source_event_ids,
        )

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            llm = create_chat_model(
                model=self._model_name, provider=self._provider,
                api_key=self._api_key, base_url=self._base_url,
            )
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            return result.content
        except Exception as exc:
            logger.warning(f"[MemoryUpdateWorker] LLM 调用失败: {exc}")
            return None

    def _parse_delta(self, raw: str, job: dict) -> Optional[MemoryDelta]:
        try:
            payload = json.loads(_strip_json_fence(raw))
            return MemoryDelta.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(f"[MemoryUpdateWorker] Delta 解析失败 job={job['job_id']}: {exc}")
            return None

    def _drop_low_confidence_adds(self, delta: MemoryDelta) -> MemoryDelta:
        """设计文档 §6.4：confidence 低于丢弃阈值的 `add` 直接不落库。"""
        kept: list[FactOperation] = [
            operation for operation in delta.fact_operations
            if not (operation.op == FactOperationType.ADD and operation.confidence < self._discard_confidence_threshold)
        ]
        if len(kept) == len(delta.fact_operations):
            return delta
        return delta.model_copy(update={"fact_operations": kept})

    async def _record_audit(self, action: MemoryAuditAction, job: dict, *, detail: str) -> None:
        if self._audit_store is None:
            return
        try:
            await self._audit_store.record(
                action=action, user_id=job["user_id"], source=MemorySource.EXTRACTOR,
                conversation_id=job["conversation_id"], trace_id=job.get("trace_id"), detail=detail,
            )
        except Exception as exc:
            logger.warning(f"[MemoryUpdateWorker] 审计写入失败 action={action}: {exc}")
