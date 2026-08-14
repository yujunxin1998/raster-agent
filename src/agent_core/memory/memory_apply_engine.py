"""把一个已校验通过的 MemoryDelta 落库（设计文档 §7.5）。

`MemoryUpdateJobStore.claim_next()` 用 `FOR UPDATE SKIP LOCKED` 保证同一个
`memory_update_job` 不会被两个 Worker 并发处理，所以这里不需要额外的
"事件/任务是否已应用"幂等检查。Profile Patch 与每条 Fact Operation 各自在
自己的 Store 方法内部开事务，不是整个 Delta 包在一个大事务里——设计文档
理想是"一个事务内完成"，但这里退一步：Profile Patch 的乐观锁revision +
Fact 侧确定性去重（`normalize_fact` 唯一键）/状态迁移校验，本身就让"部分
应用后重试"是安全的（重放已应用的操作要么是 no-op 要么转成 reinforce，不
会产生重复数据），换来不用把 Store 全部改造成"调用方传 conn"的复杂度。
真正需要强一致的是"同一条 Fact 的当前状态"，而这一点由每个 Fact Operation
自己的行锁保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch
from src.common.constants import FactOperationType, MemoryAuditAction, MemorySource
from src.storage.memory_audit_store import MemoryAuditStore
from src.storage.user_memory_fact_store import UserMemoryFactStore
from src.storage.user_profile_store import UserProfileStore


class ProfilePatchConflictError(Exception):
    """Profile 乐观锁连续两次冲突，交给 Worker 判断是否重试。"""


@dataclass
class ApplyOutcome:
    """本次 Apply 的结果摘要，供 Worker 记录日志/指标。"""

    profile_patched: bool = False
    fact_results: list[dict] = field(default_factory=list)


class MemoryApplyEngine:
    """把 MemoryDelta 转换成对 Profile/Fact Store 的实际调用。"""

    def __init__(
        self,
        profile_store: UserProfileStore,
        fact_store: UserMemoryFactStore,
        audit_store: Optional[MemoryAuditStore],
        *,
        active_confidence_threshold: float,
    ) -> None:
        """初始化 Apply Engine。

        Args:
            profile_store: L1/L2 画像存储。
            fact_store: L3 Facts 规范化主存。
            audit_store: 审计存储；为 None 时静默跳过审计（辅助能力容错，
                与项目里其它 Store 的降级哲学一致）。
            active_confidence_threshold: 设计文档 §6.4 的 confidence 分档——
                `add` 操作的 Fact 达到此置信度直接 `active`，否则 `pending`
                （低于丢弃阈值的 Fact 由 Worker 在生成 Delta 前就不应该提交，
                Apply Engine 只负责 active/pending 这一档判断）。
        """
        self._profile_store = profile_store
        self._fact_store = fact_store
        self._audit_store = audit_store
        self._active_confidence_threshold = active_confidence_threshold

    async def apply(
        self, delta: MemoryDelta, *, user_id: str, source_event_id: Optional[str],
    ) -> ApplyOutcome:
        """应用一个 MemoryDelta。

        Args:
            delta: 已通过 `validate_delta()` 校验的 Delta。
            user_id: 归属用户 ID。
            source_event_id: 触发本次更新的 memory_event ID，写入审计/field_meta。

        Returns:
            本次应用的结果摘要。

        Raises:
            ProfilePatchConflictError: Profile 乐观锁连续冲突，调用方应让
                Worker 对该 Job 走重试路径（下次会用全新快照重新生成 Delta）。
        """
        outcome = ApplyOutcome()

        if delta.profile_patches:
            await self._apply_profile_patches(delta.profile_patches, user_id, source_event_id)
            outcome.profile_patched = True

        for operation in delta.fact_operations:
            result = await self._apply_fact_operation(operation, user_id, source_event_id)
            outcome.fact_results.append(result)

        return outcome

    async def _apply_profile_patches(
        self, patches: list[ProfilePatch], user_id: str, source_event_id: Optional[str],
    ) -> None:
        patch_dicts = [
            {"field": patch.field, "op": patch.op.value, "value": patch.value, "confidence": patch.confidence}
            for patch in patches
        ]

        snapshot = await self._profile_store.get_with_revision(user_id)
        ok = await self._profile_store.apply_patches(
            user_id, patch_dicts, expected_revision=snapshot["revision"], source_event_id=source_event_id,
        )
        if not ok:
            # 乐观锁冲突：重读一次快照重放一次（设计文档 §7.5："重新读快照并再次判断"）。
            snapshot = await self._profile_store.get_with_revision(user_id)
            ok = await self._profile_store.apply_patches(
                user_id, patch_dicts, expected_revision=snapshot["revision"], source_event_id=source_event_id,
            )
        if not ok:
            raise ProfilePatchConflictError(f"user={user_id} Profile revision 连续冲突，放弃本次 Patch")

        await self._record_audit(
            MemoryAuditAction.PROFILE_PATCHED, user_id, source_event_id,
            detail=f"fields={[patch.field for patch in patches]}",
        )

    async def _apply_fact_operation(
        self, operation: FactOperation, user_id: str, source_event_id: Optional[str],
    ) -> dict:
        if operation.op == FactOperationType.ADD:
            status = "active" if operation.confidence >= self._active_confidence_threshold else "pending"
            result = await self._fact_store.add_or_reinforce(
                user_id=user_id, content=operation.content, category=operation.category,
                importance=operation.importance, confidence=operation.confidence, status=status,
                source_event_id=source_event_id, evidence_message_ids=operation.evidence_message_ids,
            )
            action = MemoryAuditAction.FACT_REINFORCED if result["action"] == "reinforced" else MemoryAuditAction.FACT_ADDED
            await self._record_audit(action, user_id, source_event_id, memory_id=result["fact_id"], detail=operation.content)
            return result

        if operation.op == FactOperationType.UPDATE:
            ok = await self._fact_store.update(
                operation.target_fact_id, user_id, content=operation.content,
                category=operation.category, importance=operation.importance, confidence=operation.confidence,
            )
            await self._record_audit(
                MemoryAuditAction.UPDATE, user_id, source_event_id, memory_id=operation.target_fact_id,
            )
            return {"fact_id": operation.target_fact_id, "action": "updated", "ok": ok}

        if operation.op == FactOperationType.SUPERSEDE:
            new_fact_id = await self._fact_store.supersede(
                target_fact_id=operation.target_fact_id, user_id=user_id, content=operation.content,
                category=operation.category, importance=operation.importance, confidence=operation.confidence,
                source_event_id=source_event_id, evidence_message_ids=operation.evidence_message_ids,
            )
            await self._record_audit(
                MemoryAuditAction.FACT_SUPERSEDED, user_id, source_event_id,
                memory_id=operation.target_fact_id, detail=f"reason={operation.reason} new_fact_id={new_fact_id}",
            )
            return {"fact_id": new_fact_id, "action": "superseded", "old_fact_id": operation.target_fact_id}

        if operation.op == FactOperationType.ARCHIVE:
            ok = await self._fact_store.archive(operation.target_fact_id, user_id)
            await self._record_audit(
                MemoryAuditAction.FACT_ARCHIVED, user_id, source_event_id, memory_id=operation.target_fact_id,
            )
            return {"fact_id": operation.target_fact_id, "action": "archived", "ok": ok}

        # REINFORCE
        ok = await self._fact_store.reinforce(operation.target_fact_id, user_id, confidence=operation.confidence)
        await self._record_audit(
            MemoryAuditAction.FACT_REINFORCED, user_id, source_event_id, memory_id=operation.target_fact_id,
        )
        return {"fact_id": operation.target_fact_id, "action": "reinforced", "ok": ok}

    async def _record_audit(
        self,
        action: MemoryAuditAction,
        user_id: str,
        source_event_id: Optional[str],
        *,
        memory_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        if self._audit_store is None:
            return
        try:
            await self._audit_store.record(
                action=action, user_id=user_id, source=MemorySource.EXTRACTOR,
                memory_id=memory_id, trace_id=source_event_id, detail=detail,
            )
        except Exception as exc:
            logger.warning(f"[MemoryApplyEngine] 审计写入失败 action={action} user={user_id}: {exc}")
