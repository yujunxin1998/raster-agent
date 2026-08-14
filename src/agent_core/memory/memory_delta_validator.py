"""MemoryDelta 校验（设计文档 §7.4 第 3、4 步）。

Pydantic 解析阶段已经完成了"JSON 可解析"和"Schema/枚举/长度/置信度范围正确"
两步（构造 `MemoryDelta` 失败即在那一步被拒绝）。这个模块负责剩下两步：
`evidence_message_ids`/`target_fact_id` 是否确实属于本次可见范围、以及
敏感信息/状态迁移合法性/最大操作数/文本长度这些需要"运行时上下文"才能判断
的规则——都不是纯 Schema 能表达的约束，所以单独拆出来，不塞进 Pydantic
`model_validator`（那里没有"这次对话看到了哪些候选 Fact"这个上下文）。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta, ProfilePatch
from src.agent_core.memory.memory_sensitive_filter import contains_sensitive_info
from src.common.constants import FactOperationType, MemoryStatus

_MUTABLE_FACT_STATUSES = frozenset({MemoryStatus.ACTIVE.value, MemoryStatus.PENDING.value})


@dataclass(frozen=True)
class DeltaRejection:
    """Delta 校验失败的原因，供 Worker 决定重试还是转 dead-letter。"""

    reason: str


def validate_delta(
    delta: MemoryDelta,
    *,
    visible_message_ids: set[str],
    visible_facts: dict[str, dict],
    max_profile_patches: int,
    max_fact_operations: int,
    max_text_length: int,
    sensitive_filter_enabled: bool,
) -> DeltaRejection | None:
    """校验一个已通过 Pydantic 解析的 MemoryDelta 是否可以进入 Apply Engine。

    Args:
        delta: 已解析的 MemoryDelta。
        visible_message_ids: 本次 Worker 处理窗口内的对话消息 ID 集合，
            `evidence_message_ids` 必须是其子集。
        visible_facts: 本次传给 LLM 的候选 Fact（`fact_id -> {status, ...}`），
            `target_fact_id` 必须能在这里找到，且状态必须允许被变更。
        max_profile_patches: 单个 Delta 允许的最大 Profile Patch 条数。
        max_fact_operations: 单个 Delta 允许的最大 Fact Operation 条数。
        max_text_length: Profile/Fact 文本内容的最大长度。
        sensitive_filter_enabled: 是否启用敏感信息检测。

    Returns:
        校验失败时返回 `DeltaRejection`（reason 已脱敏，可以直接落审计日志/
        dead-letter）；通过校验返回 None。
    """
    if len(delta.profile_patches) > max_profile_patches:
        return DeltaRejection(f"profile_patches 超过上限 {max_profile_patches}")
    if len(delta.fact_operations) > max_fact_operations:
        return DeltaRejection(f"fact_operations 超过上限 {max_fact_operations}")

    for patch in delta.profile_patches:
        rejection = _validate_profile_patch(patch, visible_message_ids, max_text_length, sensitive_filter_enabled)
        if rejection:
            return rejection

    for operation in delta.fact_operations:
        rejection = _validate_fact_operation(
            operation, visible_message_ids, visible_facts, max_text_length, sensitive_filter_enabled,
        )
        if rejection:
            return rejection

    return None


def _validate_profile_patch(
    patch: ProfilePatch, visible_message_ids: set[str], max_text_length: int, sensitive_filter_enabled: bool,
) -> DeltaRejection | None:
    if not set(patch.evidence_message_ids).issubset(visible_message_ids):
        return DeltaRejection(f"profile_patch[{patch.field}] 的 evidence_message_ids 超出本次可见对话范围")
    if len(patch.value) > max_text_length:
        return DeltaRejection(f"profile_patch[{patch.field}] 内容超过长度上限 {max_text_length}")
    if sensitive_filter_enabled and contains_sensitive_info(patch.value):
        return DeltaRejection(f"profile_patch[{patch.field}] 命中敏感信息过滤规则")
    return None


def _validate_fact_operation(
    operation: FactOperation,
    visible_message_ids: set[str],
    visible_facts: dict[str, dict],
    max_text_length: int,
    sensitive_filter_enabled: bool,
) -> DeltaRejection | None:
    if not set(operation.evidence_message_ids).issubset(visible_message_ids):
        return DeltaRejection(f"fact_operation[{operation.op}] 的 evidence_message_ids 超出本次可见对话范围")

    if operation.content and len(operation.content) > max_text_length:
        return DeltaRejection(f"fact_operation[{operation.op}] 内容超过长度上限 {max_text_length}")

    if operation.content and sensitive_filter_enabled and contains_sensitive_info(operation.content):
        return DeltaRejection(f"fact_operation[{operation.op}] 命中敏感信息过滤规则")

    if operation.op == FactOperationType.ADD:
        return None  # add 不引用既有 Fact，无需归属/状态校验

    target = visible_facts.get(operation.target_fact_id)
    if target is None:
        return DeltaRejection(
            f"fact_operation[{operation.op}] 的 target_fact_id={operation.target_fact_id!r} "
            "不在本次可见候选范围内"
        )
    if target.get("status") not in _MUTABLE_FACT_STATUSES:
        return DeltaRejection(
            f"fact_operation[{operation.op}] 的目标 Fact 当前状态为 {target.get('status')!r}，不允许再变更"
        )
    return None
