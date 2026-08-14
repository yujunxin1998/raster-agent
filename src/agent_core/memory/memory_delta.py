"""MemoryDelta 契约（Memory v2 核心，设计文档 §5）。

LLM 不生成整份 Memory，也不能直接覆盖数据库——只输出"建议发生哪些变更"的
严格 JSON，经 Pydantic 解析 + `memory_delta_validator.validate_delta()` 校验后，
才交给 `MemoryApplyEngine` 落库。语义判断在 LLM，合法性/幂等/去重/事务/权限/
审计全部在代码里，这是 Memory v2 相对旧版本"两次独立 LLM 调用各自决定写什么"
的核心差异。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.common.constants import FactOperationType, ProfilePatchOp

_PROFILE_FIELDS = frozenset({
    "work_context", "personal_context", "top_of_mind",
    "recent_months", "earlier_context", "long_term_background",
})


class ProfilePatch(BaseModel):
    """对 L1/L2 画像/时间线某一个字段的一次变更意图（设计文档 §5.3）。"""

    field: str
    op: ProfilePatchOp
    value: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence_message_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "ProfilePatch":
        if self.field not in _PROFILE_FIELDS:
            raise ValueError(f"非法的 Profile 字段: {self.field!r}")
        if self.op != ProfilePatchOp.CLEAR and not self.value.strip():
            raise ValueError(f"op={self.op} 时 value 不能为空")
        if not self.evidence_message_ids:
            raise ValueError("evidence_message_ids 不能为空")
        return self


class FactOperation(BaseModel):
    """对 L3 Facts 的一次变更意图（设计文档 §5.4）。

    字段随 `op` 不同而有不同的必填组合，由 `_validate` 按文档表格强制校验，
    而不是把每种操作拆成独立的子模型——LLM 输出的是同一个 JSON 结构，用一个
    宽松字段集合 + 事后校验更贴近实际解析场景。
    """

    op: FactOperationType
    target_fact_id: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    importance: int = Field(ge=1, le=10, default=5)
    confidence: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    reason: Optional[str] = None
    evidence_message_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "FactOperation":
        if self.op == FactOperationType.ADD:
            if not (self.content and self.content.strip()):
                raise ValueError("add 操作必须包含 content")
            if not self.category:
                raise ValueError("add 操作必须包含 category")
            if self.confidence is None:
                raise ValueError("add 操作必须包含 confidence")
            if not self.evidence_message_ids:
                raise ValueError("add 操作必须包含 evidence_message_ids")
        elif self.op == FactOperationType.UPDATE:
            if not self.target_fact_id:
                raise ValueError("update 操作必须显式给出 target_fact_id")
        elif self.op == FactOperationType.SUPERSEDE:
            if not self.target_fact_id:
                raise ValueError("supersede 操作必须给出旧 Fact ID")
            if not (self.content and self.content.strip()):
                raise ValueError("supersede 操作必须给出新 Fact 内容")
            if not self.category:
                raise ValueError("supersede 操作必须给出 category")
            if not self.reason:
                raise ValueError("supersede 操作必须给出理由")
            if not self.evidence_message_ids:
                raise ValueError("supersede 操作必须给出证据")
        elif self.op == FactOperationType.ARCHIVE:
            if not self.target_fact_id:
                raise ValueError("archive 操作必须给出目标 Fact ID")
            if not self.evidence_message_ids:
                raise ValueError("archive 操作必须给出明确证据")
        elif self.op == FactOperationType.REINFORCE:
            if not self.target_fact_id:
                raise ValueError("reinforce 操作必须给出目标 Fact ID")
        return self


class MemoryDelta(BaseModel):
    """一次 MemoryUpdateWorker 处理产出的完整更新意图（设计文档 §5.2 示例结构）。"""

    schema_version: int = 1
    source_event_ids: list[str] = Field(default_factory=list)
    profile_patches: list[ProfilePatch] = Field(default_factory=list)
    fact_operations: list[FactOperation] = Field(default_factory=list)
