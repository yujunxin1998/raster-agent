"""Lead Agent 与 subagent 之间共享的轻量通讯协议。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from src.agent_core.middlewares.context import AgentRuntimeContext

SubagentStatus = Literal["succeeded", "failed", "timed_out", "cancelled", "degraded"]


@dataclass(frozen=True)
class SubagentResult:
    """一次 subagent 执行的结构化终态。"""

    task_id: str
    agent_name: str
    status: SubagentStatus
    output: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    artifacts: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        if self.output:
            return self.output
        detail = self.error_message or self.error_code or self.status
        return f"[{self.agent_name}] 执行失败: {detail}"


def build_delegation_task_id(
    *, conversation_id: str, subagent_type: str, task: str, parent_task_id: str | None = None,
) -> str:
    """根据业务身份生成稳定任务 ID，用于幂等合并相同委派。"""

    payload = json.dumps(
        {
            "conversation_id": conversation_id,
            "parent_task_id": parent_task_id,
            "subagent_type": subagent_type,
            "task": task,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "subtask_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def derive_subagent_context(
    parent: AgentRuntimeContext, *, task_id: str, agent_name: str,
) -> AgentRuntimeContext:
    """从父运行上下文派生隔离的子任务上下文，不复用可变缓存。"""

    return replace(
        parent,
        task_id=task_id,
        parent_task_id=parent.task_id,
        root_task_id=parent.root_task_id or task_id,
        agent_name=agent_name,
        delegation_depth=parent.delegation_depth + 1,
        memory_cache={},
    )
