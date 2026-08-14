"""resolve_tools()：把 RegistrySnapshot + 本次请求级定义，解析成模型实际
可用的工具列表（设计文档 4.3 节）。

权限裁剪在这里提前做一遍，是为了不把用户看不到的工具 schema 喂给模型、
也让"模型为什么看不到这个工具"这件事可追溯；但这不能替代
`GuardrailMiddleware`——那是运行时真正执行前的最终防线，两层职责不重叠，
都要保留（见模块文档最后一段）。
"""
from __future__ import annotations

import asyncio

from loguru import logger

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import (
    RejectedTool,
    ResolvedToolSet,
    ToolDefinition,
    source_priority,
)


def _merge(
    application_definitions: dict[str, ToolDefinition],
    request_definitions: list[ToolDefinition],
) -> dict[str, ToolDefinition]:
    """按 `source_type` 优先级合并 application 级快照与本次请求级定义。

    builtin > skill/subagent/mcp > frontend——同名时优先级低的一方被丢弃并
    记 warning，而不是像 `CustomToolConverter.merge_tools` 原来那样谁后
    合并谁赢（设计文档 4.3 节，修复 1.4 节提到的静默覆盖问题）。两个
    application 级来源之间的同名冲突不会走到这里——那种情况在
    `RegistrySnapshot.replace_source` 阶段就已经被拒绝发布。
    """
    merged = dict(application_definitions)
    for definition in request_definitions:
        existing = merged.get(definition.model_name)
        if existing is None:
            merged[definition.model_name] = definition
            continue
        if source_priority(definition.source_type) < source_priority(existing.source_type):
            merged[definition.model_name] = definition
        else:
            logger.warning(
                f"[ToolResolver] model_name={definition.model_name!r} 冲突: "
                f"来源 {definition.source_type}({definition.source_id}) 优先级不高于已存在的 "
                f"{existing.source_type}({existing.source_id})，本次丢弃前者"
            )
    return merged


async def resolve_tools(
    *,
    snapshot: RegistrySnapshot,
    request_definitions: list[ToolDefinition] | None = None,
    user_id: str | None,
    guardrail_provider: GuardrailProvider,
    conversation_id: str | None = None,
    thinking: bool = False,
) -> ResolvedToolSet:
    """解析出本次请求实际可用的工具集（设计文档 4.3 节）。

    Args:
        snapshot: 当前 `ToolRegistry` 快照（application 级）。
        request_definitions: 本次请求的前端工具（request 级），可为空。
        user_id: 发起请求的用户 ID，透传给 `GuardrailProvider`。
        guardrail_provider: 权限校验器。
        conversation_id: 归属会话 ID，透传进 `GuardrailContext`（当前内置
            `AllowlistGuardrailProvider` 不使用这个字段，预留给未来更细
            粒度的策略实现）。
        thinking: 是否处于深度思考模式，同样透传进 `GuardrailContext`。

    Returns:
        `ResolvedToolSet`：实际可用的 BaseTool 列表 + 被拒绝的定义列表 +
        本次解析所依据的 `registry_revision`。
    """
    merged = _merge(dict(snapshot.by_model_name), request_definitions or [])
    candidates = [definition for definition in merged.values() if definition.enabled]

    # 各条定义的权限校验相互独立，并发发出而不是逐条 await——
    # `AllowlistGuardrailProvider.check()` 底层是两次 Postgres 查询，
    # 工具数量上升后串行等待会明显拖慢每次请求的启动延迟。
    decisions = await asyncio.gather(
        *(
            guardrail_provider.check(
                user_id=user_id,
                tool_name=definition.permissions_key,
                tool_args={},
                context=GuardrailContext(
                    conversation_id=conversation_id,
                    thinking=thinking,
                    extra={"scope": definition.permissions_scope},
                ),
            )
            for definition in candidates
        )
    )

    allowed: list[ToolDefinition] = []
    rejected: list[RejectedTool] = []
    for definition, decision in zip(candidates, decisions):
        if decision.is_allowed:
            allowed.append(definition)
        else:
            rejected.append(
                RejectedTool(
                    canonical_name=definition.canonical_name,
                    model_name=definition.model_name,
                    reason=decision.reason,
                )
            )

    if rejected:
        logger.info(
            f"[ToolResolver] registry_revision={snapshot.revision} user_id={user_id} "
            f"裁剪 {len(rejected)} 个工具: {[(r.model_name, r.reason) for r in rejected]}"
        )

    return ResolvedToolSet(
        registry_revision=snapshot.revision,
        tools=[definition.build_tool() for definition in allowed],
        rejected=rejected,
    )
