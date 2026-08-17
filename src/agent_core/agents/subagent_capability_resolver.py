"""`resolve_subagent_capabilities()`：`task` 工具真正派发子 Agent 前，把
`SubagentProfile` 声明的能力依赖解析成实际可用的 Tool/Skill 中间件。

对应问题：`SubagentProfile` 曾经用 `tools_factory`/`middleware_factory` 直接
返回构造好的对象，没有任何环节校验"这个 Profile 声明的能力现在是否真的可用"
——一个依赖被下线的工具/技能，子 Agent 会被现造出来但拿不到真实能力，模型可能
凭自己已有知识伪装成"确实执行过"的结果回答（`web-researcher` 这类必须真的
发起外部动作才可信的子 Agent 尤其危险）。本模块在派发前统一解析，必需能力
缺失时 fail fast（不创建子 Agent），可选能力缺失时降级执行。

**Tool 解析**：先查 `RegistrySnapshot.by_model_name`（Lead Agent 自己也在用
的同一份 application 级快照），查不到再查调用方传入的 `subagent_only_tools`
兜底表——`web_search` 这类"故意不登记进 ToolRegistry、只给 subagent 用"的
工具的来源（见 `tools/registry/providers.py` 的说明：登记进去会让它意外出现
在 Lead Agent 自己的工具列表里，不是想要的副作用）。找到后统一走
`GuardrailProvider.check()`——这是子 Agent 工具第一次真正拿到按用户区分的
权限控制：`sub_agent_factory.py` 的子 Agent 默认不挂任何中间件（只有
`SkillMiddleware` 是例外），子 Agent 自己的工具调用完全不经过
`GuardrailMiddleware`，这里补上的是唯一一层。

**Skill 解析**：只做存在性校验（`SkillRegistry.get()`），不重复做 Guardrail——
`SkillMiddleware` 真正激活技能时已经会做（构造时传入的 `guardrail_provider`），
这里再做一遍是重复劳动。`allowed_skill_categories` 不预解析成具体技能名，
原样传给 `SkillMiddleware` 构造参数，由它在每次模型调用时动态决定命中哪些——
这是 `SkillMiddleware` 已有的、故意"不缓存、每次现算"的设计（保证 Skill
热重载后下一次调用立刻可见），本模块不应该在这之前再插一层缓存把它冻结住。

**在途任务隔离**：只对 Tool 成立，不对 Skill 成立，这是两者已有设计的自然
结果，不是本模块新引入的不一致——`resolved.tools` 是解析时刻的具体 BaseTool
对象列表，构造完就不再回头查 `ToolRegistry`（`ToolRegistry.publish()` 是
copy-on-write，旧对象继续可用，见其模块文档）；但 `SkillMiddleware` 构造时
接收的是**存活的 `SkillManager` 单例**（不是某一时刻的 `SkillRegistry` 快照），
会在子 Agent 运行期间的每次模型调用里现读 `skill_manager.registry`——这正是
它一直以来的热重载响应设计（见 `skill_middleware.py` 的说明"不缓存，保证
Skill 热重载后下一次模型调用就能看到最新技能"），本模块沿用，不做改动。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from loguru import logger

from src.agent_core.agents.subagent_profiles import SubagentProfile
from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_manager import SkillManager
from src.common.exceptions import SkillNotFoundError

if TYPE_CHECKING:
    # 不能在模块顶层 import：`tools/registry/__init__.py` 会 import
    # `providers.py`，而 `providers.py` 需要 `delegation_tools.build_task_tool`
    # 登记 `subagent` 来源，`delegation_tools.py` 又 import 本模块——顶层
    # 互相 import 会在包初始化到一半时炸出 ImportError，见
    # `delegation_tools.py` 里同样问题的说明。这里只用来做类型标注，`from
    # __future__ import annotations` 已经让标注整体延迟求值，不需要运行时
    # 真的导入这个类。
    from src.agent_core.tools.registry.snapshot import RegistrySnapshot

_SUBAGENT_TOOL_PERMISSION_SCOPE = "subagent"


@dataclass(frozen=True)
class ResolvedSubagentCapabilities:
    """`resolve_subagent_capabilities()` 的返回结果，直接喂给 `run_subagent()`。

    Attributes:
        tools: 实际可用的 BaseTool 列表（必需 + 通过校验的可选能力）。
        middleware: 实际挂载的中间件列表（目前只可能是一条 `SkillMiddleware`，
            `allowed_skill_categories` 为空时是空列表）。
        tool_registry_revision: 本次解析所依据的 `RegistrySnapshot.revision`。
        skill_registry_revision: 本次解析所依据的 `SkillRegistry.revision`。
        missing_optional: 缺失或被拒绝的可选能力名（工具名/技能名混在一起，
            日志用途，不需要区分类型）。
    """

    tools: list[BaseTool]
    middleware: list[AgentMiddleware]
    tool_registry_revision: int
    skill_registry_revision: int
    missing_optional: tuple[str, ...]


@dataclass(frozen=True)
class SubagentRunMetadata:
    """一次 `task` 派发实际使用的能力版本，供结构化日志追溯。"""

    subagent_type: str
    tool_registry_revision: int
    skill_registry_revision: int
    degraded_capabilities: tuple[str, ...]


class SubagentCapabilityUnavailable(Exception):
    """必需能力（Tool 或 Skill）当前不可用，调用方应拒绝派发，不创建子 Agent。"""

    def __init__(self, subagent_type: str, missing: list[str]) -> None:
        self.subagent_type = subagent_type
        self.missing = missing
        super().__init__(f"{subagent_type} 缺少必需能力: {', '.join(missing)}")


async def _resolve_tool(
    name: str,
    *,
    snapshot: RegistrySnapshot,
    subagent_only_tools: dict[str, BaseTool],
    guardrail_provider: GuardrailProvider,
    user_id: str | None,
    context: GuardrailContext,
) -> BaseTool | None:
    """按名字解析单个工具能力：先查共享注册表，再查 subagent 专属兜底表；
    找到后过一遍 Guardrail，被拒绝时视同不可用（返回 None）。
    """
    definition = snapshot.by_model_name.get(name)
    if definition is not None:
        permissions_key = definition.permissions_key
        permissions_scope = definition.permissions_scope
        build_tool = definition.build_tool
    else:
        static_tool = subagent_only_tools.get(name)
        if static_tool is None:
            return None
        permissions_key = name
        permissions_scope = _SUBAGENT_TOOL_PERMISSION_SCOPE
        build_tool = lambda t=static_tool: t  # noqa: E731

    decision = await guardrail_provider.check(
        user_id=user_id,
        tool_name=permissions_key,
        tool_args={},
        context=GuardrailContext(
            conversation_id=context.conversation_id,
            thinking=context.thinking,
            extra={"scope": permissions_scope},
        ),
    )
    if not decision.is_allowed:
        logger.info(f"[SubagentCapabilityResolver] 工具 {name!r} 被 Guardrail 拒绝: {decision.reason}")
        return None
    return build_tool()


async def resolve_subagent_capabilities(
    *,
    profile: SubagentProfile,
    snapshot: RegistrySnapshot,
    skill_manager: SkillManager,
    subagent_only_tools: dict[str, BaseTool],
    guardrail_provider: GuardrailProvider,
    user_id: str | None,
    conversation_id: str | None = None,
    thinking: bool = False,
) -> ResolvedSubagentCapabilities:
    """派发前解析一个 `SubagentProfile` 声明的全部能力依赖。

    Args:
        profile: 待派发的 subagent profile（声明式，见 `subagent_profiles.py`）。
        snapshot: 当前 `ToolRegistry` 快照（application 级，跟 Lead Agent
            自己 `resolve_tools()` 用的是同一份）。
        skill_manager: 全局 `SkillManager` 单例——不是某一时刻的
            `SkillRegistry` 快照，因为需要传给 `SkillMiddleware` 保持它
            "每次模型调用现读最新 registry" 的既有热重载响应行为（见模块
            docstring "在途任务隔离"一节）。
        subagent_only_tools: 故意不登记进 `ToolRegistry` 的 subagent 专属
            工具兜底表（`subagent_profiles.py::SUBAGENT_ONLY_TOOLS`）。
        guardrail_provider: 权限校验器。
        user_id: 发起请求的用户 ID。
        conversation_id: 归属会话 ID，透传进 `GuardrailContext`。
        thinking: 是否处于深度思考模式，透传进 `GuardrailContext`。

    Returns:
        `ResolvedSubagentCapabilities`。

    Raises:
        SubagentCapabilityUnavailable: 任意一个 `required_tools`/
            `required_skills` 缺失或被 Guardrail 拒绝。
    """
    context = GuardrailContext(conversation_id=conversation_id, thinking=thinking)
    required_names = sorted(profile.required_tools)
    optional_names = sorted(profile.optional_tools)
    all_names = required_names + optional_names

    resolved_tools = await asyncio.gather(*(
        _resolve_tool(
            name, snapshot=snapshot, subagent_only_tools=subagent_only_tools,
            guardrail_provider=guardrail_provider, user_id=user_id, context=context,
        )
        for name in all_names
    ))
    resolved_by_name = dict(zip(all_names, resolved_tools))

    missing: list[str] = [name for name in required_names if resolved_by_name[name] is None]
    missing_optional: list[str] = [name for name in optional_names if resolved_by_name[name] is None]

    for skill_name in sorted(profile.required_skills):
        try:
            skill_manager.registry.get(skill_name)
        except SkillNotFoundError:
            missing.append(skill_name)

    if missing:
        raise SubagentCapabilityUnavailable(profile.name, missing)

    middleware: list[AgentMiddleware] = []
    if profile.allowed_skill_categories:
        from src.agent_core.agents.skill_middleware import SkillMiddleware

        middleware.append(
            SkillMiddleware(
                skill_manager,
                guardrail_provider,
                SkillActivationService(skill_manager.registry),
                allowed_categories=profile.allowed_skill_categories,
            )
        )

    return ResolvedSubagentCapabilities(
        tools=[resolved_by_name[name] for name in all_names if resolved_by_name[name] is not None],
        middleware=middleware,
        tool_registry_revision=snapshot.revision,
        skill_registry_revision=skill_manager.registry.revision,
        missing_optional=tuple(missing_optional),
    )
