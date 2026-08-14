"""通用 `load_skill` 工具：把一个 WORKFLOW 技能名解析成完整指令文本。

对应 `docs/Skill注入与Load-Skill重构设计.md` 第四节。这里的实现是"自包含"
的——内部自己做 Guardrail 校验 + 调用 `SkillActivationService`，在没有任何
中间件保护的场景下（`sub_agent_factory.py::run_subagent()` 现造的子 Agent
不挂任何中间件）也能独立、安全地工作。

Lead Agent 场景下，`agent_core.agents.skill_middleware.SkillMiddleware` 会在
`awrap_tool_call` 里拦截对 `load_skill` 的调用并在到达这里之前就短路返回
（校验 + 加载逻辑在中间件里做了一遍一模一样的事），所以本文件的
`_invoke` 实际上是"给没有 SkillMiddleware 保护的调用方（subagent）用的
兜底实现"，不是重复造轮子——两处复用同一个 `SkillActivationService`/
`GuardrailProvider`，逻辑只有一份，只是"谁来触发"这一层不同。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.common.exceptions import SkillDefinitionInvalidError, SkillNotFoundError

LOAD_SKILL_TOOL_NAME = "load_skill"
_TOOL_DESCRIPTION = "按名称加载一个工作流技能的完整操作指令。可用技能见系统提示词里的技能目录。"


class LoadSkillInput(BaseModel):
    skill_name: str = Field(description="要加载的技能名称，从系统提示词的技能目录里选择")


def create_load_skill_tool(
    activation_service: SkillActivationService,
    guardrail_provider: GuardrailProvider,
    allowed_categories: frozenset[str],
) -> StructuredTool:
    """构造一个绑定了具体技能范围的 `load_skill` 工具实例。

    Args:
        activation_service: 技能激活服务。
        guardrail_provider: 权限校验器，按被请求的具体 `skill_name`（不是
            固定的 `"load_skill"`）做校验，保留"可以单独禁用某一个技能"的
            既有粒度。
        allowed_categories: 调用方允许加载的技能分类集合（Lead Agent 传
            `_LEAD_AGENT_SKILL_CATEGORIES`，`web-researcher` 传
            `frozenset({"web_search"})`）。

    Returns:
        绑定好范围的 `StructuredTool` 实例。
    """

    async def _invoke(skill_name: str, config: RunnableConfig) -> str:
        configurable = config.get("configurable", {}) if config else {}
        user_id = configurable.get("user_id")
        conversation_id = configurable.get("thread_id")

        decision = await guardrail_provider.check(
            user_id=user_id,
            tool_name=skill_name,
            tool_args={},
            context=GuardrailContext(conversation_id=conversation_id),
        )
        if not decision.is_allowed:
            return decision.reason

        try:
            return await activation_service.activate(skill_name, allowed_categories)
        except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
            return str(exc)

    return StructuredTool(
        name=LOAD_SKILL_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema=LoadSkillInput,
        coroutine=_invoke,
    )
