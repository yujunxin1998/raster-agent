"""Lead Agent 专属：动态注入技能目录 + 拦截 `load_skill` 调用做加载与权限校验。

对应 `docs/Skill注入与Load-Skill重构设计.md` 第四、五节。不进
`agent_core/loop.py::build_middlewares()`——`web-researcher` 等 subagent
（`sub_agent_factory.py::run_subagent()` 现造的子 Agent）不挂任何中间件，本类
只在 `lead_agent.py` 里随 Lead Agent 专属中间件追加，与 `DatasourceRoutingMiddleware`
同一挂载方式、同一个不进通用流水线的原因。

把 Skill 的"发现"（目录注入）和"激活"（校验+加载正文）都收在这一个中间件里，
而不是让 `load_skill` 自己的 `StructuredTool.coroutine` 承担业务逻辑——`load_skill`
在 Lead Agent 场景下只是一个"让模型能够发起这次调用"的 schema 占位，真正的
校验和加载发生在 `awrap_tool_call` 里，短路返回，永远不会调用到底层
`handler(request)`（即不会执行 `skill_load_tool.py::create_load_skill_tool`
里注册的那份 `_invoke`）。这与 `GuardrailMiddleware` 拒绝时"不调用 handler,
直接返回 error ToolMessage"是完全一致的既有模式。

`skill_load_tool.py` 里的自包含实现仍然保留，是因为 `web-researcher` subagent
没有任何中间件可以拦截——那条路径上 `load_skill` 只能靠自己的 `_invoke` 独立
完成权限校验和加载，两边复用同一个 `SkillActivationService`/`GuardrailProvider`
实例，逻辑只有一份，只是"谁来触发"这一层因为 subagent 的中间件缺位而不同。

挂载顺序约束：必须排在 `build_middlewares()` 产出的
`ToolAuditMiddleware`/`ToolErrorHandlingMiddleware`/`LoopDetectionMiddleware`
之后（列表里越靠后越贴近实际执行——见 `loop.py` 的顺序说明），这样
`load_skill` 调用依然会被审计日志、异常兜底、死循环检测覆盖，本类只是
这条链路里最内层、真正做"要不要把技能正文喂给模型"这个决策的一层；同时
必须排在 `StreamingModelMiddleware` 之前（后者必须是模型调用链路的最内层，
见其模块文档），顺序为
`[*build_middlewares(...), DatasourceRoutingMiddleware(), SkillMiddleware(...), StreamingModelMiddleware()]`。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_catalog import build_skill_catalog
from src.agent_core.skills.skill_definition import SkillDefinition, SkillKind
from src.agent_core.skills.skill_load_tool import LOAD_SKILL_TOOL_NAME
from src.agent_core.skills.skill_manager import SkillManager
from src.common.exceptions import SkillDefinitionInvalidError, SkillNotFoundError

_SKILL_CATALOG_TAG = "skill_catalog"


class SkillMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """动态注入 WORKFLOW 技能目录，并拦截 `load_skill` 调用完成校验与加载。"""

    def __init__(
        self,
        skill_manager: SkillManager,
        guardrail_provider: GuardrailProvider,
        activation_service: SkillActivationService,
        allowed_categories: frozenset[str],
    ) -> None:
        """初始化中间件。

        Args:
            skill_manager: 全局 `SkillManager` 单例，取其 `registry` 现算目录
                （不缓存，保证 Skill 热重载后下一次模型调用就能看到最新技能）。
            guardrail_provider: 权限校验器，按被请求的具体 `skill_name`（不是
                固定的 `"load_skill"`）做校验。
            activation_service: 技能激活服务。
            allowed_categories: 这个 Agent 允许加载的技能分类集合。
        """
        super().__init__()
        self._skill_manager = skill_manager
        self._guardrail_provider = guardrail_provider
        self._activation_service = activation_service
        self._allowed_categories = allowed_categories

    def _workflow_skills(self) -> list[SkillDefinition]:
        """现算当前分类范围内的 WORKFLOW 技能，不缓存。"""
        return [
            skill
            for category in self._allowed_categories
            for skill in self._skill_manager.registry.by_category(category)
            if skill.kind is SkillKind.WORKFLOW
        ]

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """把当前可用的 WORKFLOW 技能目录追加进本次模型调用的 system_prompt。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            模型调用结果。
        """
        catalog = build_skill_catalog(self._workflow_skills())
        if catalog:
            block = f"<{_SKILL_CATALOG_TAG}>\n{catalog}\n</{_SKILL_CATALOG_TAG}>"
            merged_prompt = f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
            request = request.override(system_prompt=merged_prompt)
        return await handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """拦截 `load_skill` 调用，短路完成权限校验 + 加载；其余工具调用原样透传。

        Args:
            request: 工具调用请求。
            handler: 执行工具调用的回调（非 `load_skill` 调用时原样转交）。

        Returns:
            `load_skill` 调用：携带加载结果或拒绝原因的 `ToolMessage`，不会
            调用 `handler`；其余调用：`handler(request)` 的原始结果。
        """
        tool_call = request.tool_call
        if tool_call["name"] != LOAD_SKILL_TOOL_NAME:
            return await handler(request)

        context = request.runtime.context
        skill_name = str((tool_call.get("args") or {}).get("skill_name") or "")

        decision = await self._guardrail_provider.check(
            user_id=context.user_id if context else None,
            tool_name=skill_name,
            tool_args={},
            context=GuardrailContext(conversation_id=context.conversation_id if context else None),
        )
        if not decision.is_allowed:
            return ToolMessage(content=decision.reason, tool_call_id=tool_call["id"], status="error")

        try:
            content = await self._activation_service.activate(skill_name, self._allowed_categories)
        except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
            content = str(exc)

        return ToolMessage(content=content, tool_call_id=tool_call["id"])
