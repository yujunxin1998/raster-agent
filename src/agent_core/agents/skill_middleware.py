"""Lead Agent 专属：Skill 发现、预路由、激活与资源读取的唯一运行入口。

对应 `docs/Skill注入与Load-Skill重构设计.md` 第四、五节的原始落地，以及
`docs/Skill与Tool完全解耦重构设计.md` 第 7 节的"自动路由为主，load_skill
为补充"升级。不进 `agent_core/loop.py::build_middlewares()`——`web-researcher`
等 subagent（`sub_agent_factory.py::run_subagent()` 现造的子 Agent）不挂任何
中间件，本类只在 `lead_agent.py` 里随 Lead Agent 专属中间件追加，与
`DatasourceRoutingMiddleware` 同一挂载方式、同一个不进通用流水线的原因。

**自动路由（重构文档 7.1 节）**：单纯把 Skill Catalog 放进 System Prompt、
指望模型自己判断"要不要调用 load_skill"是不可靠的——任务已经命中某个技能，
模型也可能觉得自己能直接搞定而跳过。`awrap_model_call` 现在先跑一遍
`skill_router.route()`，`forced`（用户/API 显式指定，或技能自己声明
`activation: required`）的技能在**首次模型调用前**就通过
`SkillActivationService` 把正文直接注入 system_prompt，不依赖模型主动调用
`load_skill`；只有 `catalog_candidates`（`activation: automatic` 且未被显式
指定）才继续走"只给目录、模型按需 load_skill 补充加载"的旧路径。

**匹配方式范围收窄**：`skill_router.route()` 只做规则/元数据匹配（显式指定
+ `activation` 声明），不引入 Embedding/LLM 语义匹配——原因见
`skill_router.py` 模块文档。

**重复激活去重**：`PipelineState.activated_skills`（`state.py`）记录本次
Agent 运行内已经激活过的技能，同名同版本不重复把正文注入
system_prompt/消息（重构文档 7.4 节）。`awrap_model_call` 通过
`ExtendedModelResponse(command=Command(update={"activated_skills": ...}))`
写入这份状态；`awrap_tool_call` 拦截到的补充激活（`load_skill`）通过
`Command(update={...})` 写入，两处使用同一个 `_merge_activated_skills`
reducer 合并，互不冲突（并发工具调用场景与 `recent_tool_calls` 同款处理）。

挂载顺序约束（完整列表 + 校验代码见 `middleware_order.py::ORDER_CONSTRAINTS`/
`validate_middleware_order()`，这里不重复展开）：必须排在
`ToolAuditMiddleware`/`ToolErrorHandlingMiddleware`/`LoopDetectionMiddleware`
之后，这样 `load_skill` 调用依然会被审计日志、异常兜底、死循环检测覆盖；
同时必须排在 `StreamingModelMiddleware` 之前（后者必须是模型调用链路的最内层）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentMiddleware, ExtendedModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from src.agent_core.guardrail.guardrail_provider import GuardrailContext, GuardrailProvider
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.state import PipelineState
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_catalog import build_skill_catalog
from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_load_tool import LOAD_SKILL_TOOL_NAME, create_load_skill_tool
from src.agent_core.skills.skill_manager import SkillManager
from src.agent_core.skills.skill_resource_tool import create_read_skill_resource_tool
from src.agent_core.skills.skill_router import SkillRoutingDecision, route
from src.common.exceptions import SkillDefinitionInvalidError, SkillNotFoundError

_SKILL_CATALOG_TAG = "skill_catalog"
_SKILL_TAG = "skill"
_ALREADY_ACTIVATED_TEMPLATE = "技能 [{skill_name}] 已在本次对话中激活，以下内容已在你的上下文里，无需重复加载。"


class SkillMiddleware(AgentMiddleware[PipelineState, AgentRuntimeContext]):
    """动态注入技能目录 + 自动预路由激活 + 拦截 `load_skill`/`read_skill_resource`。"""

    state_schema = PipelineState

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
        # 重构文档 7.4 节建议写法：`load_skill`/`read_skill_resource` 的工具
        # schema 都由中间件自己提供，不再依赖 ToolRegistry 单独注册一条
        # `skill.load_skill` 定义（`SkillToolProvider` 已删除）。`load_skill`
        # 复用 `create_load_skill_tool`——它本身自带的 `_invoke` 会被
        # `awrap_tool_call` 短路，只取它的 schema/工具名，执行逻辑走
        # 这里的拦截分支，两边用同一个 activation_service/guardrail_provider
        # 保证行为一致。
        self.tools = [
            create_load_skill_tool(activation_service, guardrail_provider, allowed_categories),
            create_read_skill_resource_tool(skill_manager.registry, allowed_categories),
        ]

    def _visible_skills(self) -> list[SkillDefinition]:
        """现算当前分类范围内可见的全部技能，不缓存。"""
        return [
            skill
            for category in self._allowed_categories
            for skill in self._skill_manager.registry.by_category(category)
        ]

    async def _check_permission(self, context: AgentRuntimeContext | None, skill_name: str) -> bool:
        decision = await self._guardrail_provider.check(
            user_id=context.user_id if context else None,
            tool_name=skill_name,
            tool_args={},
            context=GuardrailContext(conversation_id=context.conversation_id if context else None),
        )
        if not decision.is_allowed:
            logger.info(f"[SkillMiddleware] skill.activation.denied skill={skill_name} reason={decision.reason}")
        return decision.is_allowed

    async def awrap_model_call(
        self, request: ModelRequest, handler
    ) -> ModelResponse | ExtendedModelResponse:
        """预路由 + 首次模型调用前强制激活 forced 技能 + 目录注入 catalog_candidates。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            带 `activated_skills` 状态更新命令的模型调用结果。
        """
        skills = self._visible_skills()
        if not skills:
            return await handler(request)

        context = request.runtime.context if request.runtime else None
        explicit_names = frozenset(context.explicit_skill_names) if context else frozenset()
        decision = route(skills, explicit_skill_names=explicit_names)
        already_activated: dict[str, dict] = dict(request.state.get("activated_skills") or {})

        skills_by_name = {skill.name: skill for skill in skills}
        blocks: list[str] = []
        newly_activated: dict[str, dict] = {}

        for name in decision.forced:
            if name in already_activated:
                continue
            if not await self._check_permission(context, name):
                continue
            skill = skills_by_name[name]
            try:
                body = await self._activation_service.activate(name, self._allowed_categories)
            except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
                logger.warning(f"[SkillMiddleware] skill.activation.failed skill={name} error={exc}")
                continue

            reason = decision.reasons[name]
            blocks.append(f'<{_SKILL_TAG} name="{name}" source="{reason}">\n{body}\n</{_SKILL_TAG}>')
            newly_activated[name] = {
                "skill_version": skill.version,
                "activation_source": reason,
                "activated_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.info(f"[SkillMiddleware] skill.activation.{reason} skill={name}")

        merged_activated = {**already_activated, **newly_activated}
        catalog_skills = [
            skills_by_name[name]
            for name in decision.catalog_candidates
            if name not in merged_activated
        ]
        catalog = build_skill_catalog(catalog_skills)

        system_prompt = request.system_prompt
        if blocks:
            skill_section = "\n\n".join(blocks)
            system_prompt = f"{system_prompt}\n\n{skill_section}" if system_prompt else skill_section
        if catalog:
            block = f"<{_SKILL_CATALOG_TAG}>\n{catalog}\n</{_SKILL_CATALOG_TAG}>"
            system_prompt = f"{system_prompt}\n\n{block}" if system_prompt else block
        if system_prompt != request.system_prompt:
            request = request.override(system_prompt=system_prompt)

        model_response = await handler(request)
        if not newly_activated:
            return model_response
        return ExtendedModelResponse(
            model_response=model_response,
            command=Command(update={"activated_skills": newly_activated}),
        )

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """拦截 `load_skill` 调用，短路完成权限校验 + 补充激活；其余工具调用原样透传。

        Args:
            request: 工具调用请求。
            handler: 执行工具调用的回调（非 `load_skill` 调用时原样转交）。

        Returns:
            `load_skill` 调用：携带加载结果或拒绝原因的 `ToolMessage`/更新
            `activated_skills` 的 `Command`，不会调用 `handler`；其余调用：
            `handler(request)` 的原始结果。
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

        already_activated: dict[str, dict] = dict(request.state.get("activated_skills") or {})
        if skill_name in already_activated:
            content = _ALREADY_ACTIVATED_TEMPLATE.format(skill_name=skill_name)
            return ToolMessage(content=content, tool_call_id=tool_call["id"])

        try:
            content = await self._activation_service.activate(skill_name, self._allowed_categories)
        except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
            return ToolMessage(content=str(exc), tool_call_id=tool_call["id"])

        logger.info(f"[SkillMiddleware] skill.activation.supplemental skill={skill_name}")
        skill = self._skill_manager.registry.get(skill_name)
        return Command(
            update={
                "activated_skills": {
                    skill_name: {
                        "skill_version": skill.version,
                        "activation_source": "load_skill",
                        "activated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
                "messages": [ToolMessage(content=content, tool_call_id=tool_call["id"])],
            }
        )
