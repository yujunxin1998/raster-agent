"""application 级 Provider：把内置/Skill/Subagent 三类来源投影成 ToolDefinition。

不重写任何一个已经稳定的执行工厂——`build_tool` 直接复用
`SkillToolFactory.create`/`build_task_tool`/已 import 的内置 `@tool` 函数
（设计文档 4.2 节）。前端工具是 request 级，走
`agent_core.tools.registry.frontend_provider`；MCP 走
`agent_core.tools.registry.mcp_provider`，均不在本模块。
"""
from __future__ import annotations

from typing import Protocol

from langchain_core.tools import BaseTool

from src.agent_core.agents.delegation_tools import build_task_tool
from src.agent_core.agents.subagent_profiles import list_subagent_types
from src.agent_core.guardrail.guardrail_provider import GuardrailProvider
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_definition import SkillKind
from src.agent_core.skills.skill_load_tool import create_load_skill_tool
from src.agent_core.skills.skill_manager import SkillManager
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.tools.memory_tools import recall_memory, save_memory
from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.agent_core.tools.sandbox_tool import read_file, run_command, run_python, save_output_file, write_file
from src.common.constants import SkillCategory


class ToolProvider(Protocol):
    """来源无关的发现接口（设计文档 4.2 节）。"""

    source_id: str

    def discover(self) -> list[ToolDefinition]:
        """返回该来源当前的全部工具定义。"""
        ...


# 内置工具：显式清单，随代码发布，不做反射自动扫描——设计文档"内置工具不
# 建议通过反射自动扫描"的原则，与 `lead_agent.py` 原来的字面量列表一致。
_BUILTIN_TOOLS: tuple[BaseTool, ...] = (
    save_memory,
    recall_memory,
    write_file,
    read_file,
    run_python,
    run_command,
    save_output_file,
)


class BuiltinToolProvider:
    """内置工具 Provider：进程级，随代码发布。"""

    source_id = "builtin"

    def discover(self) -> list[ToolDefinition]:
        """返回内置工具清单对应的定义（构造成本可忽略，直接返回已 import 的工具对象）。"""
        return [
            ToolDefinition(
                canonical_name=f"builtin.{tool.name}",
                model_name=tool.name,
                description=tool.description or "",
                source_type="builtin",
                source_id=self.source_id,
                scope="application",
                permissions_key=tool.name,
                build_tool=(lambda t=tool: t),
            )
            for tool in _BUILTIN_TOOLS
        ]


# 与原 `lead_agent.py::build_lead_agent` 里 `skill_manager.get_tools(category)`
# 的四次调用一一对应，`web_search` 分类留给 subagent（`web-researcher` profile
# 自己的 `tools_factory`）现取，不在这里登记——它不属于 Lead Agent 自己的工具集。
LEAD_AGENT_SKILL_CATEGORIES: tuple[str, ...] = (
    SkillCategory.GENERAL.value,
    SkillCategory.TOOL.value,
    SkillCategory.RAG.value,
    SkillCategory.DATABASE.value,
)


class SkillToolProvider:
    """Skill 工具 Provider：投影 `SkillRegistry`，按技能形态分流成两种定义。

    `SkillKind.TOOL` 技能（如 `search_knowledge_base`/`query_database`）：
    每个技能一条独立定义，执行仍然走 `SkillToolFactory.create`，与原逻辑
    完全不变。`SkillKind.WORKFLOW` 技能：不再逐个注册，收敛成一条
    `load_skill` 定义（对应 `docs/Skill注入与Load-Skill重构设计.md` 第七节）——
    实际加载与权限校验优先由 `agent_core.agents.skill_middleware.SkillMiddleware`
    拦截完成，这里注册的 `build_tool` 只是给模型看的 schema，兜底给没有
    该中间件的调用方（如 subagent）用，见 `skill_load_tool.py` 模块说明。

    `build_tool` 每次调用都现造（不缓存），与 `SkillManager.get_tools()` 原有的
    "现取不缓存"语义保持一致——渐进式披露落在 `SkillContentReader`，不受这层
    薄适配影响。
    """

    source_id = "skill"

    def __init__(self, skill_manager: SkillManager, guardrail_provider: GuardrailProvider) -> None:
        """初始化 Provider。

        Args:
            skill_manager: 全局 `SkillManager` 单例，取其 `registry` 做发现、
                `tool_factory` 做 TOOL 形态技能的惰性构建。
            guardrail_provider: 权限校验器，透传给 `load_skill` 的兜底实现
                （`create_load_skill_tool`）。
        """
        self._skill_manager = skill_manager
        self._guardrail_provider = guardrail_provider

    def discover(self, registry: SkillRegistry | None = None) -> list[ToolDefinition]:
        """按 `LEAD_AGENT_SKILL_CATEGORIES` 遍历已注册的全部技能，投影为定义。

        Args:
            registry: 显式传入时用它代替 `skill_manager.registry`——Skill
                热重载（`skill_hot_reload.py`）靠这个参数在还没有把新扫描出
                的 `SkillRegistry` 写回 `SkillManager` 之前，先算出候选
                `ToolDefinition` 列表去试发布；只有 `ToolRegistry.publish()`
                真正成功后才切换 `skill_manager.registry`，保证两者的切换
                是同一个原子操作的两半，不会出现"发布被拒绝，但
                SkillManager 已经换到新状态"的不一致。
        """
        if not self._skill_manager.enabled:
            return []

        factory = self._skill_manager.tool_factory
        source_registry = registry if registry is not None else self._skill_manager.registry
        activation_service = SkillActivationService(source_registry)
        definitions: list[ToolDefinition] = []
        has_workflow_skill = False

        for category in LEAD_AGENT_SKILL_CATEGORIES:
            for skill in source_registry.by_category(category):
                if skill.kind is SkillKind.WORKFLOW:
                    has_workflow_skill = True
                    continue
                definitions.append(
                    ToolDefinition(
                        canonical_name=f"skill.{category}.{skill.tool_name}",
                        model_name=skill.tool_name,
                        description=skill.description,
                        source_type="skill",
                        source_id=self.source_id,
                        scope="application",
                        permissions_key=skill.tool_name,
                        build_tool=(lambda s=skill: factory.create(s)),
                    )
                )

        if has_workflow_skill:
            definitions.append(
                ToolDefinition(
                    canonical_name="skill.load_skill",
                    model_name="load_skill",
                    description="按名称加载一个工作流技能的完整操作指令",
                    source_type="skill",
                    source_id=self.source_id,
                    scope="application",
                    permissions_key="load_skill",
                    build_tool=(
                        lambda: create_load_skill_tool(
                            activation_service,
                            self._guardrail_provider,
                            allowed_categories=frozenset(LEAD_AGENT_SKILL_CATEGORIES),
                        )
                    ),
                )
            )
        return definitions


class SubagentToolProvider:
    """`task` 分发工具 Provider：登记唯一一条 `subagent.task` 条目。

    对应设计文档"非目标"一节的取舍：不把每个 Subagent 拆成独立工具，
    `_PROFILES` 列表仍然作为 `task` 工具自身的动态约束元数据，不体现为
    多条 ToolDefinition。
    """

    source_id = "subagent"

    def discover(self) -> list[ToolDefinition]:
        """当前是否存在任何已注册的 subagent 类型，决定 `task` 工具是否登记。"""
        if not list_subagent_types():
            return []
        return [
            ToolDefinition(
                canonical_name="subagent.task",
                model_name="task",
                description="把子任务派给专用 subagent 执行",
                source_type="subagent",
                source_id=self.source_id,
                scope="application",
                permissions_key="task",
                build_tool=build_task_tool,
            )
        ]
