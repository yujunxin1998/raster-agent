"""application 级 Provider：把内置/Subagent 两类来源投影成 ToolDefinition。

Skill 不再是这里的一个来源——`docs/Skill与Tool完全解耦重构设计.md` 第 15
节阶段 6 之后，Skill 系统完全不进 `ToolRegistry`，`SkillMiddleware` 是
Skill 发现/激活的唯一入口（`skill_middleware.py`），跟这套"Tool 来源投影"
机制是两条独立链路。`build_tool` 直接复用 `build_task_tool`/已 import 的
内置 `@tool` 函数（设计文档 4.2 节）。前端工具是 request 级，走
`agent_core.tools.registry.frontend_provider`；MCP 走
`agent_core.tools.registry.mcp_provider`，均不在本模块。
"""
from __future__ import annotations

from typing import Protocol

from langchain_core.tools import BaseTool

from src.agent_core.agents.delegation_tools import build_task_tool
from src.agent_core.agents.subagent_profiles import list_subagent_types
from src.agent_core.tools.database_query_tool import query_database
from src.agent_core.tools.knowledge_search_tool import search_knowledge_base
from src.agent_core.tools.memory_tools import recall_memory, save_memory
from src.agent_core.tools.plan_tools import update_plan
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
    update_plan,
    # 重构文档（docs/Skill与Tool完全解耦重构设计.md）第 10 节：从"技能脚本"
    # 迁移成独立业务 Tool，原 Skill 名 query-database/search-knowledge-base
    # 已改造为纯指令型 Skill（database-analysis/knowledge-base-answering），
    # 通过 required_tools 声明指向这两个 Tool，不再各自生成 StructuredTool。
    query_database,
    search_knowledge_base,
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
# 声明为 `required_tools`，由 `subagent_capability_resolver.py` 派发前解析）
# 现取，不在这里登记——它不属于 Lead Agent 自己的工具集。
LEAD_AGENT_SKILL_CATEGORIES: tuple[str, ...] = (
    SkillCategory.GENERAL.value,
    SkillCategory.TOOL.value,
    SkillCategory.RAG.value,
    SkillCategory.DATABASE.value,
)


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
