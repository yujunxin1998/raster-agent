"""统一工具定义值对象：注册中心的核心数据模型。

对应设计文档《工具注册中心与热重载设计.md》第三节：一个只读的登记表条目，
由各 Provider 从各自的权威定义（`SkillDefinition`/`FrontendTool`/MCP
`mcp.types.Tool`/...）投影出来，执行仍然复用原有工厂（`SkillToolFactory`/
`CustomToolConverter`/`build_task_tool`），本类不参与执行，只参与
"发现 -> 校验 -> 裁剪"这条链路。

沿用项目内部值对象的惯例（`SkillDefinition`/`SubagentProfile` 均为
`@dataclass(frozen=True)`），不用 pydantic BaseModel——`build_tool` 是一个
不可 JSON 序列化的 Callable，且本类从不跨越 API/持久化边界。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from langchain_core.tools import BaseTool

SourceType = Literal["builtin", "skill", "frontend", "subagent", "mcp"]
ToolScope = Literal["application", "request"]

_DEFAULT_PERMISSION_SCOPE = "default"

# 来源可信度优先级，数字越小越可信。用于 resolve_tools 里 request 级
# （前端）定义与 application 级快照冲突时的取舍——builtin 最可信，frontend
# 是请求方自报的内容，可信度最低（设计文档 4.3 节）。
_SOURCE_PRIORITY: dict[SourceType, int] = {
    "builtin": 0,
    "skill": 1,
    "subagent": 1,
    "mcp": 1,
    "frontend": 2,
}


def source_priority(source_type: SourceType) -> int:
    """返回来源的可信度优先级，数字越小优先级越高。"""
    return _SOURCE_PRIORITY.get(source_type, 99)


@dataclass(frozen=True)
class ToolDefinition:
    """一个来源无关的工具登记条目。

    Attributes:
        canonical_name: 全局唯一名，带来源命名空间，如
            "skill.rag.search_knowledge_base"、"mcp.github.create_issue"。
        model_name: 模型实际看到的短名（通常等于原始 tool_name）。
        description: 工具描述，仅用于日志/审计展示；真正喂给模型的
            description 来自 build_tool() 产出的 BaseTool 自身。
        source_type: 来源类型。
        source_id: 来源实例标识，如 "builtin"、"skill"、"mcp:github"、
            "frontend:{client_id}:{conversation_id}"，用于
            `RegistrySnapshot.replace_source` 按来源整体替换/摘除。
        scope: "application"（启动后常驻，全部请求可见）或
            "request"（仅本次请求可见，当前只有前端工具属于这一档）。
        permissions_key: 传给 `GuardrailProvider.check()` 的 tool_name。
        permissions_scope: 传给 `GuardrailProvider.check()` 的权限维度
            （对应 `ToolPermissionStore.scope` 列），默认值等同于
            `source_type`；MCP 来源按 Server 粒度单独设置
            `f"mcp:{server_id}"`，避免一次性放开"全部 MCP 工具"（设计文档
            六节）。
        enabled: 是否参与本次解析，False 时 `resolve_tools` 直接跳过。
        build_tool: 惰性构建实际可执行的 BaseTool，真正复用原有工厂——
            每次调用都现造，与现有 `skill_manager.get_tools()` 的"现取不
            缓存"语义保持一致，不引入新的跨请求共享状态。
    """

    canonical_name: str
    model_name: str
    description: str
    source_type: SourceType
    source_id: str
    scope: ToolScope
    permissions_key: str
    build_tool: Callable[[], BaseTool]
    permissions_scope: str = _DEFAULT_PERMISSION_SCOPE
    enabled: bool = True


@dataclass(frozen=True)
class RejectedTool:
    """`resolve_tools` 阶段被裁剪掉的一条工具定义及原因。"""

    canonical_name: str
    model_name: str
    reason: str


@dataclass(frozen=True)
class ResolvedToolSet:
    """`resolve_tools` 的返回结果。

    Attributes:
        registry_revision: 本次解析所依据的 `RegistrySnapshot.revision`，
            写入 `AgentRuntimeContext` 后可用于事后追溯"这次对话当时
            模型实际能看到哪些工具"。
        tools: 实际喂给 Agent 的 BaseTool 列表。
        rejected: 被 Guardrail 拒绝或静默覆盖丢弃的定义列表。
    """

    registry_revision: int
    tools: list[BaseTool]
    rejected: list[RejectedTool]
