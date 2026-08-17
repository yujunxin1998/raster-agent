"""统一工具注册中心（设计文档《工具注册中心与热重载设计.md》）。

使用方式::

    from src.agent_core.tools.registry import get_tool_registry, resolve_tools

    resolved = await resolve_tools(
        snapshot=get_tool_registry().current_snapshot(),
        request_definitions=frontend_definitions,
        user_id=user_id,
        guardrail_provider=get_guardrail_provider(),
    )
    tools = resolved.tools

`main.py` 的 lifespan 里调用 `init_tool_registry()` 一次，随后依次
`await get_tool_registry().publish(...)` 发布 builtin/subagent/mcp 三类
application 级来源（Skill 不再是其中之一，见 `providers.py` 模块文档）；
前端工具是 request 级，不进这个全局单例，见
`frontend_provider.discover_frontend_tools`。
"""
from src.agent_core.tools.registry.frontend_provider import discover_frontend_tools
from src.agent_core.tools.registry.mcp_provider import (
    MCPConnectionManager,
    MCPServerConfig,
    get_mcp_connection_manager,
    init_mcp_connection_manager,
    parse_mcp_servers,
    shutdown_mcp_connection_manager,
)
from src.agent_core.tools.registry.providers import (
    LEAD_AGENT_SKILL_CATEGORIES,
    BuiltinToolProvider,
    SubagentToolProvider,
    ToolProvider,
)
from src.agent_core.tools.registry.resolver import resolve_tools
from src.agent_core.tools.registry.skill_hot_reload import (
    SkillHotReloader,
    get_skill_hot_reloader,
    init_skill_hot_reloader,
    shutdown_skill_hot_reloader,
)
from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import (
    RejectedTool,
    ResolvedToolSet,
    SourceType,
    ToolDefinition,
    ToolScope,
)
from src.agent_core.tools.registry.tool_registry import ToolRegistry, get_tool_registry, init_tool_registry

__all__ = [
    "ToolDefinition",
    "RejectedTool",
    "ResolvedToolSet",
    "SourceType",
    "ToolScope",
    "RegistrySnapshot",
    "ToolRegistry",
    "init_tool_registry",
    "get_tool_registry",
    "ToolProvider",
    "LEAD_AGENT_SKILL_CATEGORIES",
    "BuiltinToolProvider",
    "SubagentToolProvider",
    "discover_frontend_tools",
    "resolve_tools",
    "MCPServerConfig",
    "MCPConnectionManager",
    "parse_mcp_servers",
    "init_mcp_connection_manager",
    "get_mcp_connection_manager",
    "shutdown_mcp_connection_manager",
    "SkillHotReloader",
    "init_skill_hot_reloader",
    "get_skill_hot_reloader",
    "shutdown_skill_hot_reloader",
]
