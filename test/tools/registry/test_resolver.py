"""resolve_tools() 单元测试：冲突优先级合并 + 权限裁剪（设计文档 4.3 节）。"""
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.tools.registry.resolver import _merge, resolve_tools
from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import ToolDefinition


def _fake_tool(name: str) -> BaseTool:
    return StructuredTool(name=name, description=name, args_schema=None, func=lambda: "ok")


def _definition(
    *, model_name: str, source_type: str, source_id: str | None = None, enabled: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        canonical_name=f"{source_type}.{model_name}",
        model_name=model_name,
        description="",
        source_type=source_type,
        source_id=source_id or source_type,
        scope="application" if source_type != "frontend" else "request",
        permissions_key=model_name,
        enabled=enabled,
        build_tool=(lambda n=model_name: _fake_tool(n)),
    )


class _FakeGuardrailProvider:
    """按 tool_name 精确匹配拒绝名单的伪权限校验器。"""

    def __init__(self, denied: set[str] | None = None) -> None:
        self._denied = denied or set()
        self.checked_tool_names: list[str] = []

    async def check(self, *, user_id, tool_name, tool_args, context):  # noqa: D102
        self.checked_tool_names.append(tool_name)
        if tool_name in self._denied:
            return GuardrailDecision.deny(f"{tool_name} 被拒绝")
        return GuardrailDecision.allow()


def test_merge_builtin_wins_over_frontend_same_model_name() -> None:
    """builtin > frontend：前端同名工具应该被丢弃，而不是覆盖内置工具
    （修复设计文档 1.4 节指出的静默覆盖问题）。"""
    application = {"search": _definition(model_name="search", source_type="builtin")}
    frontend_definition = _definition(model_name="search", source_type="frontend")

    merged = _merge(application, [frontend_definition])

    assert merged["search"].source_type == "builtin"


def test_merge_frontend_only_tool_is_kept() -> None:
    application: dict[str, ToolDefinition] = {}
    frontend_definition = _definition(model_name="show_map", source_type="frontend")

    merged = _merge(application, [frontend_definition])

    assert merged["show_map"].source_type == "frontend"


async def test_resolve_tools_skips_disabled_definitions_without_guardrail_check() -> None:
    snapshot = RegistrySnapshot.empty().replace_source(
        "builtin", [_definition(model_name="disabled_tool", source_type="builtin", enabled=False)]
    )
    provider = _FakeGuardrailProvider()

    resolved = await resolve_tools(snapshot=snapshot, user_id="u1", guardrail_provider=provider)

    assert resolved.tools == []
    assert resolved.rejected == []
    assert "disabled_tool" not in provider.checked_tool_names


async def test_resolve_tools_denied_definition_goes_to_rejected_not_tools() -> None:
    snapshot = RegistrySnapshot.empty().replace_source(
        "builtin",
        [
            _definition(model_name="allowed_tool", source_type="builtin"),
            _definition(model_name="denied_tool", source_type="builtin"),
        ],
    )
    provider = _FakeGuardrailProvider(denied={"denied_tool"})

    resolved = await resolve_tools(snapshot=snapshot, user_id="u1", guardrail_provider=provider)

    resolved_tool_names = {tool.name for tool in resolved.tools}
    rejected_model_names = {r.model_name for r in resolved.rejected}
    assert resolved_tool_names == {"allowed_tool"}
    assert rejected_model_names == {"denied_tool"}
    assert resolved.registry_revision == snapshot.revision


async def test_resolve_tools_passes_permissions_scope_into_guardrail_context() -> None:
    captured_scopes: list[str] = []

    class _ScopeCapturingProvider:
        async def check(self, *, user_id, tool_name, tool_args, context):  # noqa: D102
            captured_scopes.append(context.extra.get("scope"))
            return GuardrailDecision.allow()

    definition = ToolDefinition(
        canonical_name="mcp.github.create_issue",
        model_name="create_issue",
        description="",
        source_type="mcp",
        source_id="mcp:github",
        scope="application",
        permissions_key="create_issue",
        permissions_scope="mcp:github",
        build_tool=(lambda: _fake_tool("create_issue")),
    )
    snapshot = RegistrySnapshot.empty().replace_source("mcp:github", [definition])

    await resolve_tools(snapshot=snapshot, user_id="u1", guardrail_provider=_ScopeCapturingProvider())

    assert captured_scopes == ["mcp:github"]
