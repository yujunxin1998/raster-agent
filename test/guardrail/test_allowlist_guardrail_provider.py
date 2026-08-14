"""AllowlistGuardrailProvider 的 scope 支持单元测试。

工具注册中心引入 `ToolDefinition.permissions_scope` 后，`resolve_tools()`
会通过 `GuardrailContext.extra["scope"]` 把来源相关的 scope（如
"frontend"、"mcp:github"）传下来，`AllowlistGuardrailProvider.check()` 需要
把它转发给 `ToolPermissionStore.is_denied()`，而不是像之前那样恒传
"default"（设计文档六节）。
"""
from __future__ import annotations

from src.agent_core.guardrail.allowlist_guardrail_provider import AllowlistGuardrailProvider
from src.agent_core.guardrail.guardrail_provider import GuardrailContext


class _FakeSkillSettingsStore:
    async def is_disabled(self, user_id: str, tool_name: str) -> bool:
        return False


class _FakeToolPermissionStore:
    def __init__(self) -> None:
        self.last_call_kwargs: dict | None = None

    async def is_denied(self, user_id: str, tool_name: str, scope: str = "default") -> bool:
        self.last_call_kwargs = {"user_id": user_id, "tool_name": tool_name, "scope": scope}
        return False


async def test_check_forwards_scope_from_context_extra() -> None:
    tool_permission_store = _FakeToolPermissionStore()
    provider = AllowlistGuardrailProvider(_FakeSkillSettingsStore(), tool_permission_store)

    await provider.check(
        user_id="u1",
        tool_name="create_issue",
        tool_args={},
        context=GuardrailContext(extra={"scope": "mcp:github"}),
    )

    assert tool_permission_store.last_call_kwargs == {
        "user_id": "u1", "tool_name": "create_issue", "scope": "mcp:github",
    }


async def test_check_defaults_to_default_scope_when_not_provided() -> None:
    """兼容未传 `extra["scope"]` 的老调用点（如中间件链上的 `GuardrailMiddleware`）。"""
    tool_permission_store = _FakeToolPermissionStore()
    provider = AllowlistGuardrailProvider(_FakeSkillSettingsStore(), tool_permission_store)

    await provider.check(user_id="u1", tool_name="run_python", tool_args={}, context=GuardrailContext())

    assert tool_permission_store.last_call_kwargs["scope"] == "default"


async def test_check_no_user_id_short_circuits_without_touching_stores() -> None:
    tool_permission_store = _FakeToolPermissionStore()
    provider = AllowlistGuardrailProvider(_FakeSkillSettingsStore(), tool_permission_store)

    decision = await provider.check(user_id=None, tool_name="run_python", tool_args={}, context=GuardrailContext())

    assert decision.is_allowed
    assert tool_permission_store.last_call_kwargs is None
