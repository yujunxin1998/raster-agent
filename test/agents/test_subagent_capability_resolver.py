"""`resolve_subagent_capabilities()` 单元测试。

覆盖：必需 Tool 存在/缺失、可选 Tool 缺失降级、必需 Skill 缺失、注册表版本号
记录、不同用户经 Guardrail 得到不同能力集合、失败信息不泄漏敏感内容。
"在途任务不受后续 Registry 切换影响" 覆盖在
`test_sub_agent_config_isolation.py`（需要真正验证"解析结果被持有后，源头
被替换也不影响"，跟本文件"解析本身对不对"是两个层面）。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent_core.agents.subagent_capability_resolver import (
    SubagentCapabilityUnavailable,
    resolve_subagent_capabilities,
)
from src.agent_core.agents.subagent_profiles import SubagentProfile
from src.agent_core.guardrail.guardrail_provider import GuardrailDecision, GuardrailProvider
from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.common.constants import SkillCategory


def _tool_definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        canonical_name=f"builtin.{name}", model_name=name, description="",
        source_type="builtin", source_id="builtin", scope="application",
        permissions_key=name, build_tool=lambda: MagicMock(name=f"tool:{name}"),
    )


def _snapshot(*tool_names: str) -> RegistrySnapshot:
    snapshot = RegistrySnapshot.empty()
    if tool_names:
        snapshot = snapshot.replace_source("builtin", [_tool_definition(n) for n in tool_names])
    return snapshot


def _skill_manager(*skills: SkillDefinition):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return SimpleNamespace(registry=registry)


def _skill_definition(name: str) -> SkillDefinition:
    return SkillDefinition(
        name=name, description="test skill", category=SkillCategory.WEB_SEARCH,
        skill_dir=Path("skills/core") / name, activation="automatic",
    )


class _FakeGuardrailProvider(GuardrailProvider):
    """按 `tool_name`/`user_id` 决定放行还是拒绝的假 Provider。"""

    def __init__(self, deny_tool_names: frozenset[str] = frozenset(),
                 deny_for_user: dict[str, frozenset[str]] | None = None, deny_reason: str = "未获授权执行") -> None:
        self._deny_tool_names = deny_tool_names
        self._deny_for_user = deny_for_user or {}
        self._deny_reason = deny_reason

    async def check(self, *, user_id, tool_name, tool_args, context) -> GuardrailDecision:
        deny_names = self._deny_tool_names | self._deny_for_user.get(user_id, frozenset())
        if tool_name in deny_names:
            return GuardrailDecision.deny(self._deny_reason)
        return GuardrailDecision.allow()


_ALLOW_ALL = _FakeGuardrailProvider()


def _profile(**overrides) -> SubagentProfile:
    defaults = dict(name="test-agent", description="test", system_prompt_factory=lambda: "sp")
    defaults.update(overrides)
    return SubagentProfile(**defaults)


async def test_required_tool_present_resolves_successfully() -> None:
    profile = _profile(required_tools=frozenset({"web_search"}))

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
        subagent_only_tools={"web_search": MagicMock(name="web_search")},
        guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert len(resolved.tools) == 1
    assert resolved.missing_optional == ()


async def test_required_tool_missing_raises_capability_unavailable() -> None:
    """必需能力缺失（在 ToolRegistry 快照和 subagent 专属兜底表里都查不到）
    必须 fail fast，不能让缺了搜索工具的 web-researcher 继续跑。
    """
    profile = _profile(required_tools=frozenset({"web_search"}))

    with pytest.raises(SubagentCapabilityUnavailable) as exc_info:
        await resolve_subagent_capabilities(
            profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
            subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
        )

    assert exc_info.value.subagent_type == "test-agent"
    assert exc_info.value.missing == ["web_search"]


async def test_required_tool_found_via_tool_registry_snapshot() -> None:
    """跟 `SUBAGENT_ONLY_TOOLS` 不同的另一条来源：本来就登记在共享
    `ToolRegistry` 里的工具（比如未来的 MCP 工具）也应该能被解析到。
    """
    profile = _profile(required_tools=frozenset({"registry_tool"}))

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot("registry_tool"), skill_manager=_skill_manager(),
        subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert len(resolved.tools) == 1


async def test_optional_tool_missing_degrades_instead_of_failing() -> None:
    profile = _profile(
        required_tools=frozenset({"web_search"}), optional_tools=frozenset({"maybe_tool"}),
    )

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
        subagent_only_tools={"web_search": MagicMock(name="web_search")},
        guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert resolved.missing_optional == ("maybe_tool",)
    assert len(resolved.tools) == 1  # 只有 web_search，maybe_tool 被剔除


async def test_required_skill_missing_raises_capability_unavailable() -> None:
    profile = _profile(required_skills=frozenset({"some-required-skill"}))

    with pytest.raises(SubagentCapabilityUnavailable) as exc_info:
        await resolve_subagent_capabilities(
            profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
            subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
        )

    assert exc_info.value.missing == ["some-required-skill"]


async def test_required_skill_present_resolves_successfully() -> None:
    profile = _profile(required_skills=frozenset({"some-required-skill"}))
    skill_manager = _skill_manager(_skill_definition("some-required-skill"))

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=skill_manager,
        subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert resolved.missing_optional == ()


async def test_registry_revisions_are_recorded_on_resolved_result() -> None:
    profile = _profile()
    snapshot = _snapshot("registry_tool")  # revision=1（RegistrySnapshot.empty() 是 0，replace_source 一次 +1）
    skill_manager = _skill_manager()
    skill_manager.registry.revision = 3

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=snapshot, skill_manager=skill_manager,
        subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert resolved.tool_registry_revision == snapshot.revision == 1
    assert resolved.skill_registry_revision == 3


async def test_different_users_get_different_capabilities_via_guardrail() -> None:
    profile = _profile(required_tools=frozenset({"web_search"}))
    guardrail = _FakeGuardrailProvider(deny_for_user={"blocked-user": frozenset({"web_search"})})
    subagent_only_tools = {"web_search": MagicMock(name="web_search")}

    resolved_allowed = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
        subagent_only_tools=subagent_only_tools, guardrail_provider=guardrail, user_id="normal-user",
    )
    assert len(resolved_allowed.tools) == 1

    with pytest.raises(SubagentCapabilityUnavailable) as exc_info:
        await resolve_subagent_capabilities(
            profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
            subagent_only_tools=subagent_only_tools, guardrail_provider=guardrail, user_id="blocked-user",
        )
    assert exc_info.value.missing == ["web_search"]


async def test_failure_message_does_not_leak_guardrail_reason_or_internal_details() -> None:
    """`SubagentCapabilityUnavailable` 的消息只应该包含能力名——Guardrail 的
    拒绝原因（可能带内部策略细节）只进日志，不进异常消息，更不会被
    `task` 工具原样透传给模型/用户。
    """
    profile = _profile(required_tools=frozenset({"web_search"}))
    sensitive_reason = "denied: internal policy secret-token=sk-abcdef1234567890 path=/etc/secrets"
    guardrail = _FakeGuardrailProvider(deny_tool_names=frozenset({"web_search"}), deny_reason=sensitive_reason)

    with pytest.raises(SubagentCapabilityUnavailable) as exc_info:
        await resolve_subagent_capabilities(
            profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
            subagent_only_tools={"web_search": MagicMock(name="web_search")},
            guardrail_provider=guardrail, user_id="u1",
        )

    message = str(exc_info.value)
    assert "sk-abcdef1234567890" not in message
    assert "/etc/secrets" not in message
    assert exc_info.value.missing == ["web_search"]


async def test_allowed_skill_categories_mounts_skill_middleware() -> None:
    from src.agent_core.agents.skill_middleware import SkillMiddleware

    profile = _profile(allowed_skill_categories=frozenset({SkillCategory.WEB_SEARCH.value}))

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
        subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert len(resolved.middleware) == 1
    assert isinstance(resolved.middleware[0], SkillMiddleware)


async def test_no_skill_categories_means_no_middleware_mounted() -> None:
    profile = _profile()

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=_snapshot(), skill_manager=_skill_manager(),
        subagent_only_tools={}, guardrail_provider=_ALLOW_ALL, user_id="u1",
    )

    assert resolved.middleware == []
