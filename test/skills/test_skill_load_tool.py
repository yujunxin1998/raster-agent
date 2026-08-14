"""`create_load_skill_tool` 自包含实现单元测试（Skill 注入重构设计文档第四节）。

这是给没有 `SkillMiddleware` 保护的调用方（`web-researcher` subagent）用的
真正执行路径，必须独立完成权限校验 + 加载，不依赖任何中间件。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_load_tool import create_load_skill_tool
from src.agent_core.skills.skill_loader import SkillLoader


class _FakeGuardrailProvider:
    def __init__(self, decision: GuardrailDecision) -> None:
        self._decision = decision
        self.last_call_kwargs: dict | None = None

    async def check(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._decision


def _write_workflow_skill(tmp_path: Path, *, name: str = "demo-workflow") -> Path:
    public_root = tmp_path / "public"
    skill_dir = public_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntool_name: {name}\ndescription: 测试工作流技能\ncategory: general\n---\n\n正文指令\n",
        encoding="utf-8",
    )
    return public_root


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", user_id="u1"))


async def test_allowed_skill_returns_activated_instructions(tmp_path: Path) -> None:
    public_root = _write_workflow_skill(tmp_path)
    registry = SkillLoader([public_root]).load()
    provider = _FakeGuardrailProvider(GuardrailDecision.allow())
    tool = create_load_skill_tool(
        SkillActivationService(registry), provider, allowed_categories=frozenset({"general"})
    )

    result = await tool.coroutine(skill_name="demo-workflow", runtime=_runtime())

    assert "正文指令" in result
    # 按具体技能名校验，不是恒定的 "load_skill"（保留可单独禁用某个技能的粒度）。
    assert provider.last_call_kwargs["tool_name"] == "demo-workflow"
    assert provider.last_call_kwargs["user_id"] == "u1"


async def test_denied_skill_returns_reason_without_loading(tmp_path: Path) -> None:
    public_root = _write_workflow_skill(tmp_path)
    registry = SkillLoader([public_root]).load()
    provider = _FakeGuardrailProvider(GuardrailDecision.deny("该技能已被禁用"))
    tool = create_load_skill_tool(
        SkillActivationService(registry), provider, allowed_categories=frozenset({"general"})
    )

    result = await tool.coroutine(skill_name="demo-workflow", runtime=_runtime())

    assert result == "该技能已被禁用"


async def test_unknown_skill_returns_error_text_not_exception(tmp_path: Path) -> None:
    public_root = _write_workflow_skill(tmp_path)
    registry = SkillLoader([public_root]).load()
    provider = _FakeGuardrailProvider(GuardrailDecision.allow())
    tool = create_load_skill_tool(
        SkillActivationService(registry), provider, allowed_categories=frozenset({"general"})
    )

    result = await tool.coroutine(skill_name="does-not-exist", runtime=_runtime())

    assert "未找到" in result
