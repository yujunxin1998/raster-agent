"""`SkillToolProvider.discover()` 单元测试：TOOL/WORKFLOW 技能分流
（Skill 注入重构设计文档第七节）。
"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.tools.registry.providers import SkillToolProvider


class _FakeGuardrailProvider:
    async def check(self, **kwargs):
        return GuardrailDecision.allow()


class _FakeSkillManager:
    def __init__(self, registry, enabled: bool = True) -> None:
        self.registry = registry
        self.enabled = enabled
        self.tool_factory = _FakeToolFactory()


class _FakeToolFactory:
    def create(self, skill):
        return object()  # 本测试只关心 ToolDefinition 的形状，不关心真实 StructuredTool


def _write_tool_skill(root: Path, *, name: str = "query_database", category: str = "database") -> None:
    skill_dir = root / name.replace("_", "-")
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntool_name: {name}\ndescription: 参数化工具技能\ncategory: {category}\n"
        "parameters:\n  - name: query_text\n    type: string\n    required: true\n---\n\n正文\n",
        encoding="utf-8",
    )


def _write_workflow_skill(root: Path, *, name: str = "data-analysis", category: str = "database") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntool_name: {name}\ndescription: 工作流技能\ncategory: {category}\n---\n\n正文\n",
        encoding="utf-8",
    )


def test_discover_returns_empty_when_skill_manager_disabled(tmp_path: Path) -> None:
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry, enabled=False), _FakeGuardrailProvider())

    assert provider.discover() == []


def test_tool_kind_skill_gets_its_own_definition(tmp_path: Path) -> None:
    _write_tool_skill(tmp_path)
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry), _FakeGuardrailProvider())

    definitions = provider.discover()

    model_names = {d.model_name for d in definitions}
    assert "query_database" in model_names
    assert "load_skill" not in model_names


def test_workflow_kind_skills_collapse_into_single_load_skill_definition(tmp_path: Path) -> None:
    _write_workflow_skill(tmp_path, name="data-analysis", category="database")
    _write_workflow_skill(tmp_path, name="consulting-analysis", category="general")
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry), _FakeGuardrailProvider())

    definitions = provider.discover()

    model_names = [d.model_name for d in definitions]
    assert model_names.count("load_skill") == 1
    assert "data-analysis" not in model_names
    assert "consulting-analysis" not in model_names


def test_mixed_registry_produces_tool_definitions_plus_one_load_skill(tmp_path: Path) -> None:
    _write_tool_skill(tmp_path, name="query_database", category="database")
    _write_workflow_skill(tmp_path, name="data-analysis", category="database")
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry), _FakeGuardrailProvider())

    definitions = provider.discover()

    model_names = [d.model_name for d in definitions]
    assert model_names.count("query_database") == 1
    assert model_names.count("load_skill") == 1
    assert "data-analysis" not in model_names


async def test_load_skill_definition_build_tool_produces_working_tool(tmp_path: Path) -> None:
    _write_workflow_skill(tmp_path, name="data-analysis", category="database")
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry), _FakeGuardrailProvider())

    definitions = provider.discover()
    load_skill_definition = next(d for d in definitions if d.model_name == "load_skill")
    tool = load_skill_definition.build_tool()

    result = await tool.coroutine(
        skill_name="data-analysis", config={"configurable": {"thread_id": "c1", "user_id": "u1"}}
    )

    assert "正文" in result


def test_registry_with_no_skills_produces_no_definitions(tmp_path: Path) -> None:
    registry = SkillLoader([tmp_path]).load()
    provider = SkillToolProvider(_FakeSkillManager(registry), _FakeGuardrailProvider())

    assert provider.discover() == []
