"""`SkillDefinition.kind` 派生属性单元测试（对应 Skill 注入重构设计文档第三节）。"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_definition import SkillDefinition, SkillKind
from src.common.constants import SkillCategory


def _skill(*, parameters: list[dict]) -> SkillDefinition:
    return SkillDefinition(
        name="demo",
        tool_name="demo",
        description="",
        category=SkillCategory.GENERAL,
        skill_dir=Path("skills/public/demo"),
        parameters=parameters,
    )


def test_skill_with_parameters_is_tool_kind() -> None:
    skill = _skill(parameters=[{"name": "query", "type": "string", "required": True}])

    assert skill.kind is SkillKind.TOOL


def test_skill_without_parameters_is_workflow_kind() -> None:
    skill = _skill(parameters=[])

    assert skill.kind is SkillKind.WORKFLOW
