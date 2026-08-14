"""`SkillActivationService` 单元测试（Skill 注入重构设计文档第四节）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_loader import SkillLoader
from src.common.exceptions import SkillDefinitionInvalidError, SkillNotFoundError


def _skills_public_root(tmp_path: Path) -> Path:
    """技能必须落在名为 public/core 的父目录下——`SkillDefinition.source`
    （= `skill_dir.parent.name`）要能匹配 SKILL.md 正文里写的
    `/mnt/skills/public/...` 前缀，`SkillPathRewriter` 才能正确替换。
    """
    root = tmp_path / "public"
    root.mkdir()
    return root


def _write_workflow_skill(root: Path, *, name: str = "demo-workflow", category: str = "general") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntool_name: {name}\ndescription: 测试工作流技能\ncategory: {category}\n---\n\n"
        f"# 指令\n运行 /mnt/skills/public/{name}/scripts/run.py\n",
        encoding="utf-8",
    )
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    (references_dir / "spec.md").write_text("参考内容", encoding="utf-8")


def _write_tool_skill(root: Path, *, name: str = "demo-tool") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ntool_name: {name}\ndescription: 测试工具技能\ncategory: database\n"
        "parameters:\n  - name: query\n    type: string\n    required: true\n---\n\n正文\n",
        encoding="utf-8",
    )


async def test_activate_returns_path_rewritten_instructions_and_references(tmp_path: Path) -> None:
    public_root = _skills_public_root(tmp_path)
    _write_workflow_skill(public_root)
    registry = SkillLoader([public_root]).load()
    service = SkillActivationService(registry)

    result = await service.activate("demo-workflow", allowed_categories=frozenset({"general"}))

    skill_dir = public_root / "demo-workflow"
    assert f"{skill_dir}/scripts/run.py" in result
    assert "/mnt/skills/public/demo-workflow" not in result
    assert "参考资料" in result
    assert "参考内容" in result


async def test_activate_unknown_skill_raises_not_found(tmp_path: Path) -> None:
    registry = SkillLoader([_skills_public_root(tmp_path)]).load()
    service = SkillActivationService(registry)

    with pytest.raises(SkillNotFoundError):
        await service.activate("nope", allowed_categories=frozenset({"general"}))


async def test_activate_out_of_scope_category_raises_not_found_not_permission_error(tmp_path: Path) -> None:
    """技能存在但不在调用方 allowed_categories 范围内，用"未找到"同一措辞，不泄露存在性。"""
    public_root = _skills_public_root(tmp_path)
    _write_workflow_skill(public_root, category="web_search")
    registry = SkillLoader([public_root]).load()
    service = SkillActivationService(registry)

    with pytest.raises(SkillNotFoundError):
        await service.activate("demo-workflow", allowed_categories=frozenset({"general"}))


async def test_activate_tool_kind_skill_raises_definition_invalid(tmp_path: Path) -> None:
    public_root = _skills_public_root(tmp_path)
    _write_tool_skill(public_root)
    registry = SkillLoader([public_root]).load()
    service = SkillActivationService(registry)

    with pytest.raises(SkillDefinitionInvalidError):
        await service.activate("demo-tool", allowed_categories=frozenset({"database"}))
