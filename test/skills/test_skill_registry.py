"""`SkillRegistry.register()` 单元测试：同名 tool_name 显式报错，不静默覆盖。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.constants import SkillCategory
from src.common.exceptions import DuplicateSkillError


def _skill(tool_name: str, source: str = "public") -> SkillDefinition:
    return SkillDefinition(
        name=tool_name, tool_name=tool_name, description="", category=SkillCategory.GENERAL,
        skill_dir=Path(f"skills/{source}/{tool_name}"),
    )


def test_register_new_skill_succeeds() -> None:
    registry = SkillRegistry()

    registry.register(_skill("demo"))

    assert registry.get("demo").tool_name == "demo"


def test_register_duplicate_tool_name_raises() -> None:
    registry = SkillRegistry()
    registry.register(_skill("demo", source="core"))

    with pytest.raises(DuplicateSkillError):
        registry.register(_skill("demo", source="public"))

    # 第一个已注册的技能保持不变，不会被第二次调用篡改。
    assert registry.get("demo").source == "core"


def test_skill_loader_skips_duplicate_across_directories_without_crashing(tmp_path: Path) -> None:
    """两个技能根目录里各有一个同名 tool_name 的技能：只有先扫到的那个生效，
    第二个被跳过（记 error 日志），扫描过程本身不中断——`SkillLoader.load()`
    "单个技能失败只跳过"的既有容错策略天然覆盖了 `DuplicateSkillError`。
    """
    first_root = tmp_path / "core"
    second_root = tmp_path / "public"
    for root, description in ((first_root, "第一个"), (second_root, "第二个")):
        skill_dir = root / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: demo\ntool_name: demo\ndescription: {description}\ncategory: general\n---\n\n正文\n",
            encoding="utf-8",
        )

    registry = SkillLoader([first_root, second_root]).load()

    assert len(registry) == 1
    assert registry.get("demo").description == "第一个"
