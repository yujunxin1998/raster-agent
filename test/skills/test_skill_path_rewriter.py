"""`SkillPathRewriter` 单元测试（Skill 注入重构设计文档第六节）。"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_path_rewriter import SkillPathRewriter
from src.common.constants import SkillCategory


def _skill(skill_dir: Path) -> SkillDefinition:
    return SkillDefinition(
        name="data-analysis",
        tool_name="data-analysis",
        description="",
        category=SkillCategory.DATABASE,
        skill_dir=skill_dir,
    )


def test_rewrites_own_script_root_to_real_absolute_path() -> None:
    skill_dir = Path("/srv/raster-agent/skills/public/data-analysis")
    skill = _skill(skill_dir)
    text = "python /mnt/skills/public/data-analysis/scripts/analyze.py --action inspect"

    result = SkillPathRewriter().rewrite(skill, text)

    assert result == f"python {skill_dir}/scripts/analyze.py --action inspect"


def test_does_not_touch_other_skill_names_in_text() -> None:
    """只替换"这个技能自己的"路径前缀，不是全局正则——正文里恰好提到别的技能名字符串不受影响。"""
    skill_dir = Path("/srv/raster-agent/skills/public/data-analysis")
    skill = _skill(skill_dir)
    text = "参考 /mnt/skills/public/chart-visualization/scripts/generate.js 的输出格式"

    result = SkillPathRewriter().rewrite(skill, text)

    assert result == text


def test_rewrites_uploads_outputs_workspace_tokens() -> None:
    skill = _skill(Path("/srv/skills/public/data-analysis"))
    text = (
        "上传文件在 /mnt/user-data/uploads/data.xlsx，"
        "导出到 /mnt/user-data/outputs/result.csv，"
        "中间文件写到 /mnt/user-data/workspace/tmp.csv"
    )

    result = SkillPathRewriter().rewrite(skill, text)

    assert result == (
        "上传文件在 ../uploads/data.xlsx，导出到 ../outputs/result.csv，中间文件写到 ./tmp.csv"
    )


def test_core_source_uses_core_prefix() -> None:
    skill_dir = Path("/srv/skills/core/query-database")
    skill = _skill(skill_dir)
    text = "见 /mnt/skills/core/query-database/scripts/main.py"

    result = SkillPathRewriter().rewrite(skill, text)

    assert result == f"见 {skill_dir}/scripts/main.py"
