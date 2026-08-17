"""把一组 WORKFLOW 技能格式化成注入系统提示词的技能目录文本。

被 `agent_core.agents.skill_middleware.SkillMiddleware`（Lead Agent，动态注入）
和 `agent_core.agents.subagent_profiles`（subagent，静态拼接一次）共用同一份
格式化逻辑，避免两处目录文案各写一套、后续改格式要改两遍。
"""
from __future__ import annotations

from src.agent_core.skills.skill_definition import SkillDefinition

_CATALOG_HEADER = (
    "以下是可通过 `load_skill(skill_name)` 加载的工作流技能，"
    "加载后按返回的指令执行，不要凭空猜测流程："
)


def build_skill_catalog(skills: list[SkillDefinition]) -> str:
    """按技能名排序生成一段 `name: description` 目录文本。

    Args:
        skills: 待列出的 WORKFLOW 技能列表。

    Returns:
        目录文本；`skills` 为空时返回空字符串（调用方应据此决定要不要
        整段跳过，不要注入一个只有标题没有条目的空目录）。
    """
    if not skills:
        return ""
    lines = [_CATALOG_HEADER]
    for skill in sorted(skills, key=lambda s: s.name):
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)
