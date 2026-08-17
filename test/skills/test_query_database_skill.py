"""`skills/core/database-analysis/`、`skills/core/knowledge-base-answering/`
两个技能能否被 `SkillLoader` 正常加载的冒烟测试。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 10 节迁移：`query_database`/
`search_knowledge_base` 从"参数化技能（经 SkillToolFactory 生成 StructuredTool）"
迁移成独立业务 Tool（`database_query_tool.py`/`knowledge_search_tool.py`），
原技能改造为纯指令型 WORKFLOW 技能，改名为 `database-analysis`/
`knowledge-base-answering`（避免技能名与新 Tool 名冲突），通过 `required_tools`
声明指向对应 Tool，不再有 `parameters`/脚本入口。
"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_loader import SkillLoader
from src.common.constants import SkillCategory

_SKILLS_CORE_DIR = Path(__file__).resolve().parents[2] / "skills" / "core"


def test_database_analysis_skill_loads_as_pure_workflow() -> None:
    registry = SkillLoader([_SKILLS_CORE_DIR]).load()

    skill = registry.get("database-analysis")

    assert skill.category == SkillCategory.DATABASE
    assert skill.required_tools == ("query_database",)


def test_knowledge_base_answering_skill_loads_as_pure_workflow() -> None:
    registry = SkillLoader([_SKILLS_CORE_DIR]).load()

    skill = registry.get("knowledge-base-answering")

    assert skill.category == SkillCategory.RAG
    assert skill.required_tools == ("search_knowledge_base",)
