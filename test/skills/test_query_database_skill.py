"""`skills/core/query-database/` 技能能否被 `SkillLoader` 正常加载的冒烟测试。

对应本轮重构：`query_database` 从委派子 Agent 里的 `@tool` 函数改成一个真正的
`SKILL.md` 技能，`tool_name` 必须保持 `query_database` 不变——这是
`DatasourceRoutingMiddleware` 硬校验能否对上的关键（见该中间件的说明）。
"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_loader import SkillLoader
from src.common.constants import SkillCategory

_SKILLS_CORE_DIR = Path(__file__).resolve().parents[2] / "skills" / "core"


def test_query_database_skill_loads_with_expected_shape() -> None:
    registry = SkillLoader([_SKILLS_CORE_DIR]).load()

    skill = registry.get("query_database")

    assert skill.tool_name == "query_database"
    assert skill.category == SkillCategory.DATABASE
    assert skill.has_script()
    assert [p["name"] for p in skill.parameters] == ["query_text"]
    assert "datasource_id" in skill.runtime_context_keys


def test_search_knowledge_base_skill_still_loads() -> None:
    """回归检查：修 rag_service.py 导入路径没有连带弄坏这个技能本身能不能加载。"""
    registry = SkillLoader([_SKILLS_CORE_DIR]).load()

    skill = registry.get("search_knowledge_base")

    assert skill.category == SkillCategory.RAG
    assert skill.has_script()
