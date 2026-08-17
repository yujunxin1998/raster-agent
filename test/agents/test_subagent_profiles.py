"""`subagent_profiles.py` 单元测试：`SubagentProfile` 现在只是能力声明
（`required_tools`/`allowed_skill_categories` 等字段），不再直接持有构造好的
`BaseTool`/`AgentMiddleware` 对象——真正的解析覆盖在
`test_subagent_capability_resolver.py`，这里只测声明本身对不对。
"""
from __future__ import annotations

from src.agent_core.agents.subagent_profiles import SUBAGENT_ONLY_TOOLS, get_profile, list_subagent_types
from src.agent_core.tools.web_search_tool import web_search
from src.common.constants import SkillCategory


def test_web_researcher_is_the_only_registered_type() -> None:
    assert list_subagent_types() == ["web-researcher"]


def test_web_researcher_declares_web_search_as_required_tool() -> None:
    profile = get_profile("web-researcher")

    assert profile.required_tools == frozenset({"web_search"})
    assert profile.optional_tools == frozenset()
    assert profile.required_skills == frozenset()


def test_web_researcher_declares_web_search_skill_category() -> None:
    profile = get_profile("web-researcher")

    assert profile.allowed_skill_categories == frozenset({SkillCategory.WEB_SEARCH.value})


def test_web_researcher_system_prompt_is_lazily_generated() -> None:
    profile = get_profile("web-researcher")

    prompt = profile.system_prompt_factory()

    assert isinstance(prompt, str)
    assert prompt


def test_subagent_only_tools_covers_web_search() -> None:
    """`web_search` 故意不登记进共享 `ToolRegistry`（见 `providers.py` 的
    说明），`SUBAGENT_ONLY_TOOLS` 是 `subagent_capability_resolver.py` 解析
    `required_tools`/`optional_tools` 时的兜底来源，必须包含它。
    """
    assert SUBAGENT_ONLY_TOOLS.get("web_search") is web_search
