"""`task` 通用分发工具的 subagent 类型注册表。

对应 DeerFlow 的 `SubagentExecutor` 思路：把"专用能力"registered 成一个个
`SubagentProfile`，`task(subagent_type, task)` 工具按 `subagent_type` 查表拿到
`system_prompt`/`tools`，交给 `sub_agent_factory.py::run_subagent()` 执行——新增
一个专用能力只需要在 `_PROFILES` 里加一条，不需要再写一遍"新建 builder 函数 + 接进
`lead_agent.py` 的 `base_tools` + 写新提示词"这一整套（这正是本模块要替代的旧模式，
见 `delegation_tools.py` 历史版本的 `build_web_search_delegate_tool()` 等函数）。

`system_prompt_factory`/`tools_factory` 都设计成惰性求值（调用 `task` 工具时才执行），
原因和旧版 `delegation_tools.py` 一致：`web-researcher` 需要每次现算 `{current_date}`
（避免长期运行的进程把日期冻结在启动那一刻），技能工具集依赖 `SkillManager` 单例
（只有应用完成启动后才可用，不能在模块导入阶段就解析）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from langchain_core.tools import BaseTool

from src.agent_core.guardrail import get_guardrail_provider
from src.agent_core.prompts import prompt_factory
from src.agent_core.skills import get_skill_manager
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_catalog import build_skill_catalog
from src.agent_core.skills.skill_definition import SkillKind
from src.agent_core.skills.skill_load_tool import create_load_skill_tool
from src.agent_core.tools.web_search_tool import web_search
from src.common.constants import SkillCategory

_WEB_RESEARCHER_DESCRIPTION = (
    "联网搜索专家：获取实时资讯、URL、在线数据等。返回的是中间搜索结果，不是最终答案，"
    "拿到结果后通常还需要你自己进一步处理。传入需要搜索的具体内容。"
)

_WEB_SEARCH_CATEGORY = SkillCategory.WEB_SEARCH.value


def _web_researcher_system_prompt() -> str:
    """现算系统提示词：静态模板 + 当前 `web_search` 分类下 WORKFLOW 技能目录。

    `web-researcher` 子 Agent 不挂任何中间件（见 `sub_agent_factory.py` 的安全
    边界说明），没有 `SkillMiddleware` 可以动态注入目录，只能在这里现算一次——
    与 Lead Agent 路径的"目录注入放中间件"不同，是子 Agent 无中间件这一既有
    架构约束下的必然选择，不是设计偏好（对应
    `docs/Skill注入与Load-Skill重构设计.md` 第四、五节"两条加载路径共存"）。
    """
    prompt = prompt_factory.get("WEB_SEARCH_AGENT").replace("{current_date}", date.today().isoformat())
    workflow_skills = [
        skill
        for skill in get_skill_manager().registry.by_category(_WEB_SEARCH_CATEGORY)
        if skill.kind is SkillKind.WORKFLOW
    ]
    catalog = build_skill_catalog(workflow_skills)
    if catalog:
        prompt += f"\n\n<skill_catalog>\n{catalog}\n</skill_catalog>"
    return prompt


def _web_researcher_tools() -> list[BaseTool]:
    """现算 `web-researcher` 的工具集：`TOOL` 形态技能各自一个工具，`WORKFLOW`
    形态技能收敛成一个 `load_skill`（自包含实现，见 `skill_load_tool.py`
    模块说明——这条路径没有中间件拦截，`load_skill` 自己的 `_invoke` 就是
    真正被执行的那份逻辑，不是兜底）。
    """
    skill_manager = get_skill_manager()
    factory = skill_manager.tool_factory
    tools: list[BaseTool] = [web_search]
    has_workflow_skill = False

    for skill in skill_manager.registry.by_category(_WEB_SEARCH_CATEGORY):
        if skill.kind is SkillKind.WORKFLOW:
            has_workflow_skill = True
            continue
        tools.append(factory.create(skill))

    if has_workflow_skill:
        tools.append(
            create_load_skill_tool(
                SkillActivationService(skill_manager.registry),
                get_guardrail_provider(),
                allowed_categories=frozenset({_WEB_SEARCH_CATEGORY}),
            )
        )
    return tools


@dataclass(frozen=True)
class SubagentProfile:
    """一个可被 `task` 工具派发的专用子 Agent 配置。

    Attributes:
        name: subagent 类型标识，即 `task(subagent_type=...)` 的取值。
        description: 展示在 `task` 工具描述里，供 Lead Agent 判断何时选用。
        system_prompt_factory: 子 Agent 系统提示词的惰性构建函数。
        tools_factory: 子 Agent 工具集的惰性构建函数（**不得包含沙箱执行类工具**，
            见 `sub_agent_factory.py` 模块 docstring 的安全边界说明）。
    """

    name: str
    description: str
    system_prompt_factory: Callable[[], str]
    tools_factory: Callable[[], list[BaseTool]]


_PROFILES: dict[str, SubagentProfile] = {
    "web-researcher": SubagentProfile(
        name="web-researcher",
        description=_WEB_RESEARCHER_DESCRIPTION,
        system_prompt_factory=_web_researcher_system_prompt,
        tools_factory=_web_researcher_tools,
    ),
}


def get_profile(subagent_type: str) -> SubagentProfile | None:
    """按类型名查找已注册的 subagent profile，未注册时返回 None。"""
    return _PROFILES.get(subagent_type)


def list_subagent_types() -> list[str]:
    """返回当前已注册的全部 subagent 类型名，用于工具描述和错误提示。"""
    return list(_PROFILES.keys())
