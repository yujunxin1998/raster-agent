"""`task` 通用分发工具的 subagent 类型注册表。

对应 DeerFlow 的 `SubagentExecutor` 思路：把"专用能力"registered 成一个个
`SubagentProfile`，`task(subagent_type, task)` 工具按 `subagent_type` 查表拿到
`system_prompt`/能力依赖声明，交给 `subagent_capability_resolver.py::
resolve_subagent_capabilities()` 解析出实际可用的 Tool/Middleware，再由
`sub_agent_factory.py::run_subagent()` 执行——新增一个专用能力只需要在
`_PROFILES` 里加一条，不需要再写一遍"新建 builder 函数 + 接进 `lead_agent.py`
的 `base_tools` + 写新提示词"这一整套（这正是本模块要替代的旧模式，见
`delegation_tools.py` 历史版本的 `build_web_search_delegate_tool()` 等函数）。

**能力声明化（本模块的核心改动）**：`SubagentProfile` 不再直接返回构造好的
`BaseTool`/`AgentMiddleware` 对象（旧版 `tools_factory`/`middleware_factory`），
只声明"依赖哪些能力"（`required_tools`/`optional_tools`/`required_skills`/
`allowed_skill_categories`）。原因：旧版直接返回对象时，一个 Profile 声明的
能力如果在运行时实际不可用（工具被下线、技能被卸载），没有任何环节会发现——
子 Agent 会带着一个不存在/失效的能力被造出来，模型可能凭自己已有知识伪装成
"确实执行过"的结果回答，这对 `web-researcher` 这类"必须真的执行了外部动作
才可信"的子 Agent 是个真实风险。真正的解析（查 `ToolRegistry`/`SkillRegistry`、
过 Guardrail、必需能力缺失时 fail fast）移到 `subagent_capability_resolver.py`，
在 `delegation_tools.py` 真正派发前统一做——Profile 本身只是声明，不参与解析。

`system_prompt_factory` 仍然设计成惰性求值（调用 `task` 工具时才执行），原因
和旧版一致：`web-researcher` 需要每次现算 `{current_date}`（避免长期运行的
进程把日期冻结在启动那一刻）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from langchain_core.tools import BaseTool

from src.agent_core.prompts import prompt_factory
from src.agent_core.tools.web_search_tool import web_search
from src.common.constants import SkillCategory

_WEB_RESEARCHER_DESCRIPTION = (
    "联网搜索专家：获取实时资讯、URL、在线数据等。返回的是中间搜索结果，不是最终答案，"
    "拿到结果后通常还需要你自己进一步处理。传入需要搜索的具体内容。"
)

_WEB_SEARCH_CATEGORY = SkillCategory.WEB_SEARCH.value

#: 故意不登记进 `ToolRegistry` 的 subagent 专属工具（`tools/registry/
#: providers.py` 的说明：`web_search` 只给 subagent 用，登记进共享注册表会让
#: 它意外出现在 Lead Agent 自己的工具列表里）。`subagent_capability_resolver.
#: py::resolve_subagent_capabilities()` 在 `ToolRegistry` 快照里查不到某个
#: 声明的工具名时，会回退查这张表。
SUBAGENT_ONLY_TOOLS: dict[str, BaseTool] = {
    "web_search": web_search,
}


def _web_researcher_system_prompt() -> str:
    """现算系统提示词：静态模板 + 当日日期替换。

    Skill 目录不再在这里手工拼接——挂载的 `SkillMiddleware`（由
    `resolve_subagent_capabilities()` 按 `allowed_skill_categories` 构造）会
    在每次模型调用前动态注入（预路由 + `<skill_catalog>`），跟 Lead Agent
    走同一条路径，避免这里维护一份重复、容易和中间件逻辑跑偏的手工拼装。
    """
    return prompt_factory.get("WEB_SEARCH_AGENT").replace("{current_date}", date.today().isoformat())


@dataclass(frozen=True)
class SubagentProfile:
    """一个可被 `task` 工具派发的专用子 Agent 的能力声明。

    只声明依赖，不持有实际对象——`subagent_capability_resolver.py::
    resolve_subagent_capabilities()` 在派发前统一解析成实际可用的
    Tool/Middleware，见模块 docstring"能力声明化"一节。

    Attributes:
        name: subagent 类型标识，即 `task(subagent_type=...)` 的取值。
        description: 展示在 `task` 工具描述里，供 Lead Agent 判断何时选用。
        system_prompt_factory: 子 Agent 系统提示词的惰性构建函数。
        required_tools: 必需的工具名集合（**不得包含沙箱执行类工具**，见
            `sub_agent_factory.py` 模块 docstring 的安全边界说明）。任意一个
            缺失或被 Guardrail 拒绝，`resolve_subagent_capabilities()` 会
            拒绝派发（`SubagentCapabilityUnavailable`），不创建子 Agent。
        optional_tools: 可选的工具名集合，缺失或被拒绝时降级执行（不加入
            最终工具列表，不阻止派发）。
        required_skills: 必需的技能名集合，只做存在性校验（技能名对应
            `SkillDefinition.name`）。任意一个不存在会拒绝派发，理由同
            `required_tools`。
        allowed_skill_categories: 这个子 Agent 允许加载的技能分类集合，
            原样传给 `SkillMiddleware` 构造参数；非空时才会挂载
            `SkillMiddleware`（见 `sub_agent_factory.py` 的安全边界
            说明——只允许挂载不引入沙箱执行类工具的中间件，`SkillMiddleware`
            是目前唯一的例外）。
    """

    name: str
    description: str
    system_prompt_factory: Callable[[], str]
    required_tools: frozenset[str] = field(default_factory=frozenset)
    optional_tools: frozenset[str] = field(default_factory=frozenset)
    required_skills: frozenset[str] = field(default_factory=frozenset)
    allowed_skill_categories: frozenset[str] = field(default_factory=frozenset)


_PROFILES: dict[str, SubagentProfile] = {
    "web-researcher": SubagentProfile(
        name="web-researcher",
        description=_WEB_RESEARCHER_DESCRIPTION,
        system_prompt_factory=_web_researcher_system_prompt,
        required_tools=frozenset({"web_search"}),
        allowed_skill_categories=frozenset({_WEB_SEARCH_CATEGORY}),
    ),
}


def get_profile(subagent_type: str) -> SubagentProfile | None:
    """按类型名查找已注册的 subagent profile，未注册时返回 None。"""
    return _PROFILES.get(subagent_type)


def list_subagent_types() -> list[str]:
    """返回当前已注册的全部 subagent 类型名，用于工具描述和错误提示。"""
    return list(_PROFILES.keys())
