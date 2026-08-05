"""三个委派工具的具体定义：rag / web_search（database 本轮推迟，见包 docstring）。

每个 `build_xxx_delegate_tool()` 都是一个工厂函数，被 `lead_agent.py::build_lead_agent()`
每次组装 Lead Agent 时调用一次，不做模块级单例缓存——原因有二：
    1. 技能工具集本身就是"现取不缓存"的设计（`skill_manager.get_tools()`），
       缓存委派工具会让技能开关（`/skills/{tool_name}/toggle`）变更延迟生效。
    2. 原项目 `web_search_agent` 把 `{current_date}` 在模块导入时一次性替换成
       当天日期，长期运行的进程会在跨天后仍然使用旧日期——这里改成每次构建时
       现算，顺手修掉这个已知的小 bug。
"""
from __future__ import annotations

from datetime import date

from langchain_core.tools import BaseTool

from src.agent_core.agents.sub_agent_factory import build_delegate_tool
from src.agent_core.prompts import prompt_factory
from src.agent_core.skills import get_skill_manager
from src.agent_core.tools.database_tool import query_database
from src.agent_core.tools.web_search_tool import web_search

_RAG_TOOL_DESCRIPTION = "委派给内部知识库检索专家：查询内部文档、私有资料。传入需要查询的问题。"
_WEB_SEARCH_TOOL_DESCRIPTION = (
    "委派给联网搜索专家：获取实时资讯、URL、在线数据等。返回的是中间搜索结果，不是最终答案，"
    "拿到结果后通常还需要你自己进一步处理。传入需要搜索的具体内容。"
)
_DATABASE_TOOL_DESCRIPTION = (
    "委派给数据库查询专家：处理数据统计、筛选、排行、对比类问题，返回表格/图表/数据解读。"
    "需要当前请求已配置数据源（datasource_id），否则无法执行。传入需要查询的具体问题。"
)


def build_rag_delegate_tool() -> BaseTool:
    """构建 `delegate_to_rag_agent` 工具。"""
    return build_delegate_tool(
        tool_name="delegate_to_rag_agent",
        description=_RAG_TOOL_DESCRIPTION,
        system_prompt=prompt_factory.get("RAG_AGENT"),
        tools_factory=lambda: get_skill_manager().get_tools("rag"),
    )


def build_web_search_delegate_tool() -> BaseTool:
    """构建 `delegate_to_web_search_agent` 工具。"""
    system_prompt = prompt_factory.get("WEB_SEARCH_AGENT").replace("{current_date}", date.today().isoformat())
    return build_delegate_tool(
        tool_name="delegate_to_web_search_agent",
        description=_WEB_SEARCH_TOOL_DESCRIPTION,
        system_prompt=system_prompt,
        tools_factory=lambda: [web_search, *get_skill_manager().get_tools("web_search")],
    )


def build_database_delegate_tool() -> BaseTool:
    """构建 `delegate_to_database_agent` 工具。

    背后的 `query_database` 依赖外部 ask-db-service/ECharts 服务
    （`settings.DB_QUERY_API_URL`/`ECHART_API_URL`），当前开发环境大概率不可达
    ——这是已知限制，不影响工具本身的结构完整性；不可达时 `query_database`
    会返回错误说明文本而不是抛异常中断对话。
    """
    return build_delegate_tool(
        tool_name="delegate_to_database_agent",
        description=_DATABASE_TOOL_DESCRIPTION,
        system_prompt=prompt_factory.get("DATABASE_AGENT"),
        tools_factory=lambda: [query_database, *get_skill_manager().get_tools("database")],
    )
