"""`query_database` 独立业务 Tool。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 10.1 节：把原来
`skills/core/query-database/scripts/main.py`（技能脚本，经
`SkillContentReader.run_script()` 子进程执行）里的编排逻辑迁移成一个直接
挂在 Lead Agent 工具集上的 `@tool` 函数。`DatabaseQueryService`
（`database_tool.py`）本身不改动，只是不再经"技能脚本子进程"这层间接调用，
而是被这里直接 `await`——两阶段查询 API + ECharts 图表生成的编排逻辑与原脚本
逐行对应。

`datasource_id` 原来通过 SKILL.md frontmatter 的 `runtime_context_keys`
声明、由 `SkillToolFactory._invoke` 从 `AgentRuntimeContext` 取值后塞进
`params` 字典；现在直接从 `runtime.context.datasource_id` 读取，语义不变。

技能正文（`skills/core/query-database/SKILL.md`）改造为纯指令型
`required_tools: [query_database]`，只保留"什么时候该查、怎么解读结果"这类
指导性文字，不再声明 `parameters`/`runtime_context_keys`——那两个字段是给
已废弃的 `SkillToolFactory` 用的（见 `skill_definition.py` 模块文档"迁移
状态"），迁移完成后这个技能不再需要它们。
"""
from __future__ import annotations

import uuid

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.database_tool import _DATASOURCE_NOT_CONFIGURED_MESSAGE, ChartMetadata, _get_db_service

_QUERY_TEXT_EMPTY_MESSAGE = "query_text 不能为空。"


@tool
async def query_database(query_text: str, runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    """处理数据统计、筛选、排行、对比、占比、图表类问题（明确指向数据库数据），
    返回表格/图表/数据解读。需要当前请求已配置数据源（datasource_id），否则无法执行。

    Args:
        query_text: 需要查询的具体问题。
        runtime: 注入的运行时，取 `datasource_id`。
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return _QUERY_TEXT_EMPTY_MESSAGE

    context = runtime.context
    datasource_id = context.datasource_id if context else None
    if not datasource_id:
        return _DATASOURCE_NOT_CONFIGURED_MESSAGE

    try:
        datasource_id = int(datasource_id)
    except (TypeError, ValueError):
        return f"datasource_id 格式不合法: {datasource_id!r}，无法执行数据库查询。"

    db_service = _get_db_service()
    work_flow_run_id = str(uuid.uuid4())

    try:
        step1_data = await db_service.execute_step1(query_text, datasource_id, work_flow_run_id)
        query_result = step1_data.get("query_result", {})
        row_count = query_result.get("row_count", 0)
        data_markdown = query_result.get("data", "")
        data_json = query_result.get("data_json", [])

        conditions_xml = await db_service.generate_query_conditions_analysis(query_text, step1_data)
        analysis_block = (
            f"<analysis>\n{conditions_xml}\n"
            f"<step label='数据解读'>\n基于\"{query_text}\"的返回结果，进行数据解读。\n</step>\n"
            "</analysis>"
        )

        try:
            step2_data = await db_service.execute_step2(work_flow_run_id)
            interpretation = db_service.format_step2_output(step2_data)
        except Exception as exc:
            logger.error(f"[query_database] step2 failed, falling back to local interpretation: {exc}")
            interpretation = await db_service.generate_data_interpretation(query_text, step1_data)

        if row_count <= 1:
            return f"{analysis_block}\n\n{data_markdown}\n\n{interpretation}"

        try:
            metadata = await db_service.generate_chart_metadata(query_text, step1_data)
        except Exception as exc:
            logger.error(f"[query_database] chart metadata generation failed, using fallback: {exc}")
            metadata = ChartMetadata(table_title="查询结果", chart_title="数据分布", chart_type="bar")

        try:
            echart_config = await db_service.call_echart_api(data_json, metadata.chart_type, work_flow_run_id)
        except Exception as exc:
            logger.error(f"[query_database] echart generation failed, skipping chart: {exc}")
            echart_config = None

        if echart_config:
            content = db_service.format_multi_row_output(
                table_title=metadata.table_title, table_markdown=data_markdown,
                chart_title=metadata.chart_title, chart_type=metadata.chart_type,
                echart_config=echart_config, interpretation=interpretation,
            )
        else:
            content = f"### {metadata.table_title}\n\n{data_markdown}\n\n{interpretation}"

        return f"{analysis_block}\n\n{content}"

    except Exception as exc:
        logger.error(f"[query_database] failed: {exc}")
        return f"数据库查询失败: {exc}"
