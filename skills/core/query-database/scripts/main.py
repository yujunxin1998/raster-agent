#!/usr/bin/env python3
"""query_database 技能主脚本 —— 由 SkillContentReader 作为子进程执行。

从 stdin 读一个 JSON params 对象，调用外部两阶段数据库查询 API（+ ECharts
图表生成），把结构化结果写到 stdout。输出格式：

    {"content": "<analysis>...</analysis>\n\n...", "__metadata__": {}}

编排逻辑原样对应 `src/agent_core/tools/database_tool.py` 里原 `query_database`
`@tool` 函数的函数体，只是两处取值方式变了：`datasource_id`/`query_text` 从
`params` 读（技能脚本没有 `RunnableConfig`），不再从 `config["configurable"]` 读。
`DatabaseQueryService` 本身完全复用，未改动。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that "src.*" imports work when
# this script is executed as a subprocess from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(  # raster-agent/
    os.path.dirname(              # skills/
        os.path.dirname(          # core/
            os.path.dirname(      # query-database/
                _SCRIPT_DIR       # scripts/
            )
        )
    )
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger  # noqa: E402

from src.agent_core.tools.database_tool import (  # noqa: E402
    _DATASOURCE_NOT_CONFIGURED_MESSAGE,
    _get_db_service,
    ChartMetadata,
)


async def _run(params: dict) -> dict:
    query_text: str = (params.get("query_text") or "").strip()
    if not query_text:
        return {"content": "query_text 不能为空。", "__metadata__": {}}

    datasource_id = params.get("datasource_id")
    if not datasource_id:
        return {"content": _DATASOURCE_NOT_CONFIGURED_MESSAGE, "__metadata__": {}}

    try:
        datasource_id = int(datasource_id)
    except (TypeError, ValueError):
        return {"content": f"datasource_id 格式不合法: {datasource_id!r}，无法执行数据库查询。", "__metadata__": {}}

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
            return {"content": f"{analysis_block}\n\n{data_markdown}\n\n{interpretation}", "__metadata__": {}}

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

        return {"content": f"{analysis_block}\n\n{content}", "__metadata__": {}}

    except Exception as exc:
        logger.error(f"[query_database] failed: {exc}")
        return {"content": f"数据库查询失败: {exc}", "__metadata__": {}}


def main() -> None:
    raw = sys.stdin.read()
    try:
        params = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        error_out = {"content": f"[参数解析失败: {exc}]", "__metadata__": {}}
        print(json.dumps(error_out, ensure_ascii=False))
        sys.exit(1)

    result = asyncio.run(_run(params))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
