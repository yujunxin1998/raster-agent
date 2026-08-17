"""`query_database` 独立 Tool 单元测试（迁移自技能脚本，逻辑镜像
`test_plan_tools.py`/`test_memory_tools.py` 直接调 `.coroutine()` 的写法）。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.database_query_tool import query_database
from src.agent_core.tools.database_tool import _DATASOURCE_NOT_CONFIGURED_MESSAGE

_RUNTIME_NO_DATASOURCE = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1"))
_RUNTIME_WITH_DATASOURCE = SimpleNamespace(
    context=AgentRuntimeContext(conversation_id="c1", datasource_id="42"),
)


async def test_empty_query_text_returns_error_without_calling_service() -> None:
    result = await query_database.coroutine(query_text="  ", runtime=_RUNTIME_NO_DATASOURCE)

    assert "不能为空" in result


async def test_missing_datasource_returns_not_configured_message() -> None:
    result = await query_database.coroutine(query_text="查一下销量", runtime=_RUNTIME_NO_DATASOURCE)

    assert result == _DATASOURCE_NOT_CONFIGURED_MESSAGE


async def test_single_row_result_skips_chart_generation() -> None:
    fake_service = SimpleNamespace(
        execute_step1=AsyncMock(return_value={
            "query_result": {"row_count": 1, "data": "| a |\n|---|\n| 1 |", "data_json": [{"a": 1}]},
        }),
        generate_query_conditions_analysis=AsyncMock(return_value="<step label='查询条件'>...</step>"),
        execute_step2=AsyncMock(side_effect=Exception("step2 down")),
        generate_data_interpretation=AsyncMock(return_value="只有一条记录。"),
        format_step2_output=lambda data: "",
    )

    with patch("src.agent_core.tools.database_query_tool._get_db_service", return_value=fake_service):
        result = await query_database.coroutine(query_text="查一下销量", runtime=_RUNTIME_WITH_DATASOURCE)

    assert "<analysis>" in result
    assert "只有一条记录" in result
    fake_service.execute_step1.assert_awaited_once()


async def test_multi_row_result_includes_chart_when_echart_succeeds() -> None:
    fake_service = SimpleNamespace(
        execute_step1=AsyncMock(return_value={
            "query_result": {"row_count": 3, "data": "| a |\n|---|\n| 1 |\n| 2 |\n| 3 |", "data_json": [{"a": 1}, {"a": 2}, {"a": 3}]},
        }),
        generate_query_conditions_analysis=AsyncMock(return_value="<step label='查询条件'>...</step>"),
        execute_step2=AsyncMock(return_value={"summary": {"summary": "共 3 条"}}),
        format_step2_output=lambda data: data.get("summary", {}).get("summary", ""),
        generate_chart_metadata=AsyncMock(return_value=SimpleNamespace(table_title="结果", chart_title="分布", chart_type="bar")),
        call_echart_api=AsyncMock(return_value={"type": "bar"}),
        format_multi_row_output=lambda **kwargs: f"## {kwargs['table_title']}\n<echart>...</echart>\n{kwargs['interpretation']}",
    )

    with patch("src.agent_core.tools.database_query_tool._get_db_service", return_value=fake_service):
        result = await query_database.coroutine(query_text="查一下销量趋势", runtime=_RUNTIME_WITH_DATASOURCE)

    assert "<echart>" in result
    assert "共 3 条" in result


async def test_step1_failure_returns_error_text_not_exception() -> None:
    fake_service = SimpleNamespace(execute_step1=AsyncMock(side_effect=Exception("API 超时")))

    with patch("src.agent_core.tools.database_query_tool._get_db_service", return_value=fake_service):
        result = await query_database.coroutine(query_text="查一下销量", runtime=_RUNTIME_WITH_DATASOURCE)

    assert "数据库查询失败" in result
