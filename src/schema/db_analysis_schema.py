"""数据库查询分析结果的结构化输出模型。

原样迁移自 `diit-agent-server` 的 `src/schema/db_analysis.py`，供
`agent_core/tools/database_tool.py::DatabaseQueryService.generate_query_conditions_analysis`
的 `with_structured_output` 使用。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SQLAnalysisResult(BaseModel):
    """SQL 分析结果的结构化输出。"""

    data_source: str = Field(description="数据源名称")
    filter_conditions: str = Field(description="SQL 的筛选条件描述")
    calculation_operations: str = Field(description="SQL 的计算操作描述")
