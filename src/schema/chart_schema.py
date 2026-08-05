"""图表元数据的结构化输出模型。

原样迁移自 `diit-agent-server` 的 `src/schema/chart.py`，供
`agent_core/tools/database_tool.py::DatabaseQueryService.generate_chart_metadata`
的 `with_structured_output` 使用。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChartMetadata(BaseModel):
    """图表元数据的结构化输出。"""

    table_title: str = Field(description="表格标题，简洁描述数据内容")
    chart_title: str = Field(description="图表标题，突出数据特征")
    chart_type: Literal["bar", "line", "pie"] = Field(
        description="图表类型: bar(柱状图), line(折线图), pie(饼图)"
    )
