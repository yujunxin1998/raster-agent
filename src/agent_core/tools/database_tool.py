"""数据库自然语言查询：调用外部 ask-db-service 完成查询，多行结果时附加 ECharts 图表。

原样迁移自 `diit-agent-server` 的 `src/service/db_query_service.py` +
`src/core/tools/database_tool.py`（原项目拆成两个文件，本仓库参照
`web_search_tool.py`"一个薄 `@tool` 函数 + 一个客户端类同放一个文件"的既有
约定合并成一个文件）。核心改动只有两处：`LLMFactory.get_llm(...)` 换成
`create_chat_model(...)`；`prompts.XXX.format_map(...)` 换成
`prompt_factory.render("XXX", ...)`——`DB_ANALYSIS`/`DB_ANALYSIS_SYSTEM`/
`DATA_INTERPRETATION`/`DATA_INTERPRETATION_SYSTEM`/`CHART_METADATA_GENERATION`/
`CHART_METADATA_SYSTEM` 六条提示词已经在第一期随 prompts 模块迁移过来，
未改动内容。
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from loguru import logger

from src.agent_core.model import create_chat_model
from src.agent_core.prompts import prompt_factory
from src.config.settings import get_settings
from src.schema.chart_schema import ChartMetadata
from src.schema.db_analysis_schema import SQLAnalysisResult

_DATASOURCE_NOT_CONFIGURED_MESSAGE = "当前请求未配置数据源（datasource_id），无法执行数据库查询。"

_CHART_TYPE_NAME_MAP = {"bar": "柱状", "line": "折线", "pie": "饼"}


def _structured_output_method(provider: str) -> str:
    """结构化输出方式：DeepSeek 下 `bind_tools` 会剥离部分场景的 `tool_choice`
    （见 `agent_core/model/patched_deepseek.py` 的说明），`with_structured_output`
    默认的 `function_calling` 不可靠——模型可能直接输出纯文本、悄悄跳过"格式化
    工具"，此时结构化解析静默失败，只能靠 fallback 兜底，质量打折但不报错，
    不容易被发现。改用 `json_mode` 强制 API 层只能返回合法 JSON。
    """
    return "json_mode" if provider == "deepseek" else "function_calling"


class DatabaseQueryService:
    """负责调用外部两阶段数据库查询 API，并生成解读/图表元数据。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_base_url = settings.DB_QUERY_API_URL
        self._timeout = settings.DB_QUERY_API_TIMEOUT
        self._echart_api_url = settings.ECHART_API_URL
        self._echart_timeout = settings.ECHART_API_TIMEOUT
        self._provider = settings.PROVIDER
        self._structured_method = _structured_output_method(settings.PROVIDER)

    async def execute_step1(self, query_text: str, datasource_id: int, work_flow_run_id: str) -> dict[str, Any]:
        """执行 Step1：生成 SQL 并执行查询，返回 query_result（含 data/data_json/row_count）。"""
        request_payload = {
            "work_flow_run_id": work_flow_run_id, "query_text": query_text, "datasource_id": datasource_id,
        }
        logger.info(f"[db_query] step1 starting | url={self._api_base_url}/step1 | payload={request_payload}")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._api_base_url}/step1", json=request_payload)
                if response.status_code >= 400:
                    logger.error(f"[db_query] step1 HTTP {response.status_code} | body={response.text[:1000]}")
                response.raise_for_status()
                result = response.json()
                if result.get("code") != 200:
                    raise Exception(f"Step1 API 返回错误: {result.get('message')}")
                data = result.get("data", {})
                logger.info(
                    f"[db_query] step1 completed | row_count={data.get('query_result', {}).get('row_count', 0)}"
                )
                return data
        except httpx.HTTPError as exc:
            logger.error(f"[db_query] step1 HTTP error: {exc}")
            raise Exception(f"数据库查询失败: {exc}") from exc

    async def execute_step2(self, work_flow_run_id: str) -> dict[str, Any]:
        """执行 Step2：基于 Step1 结果生成摘要。"""
        logger.info(f"[db_query] step2 starting | work_flow_run_id={work_flow_run_id}")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._api_base_url}/step2/{work_flow_run_id}", json={})
                if response.status_code >= 400:
                    logger.error(f"[db_query] step2 HTTP {response.status_code} | body={response.text[:1000]}")
                response.raise_for_status()
                result = response.json()
                if result.get("code") != 200:
                    raise Exception(f"Step2 API 返回错误: {result.get('message')}")
                return result.get("data", {})
        except httpx.HTTPError as exc:
            logger.error(f"[db_query] step2 HTTP error: {exc}")
            raise Exception(f"摘要生成失败: {exc}") from exc

    def format_step2_output(self, step2_data: dict[str, Any]) -> str:
        """格式化 Step2 输出（API 已返回格式化好的摘要文本）。"""
        summary_text = step2_data.get("summary", {}).get("summary", "")
        if not summary_text:
            logger.warning("[db_query] step2 summary is empty")
            return "## 查询摘要：\n*（摘要生成失败）*"
        return summary_text

    async def generate_query_conditions_analysis(self, query_text: str, step1_data: dict[str, Any]) -> str:
        """使用 LLM 生成查询条件分析，返回 `<step label='查询条件'>...</step>` XML 片段。"""
        generated_sql = step1_data.get("generated_sql", "")
        datasource_name = step1_data.get("datasource_name", "未知数据源")

        prompt = prompt_factory.render(
            "DB_ANALYSIS", query=query_text, datasource_name=datasource_name, generated_sql=generated_sql,
        )
        llm = create_chat_model(temperature=0.3, max_tokens=500, timeout=10)
        structured_llm = llm.with_structured_output(SQLAnalysisResult, method=self._structured_method)
        messages = [SystemMessage(content=prompt_factory.get("DB_ANALYSIS_SYSTEM")), HumanMessage(content=prompt)]

        try:
            result: SQLAnalysisResult = await structured_llm.ainvoke(messages)
            return self._format_conditions_step(result)
        except Exception as exc:
            logger.error(f"[db_query] conditions analysis failed: {exc}")
            return self._fallback_conditions_step(datasource_name)

    async def generate_data_interpretation(self, query_text: str, step1_data: dict[str, Any]) -> str:
        """使用 LLM 对查询结果进行解读，返回普通文本。"""
        query_result = step1_data.get("query_result", {})
        data_json = query_result.get("data_json", [])
        row_count = query_result.get("row_count", 0)
        simplified_data = data_json[:10] if isinstance(data_json, list) else []

        prompt = prompt_factory.render(
            "DATA_INTERPRETATION", query=query_text, row_count=row_count, data_sample=simplified_data,
        )
        llm = create_chat_model(temperature=0.7, max_tokens=300, timeout=10)
        messages = [SystemMessage(content=prompt_factory.get("DATA_INTERPRETATION_SYSTEM")), HumanMessage(content=prompt)]

        try:
            response = await llm.ainvoke(messages)
            content = response.content
            return content.strip() if isinstance(content, str) else str(content)
        except Exception as exc:
            logger.error(f"[db_query] interpretation generation failed: {exc}")
            return f"数据解读生成失败，请查看上述查询结果。共 {row_count} 条记录。"

    def _format_conditions_step(self, result: SQLAnalysisResult) -> str:
        return "\n".join([
            "<step label='查询条件'>",
            f"- 数据来源: {result.data_source}",
            f"- 筛选条件: {result.filter_conditions}",
            f"- 计算操作: {result.calculation_operations}",
            "</step>",
        ])

    def _fallback_conditions_step(self, datasource_name: str) -> str:
        return "\n".join([
            "<step label='查询条件'>",
            f"- 数据来源: {datasource_name}",
            "- 筛选条件: (分析失败，请查看SQL语句)",
            "- 计算操作: (分析失败，请查看SQL语句)",
            "</step>",
        ])

    async def generate_chart_metadata(self, query_text: str, step1_data: dict[str, Any]) -> ChartMetadata:
        """使用 LLM 生成图表元数据（表格标题/图表标题/图表类型）。"""
        query_result = step1_data.get("query_result", {})
        data_json = query_result.get("data_json", [])
        row_count = query_result.get("row_count", 0)
        simplified_data = data_json[:5] if isinstance(data_json, list) else []

        prompt = prompt_factory.render(
            "CHART_METADATA_GENERATION", query=query_text, row_count=row_count, data_sample=simplified_data,
        )
        llm = create_chat_model(temperature=0.3, max_tokens=200, timeout=10)
        structured_llm = llm.with_structured_output(ChartMetadata, method=self._structured_method)
        messages = [SystemMessage(content=prompt_factory.get("CHART_METADATA_SYSTEM")), HumanMessage(content=prompt)]

        result: ChartMetadata | None = await structured_llm.ainvoke(messages)
        if result is None:
            raise ValueError("结构化输出解析失败，LLM 未返回有效的 ChartMetadata")
        return result

    async def call_echart_api(self, data_json: list[dict], chart_type: str, workflow_id: str) -> dict[str, Any]:
        """调用 ECharts API 生成图表配置。"""
        chart_name = _CHART_TYPE_NAME_MAP.get(chart_type, "柱状")
        content = f"{data_json} 请根据上述数据帮我生成一个{chart_name}图"

        try:
            async with httpx.AsyncClient(timeout=self._echart_timeout) as client:
                response = await client.post(
                    self._echart_api_url, json={"content": content, "workflow_run_id": workflow_id},
                )
                if response.status_code >= 400:
                    logger.error(f"[db_query] echart api HTTP {response.status_code} | body={response.text[:1000]}")
                response.raise_for_status()
                result = response.json()
                echart_config = result.get("data", {}).get("echart_config", {})
                if not echart_config:
                    raise Exception("EChart API返回空配置")
                return echart_config
        except httpx.HTTPError as exc:
            logger.error(f"[db_query] echart api HTTP error: {exc}")
            raise Exception(f"EChart生成失败: {exc}") from exc

    def format_multi_row_output(
        self, table_title: str, table_markdown: str, chart_title: str, chart_type: str,
        echart_config: dict[str, Any], interpretation: str,
    ) -> str:
        """格式化多行数据的完整输出（表格 + 图表 + 解读）。"""
        echart_json = json.dumps(echart_config, ensure_ascii=False, separators=(",", ":"))
        return (
            f"## {table_title}\n{table_markdown}\n\n"
            f"## {chart_title}\n<echart>\n<{chart_type}>\n{echart_json}\n</{chart_type}>\n</echart>\n\n"
            f"{interpretation}"
        )


_db_service: DatabaseQueryService | None = None


def _get_db_service() -> DatabaseQueryService:
    """延迟初始化模块级单例，避免未配置 `DB_QUERY_API_URL` 时在导入阶段就报错。"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseQueryService()
    return _db_service


@tool
async def query_database(query_text: str, config: RunnableConfig) -> str:
    """查询数据库并生成结构化结果（表格、图表、数据解读），适用于明确的数据统计、筛选、对比、排行类问题。

    需要当前请求已配置数据源（datasource_id），否则无法执行。
    """
    datasource_id = config.get("configurable", {}).get("datasource_id")
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
