"""`datasource_id` 强制路由的硬校验中间件（设计文档 4.1 节）。

原项目里"配置了 datasource_id 必须先路由到 database_agent"这条规则完全靠
prompt 里的一段文案约束（`supervisor.md`"强制约束"一节），模型不遵守也没有
兜底。这里把它升级成代码层面的硬拦截：本轮第一次模型响应时，如果
`datasource_id` 已配置但模型的工具调用里没有 `query_database`，注入一条
纠正性 SystemMessage 强制重试一次；不管第二次结果如何都放行，避免死循环——
这是"软提示（prompt）+ 硬拦截（本中间件）"两层防线里的第二层。

`query_database` 曾经是委派工具 `delegate_to_database_agent`（把整个查询能力包在
一个子 Agent 里），后来短暂是 `skills/core/query-database/` 技能（`tool_name`
就是这个字面量），现在（`docs/Skill与Tool完全解耦重构设计.md` 第 10.1 节迁移后）
是 `database_query_tool.py` 里的独立 `@tool` 函数——本模块这里的字面量
`_DATABASE_DELEGATE_TOOL_NAME` 必须跟那个函数的 `@tool` 名字完全一致，否则这层
硬校验会静默失效（模型明明调用了 `query_database`，中间件却因为比对的还是旧
名字而认为"没调用"，一直重试）。指导"什么时候该查、怎么解读结果"的技能正文
现在挂在同分类下的 `skills/core/database-analysis/` WORKFLOW 技能里，跟这个
Tool 是名字不同、职责分离的两个东西。

不属于 `agent_core/middlewares/` 里那 11 个通用中间件——这是 Lead Agent 的
专属业务规则（知道 `query_database` 这个具体工具名），随 `lead_agent.py` 的
中间件列表追加，不进 `agent_core/loop.py::build_middlewares()`。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext

_DATABASE_DELEGATE_TOOL_NAME = "query_database"
_CORRECTION_MESSAGE = (
    f"检测到当前请求已配置数据源（datasource_id），本轮必须先调用 "
    f"{_DATABASE_DELEGATE_TOOL_NAME} 完成查询，不得跳过、不得先做其他事情。"
)


def _is_first_response_this_turn(messages: list) -> bool:
    """判断本次模型调用是不是本轮用户发言后的第一次响应。

    从后往前扫描：先遇到 HumanMessage 说明还没有产生过响应（是第一次）；
    先遇到 AIMessage 说明已经至少响应过一轮（不是第一次）。
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return True
        if isinstance(message, AIMessage):
            return False
    return True


def _has_database_delegate_call(result: list) -> bool:
    """判断模型这次响应的工具调用里是否包含 `query_database`。"""
    return any(
        isinstance(message, AIMessage)
        and any(tool_call["name"] == _DATABASE_DELEGATE_TOOL_NAME for tool_call in (message.tool_calls or []))
        for message in result
    )


class DatasourceRoutingMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """配置了 datasource_id 时，强制本轮第一次响应必须调用数据库委派工具。"""

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """必要时重试一次模型调用，注入纠正提示。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            模型调用结果（可能是重试后的结果）。
        """
        context = request.runtime.context
        datasource_id = context.datasource_id if context else None

        if not datasource_id or not _is_first_response_this_turn(request.messages):
            return await handler(request)

        response = await handler(request)
        if _has_database_delegate_call(response.result):
            return response

        logger.warning(
            f"[DatasourceRoutingMiddleware] 已配置 datasource_id={datasource_id} 但模型未调用 "
            f"{_DATABASE_DELEGATE_TOOL_NAME}，注入纠正提示并重试一次"
        )
        # Do not append a SystemMessage to the end of the history. Qwen/vLLM
        # chat templates require system messages to appear before all user and
        # assistant messages. Override the transient system prompt instead;
        # this keeps the checkpoint/history unchanged and lets the model
        # adapter place the prompt at the beginning of the request.
        corrected_prompt = (
            f"{request.system_prompt}\n\n{_CORRECTION_MESSAGE}"
            if request.system_prompt
            else _CORRECTION_MESSAGE
        )
        corrected_request = request.override(system_prompt=corrected_prompt)
        return await handler(corrected_request)
