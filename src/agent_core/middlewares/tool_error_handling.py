"""工具异常统一处理中间件（设计文档 4.2 节 #7）。

把现在分散在各处的"工具失败不中断对话"容错逻辑（`custom_tool_converter.py`
里的 `tool_execute`、`skill_content_reader.py::assemble` 的 try/except）收口
成一处通用中间件：任何工具调用抛出的异常都转换成 error `ToolMessage`，
而不是向上传播中断整轮对话，Agent 拿到错误信息后可以自行决定怎么继续。
"""
from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext

_TOOL_ERROR_MESSAGE_TEMPLATE = "[工具执行错误: {error}]"


class ToolErrorHandlingMiddleware(AgentMiddleware[object, AgentRuntimeContext]):
    """把工具调用抛出的异常统一转换为 error ToolMessage。"""

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """执行工具调用，捕获异常并转换成 error ToolMessage。

        Args:
            request: 工具调用请求。
            handler: 执行工具调用的回调。

        Returns:
            正常时返回 `handler(request)` 的结果；异常时返回携带错误说明的
            error `ToolMessage`。
        """
        tool_name = request.tool_call["name"]
        try:
            return await handler(request)
        except Exception as exc:
            logger.error(f"[ToolErrorHandlingMiddleware] 工具执行异常 tool={tool_name} error={exc}")
            return ToolMessage(
                content=_TOOL_ERROR_MESSAGE_TEMPLATE.format(error=exc),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
