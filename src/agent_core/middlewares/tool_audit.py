"""工具调用审计中间件（设计文档 4.2 节 #6）。

原项目 `EvalEventEmitter` 的 `eval:tool` 埋点这次没有随基础设施一起迁移过来
（属于编排层/评估体系的一部分），本中间件先用 loguru 记录起止时间、耗时、
状态，作为过渡期的替代实现；等真正的评估事件体系迁移过来后，只需要把
这里的日志调用换成事件发射调用，中间件的位置和职责不需要变。
"""
from __future__ import annotations

import time

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext


class ToolAuditMiddleware(AgentMiddleware[object, AgentRuntimeContext]):
    """记录每次工具调用的起止时间、耗时、结果状态。"""

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """在工具调用前后记录审计日志。

        Args:
            request: 工具调用请求。
            handler: 执行工具调用的回调。

        Returns:
            `handler(request)` 的原始结果（本中间件只做旁路记录，不修改结果）。
        """
        context = request.runtime.context
        tool_name = request.tool_call["name"]
        started_at = time.monotonic()

        logger.info(
            f"[ToolAudit] 开始 tool={tool_name} "
            f"user_id={context.user_id if context else None} "
            f"conversation_id={context.conversation_id if context else None}"
        )
        try:
            result = await handler(request)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            logger.info(f"[ToolAudit] 异常 tool={tool_name} duration_ms={duration_ms} error={exc}")
            raise

        duration_ms = int((time.monotonic() - started_at) * 1000)
        status = getattr(result, "status", "success") if isinstance(result, ToolMessage) else "success"
        logger.info(f"[ToolAudit] 结束 tool={tool_name} duration_ms={duration_ms} status={status}")
        return result
