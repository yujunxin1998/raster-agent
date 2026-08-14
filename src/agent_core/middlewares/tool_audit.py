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
from src.agent_core.tools.registry.tool_registry import get_tool_registry


def _resolve_source_type(tool_name: str) -> str | None:
    """按当前快照查该工具的来源类型，仅用于日志展示（工具注册中心设计文档
    4.4 节："排障时要注意失败原因可能来自远程 MCP Server，`ToolAuditMiddleware`
    记录时应该把 source_type=mcp 带上，方便一眼区分"）。

    ToolRegistry 未初始化（如中间件单元测试直接构造实例，不经过完整
    `main.py` lifespan）时静默返回 None，不影响审计日志本身的记录。
    """
    try:
        snapshot = get_tool_registry().current_snapshot()
    except RuntimeError:
        return None
    definition = snapshot.by_model_name.get(tool_name)
    return definition.source_type if definition else None


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
        source_type = _resolve_source_type(tool_name)
        registry_revision = context.registry_revision if context else None
        started_at = time.monotonic()

        logger.info(
            f"[ToolAudit] 开始 tool={tool_name} source_type={source_type} "
            f"registry_revision={registry_revision} "
            f"user_id={context.user_id if context else None} "
            f"conversation_id={context.conversation_id if context else None}"
        )
        try:
            result = await handler(request)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            logger.info(
                f"[ToolAudit] 异常 tool={tool_name} source_type={source_type} "
                f"duration_ms={duration_ms} error={exc}"
            )
            raise

        duration_ms = int((time.monotonic() - started_at) * 1000)
        status = getattr(result, "status", "success") if isinstance(result, ToolMessage) else "success"
        logger.info(
            f"[ToolAudit] 结束 tool={tool_name} source_type={source_type} "
            f"duration_ms={duration_ms} status={status}"
        )
        return result
