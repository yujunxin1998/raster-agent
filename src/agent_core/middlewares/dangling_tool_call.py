"""修复消息历史里两类不合法的 tool_calls/ToolMessage 配对状态（设计文档 4.2 节）。

原来是 `agent_core/agents/dangling_tool_calls.py` 里两个直接操作
`agent.aget_state`/`aupdate_state` 的普通函数，在 `chat_pipeline.py` 里手动
调用——这是从重构前的老架构（`chat_service.py::_sanitize_dangling_tool_calls`，
没有中间件概念的手写编排循环）原样迁移过来的历史遗留，迁移到中间件流水线后
没有跟着转换成中间件写法。现在改造成 `abefore_agent` 钩子：语义完全等价
（判断逻辑只看 `AIMessage.tool_calls`/`ToolMessage` 配对关系，跟本轮新加的
`HumanMessage` 无关；`abefore_agent` 在整个 Agent 循环最开始、任何模型调用
之前触发一次，修复效果跟"астream() 之前手动跑一遍"完全一样），但纳入了标准
中间件流水线，跟 `SummarizationMiddleware` 保持一致的写法。

两类问题：

1. 悬空的 tool_calls（有调用没响应）：进程在工具执行过程中崩溃/重启，
   checkpointer 里持久化了"模型决定调用某个工具"但"工具执行结果从未写回"的
   半截状态；下一轮请求把这段历史原样喂给模型时，大多数供应商的 API 会因为
   消息序列不合法（`INVALID_CHAT_HISTORY`）直接拒绝整个请求。
2. 孤儿 ToolMessage（有响应没调用）：`MemoryCompressor.compress_messages()`
   在引入"避免切断 tool_calls/ToolMessage 配对"的修复之前，可能已经把发起
   某次工具调用的 `AIMessage` 压缩掉，但对应的 `ToolMessage` 留在了保留
   窗口里；这类历史遗留的孤儿 `ToolMessage` 不会随着压缩逻辑修复自动消失，
   需要每轮开始前主动扫描清理，否则同一会话会反复触发
   `Messages with role 'tool' must be a response to a preceding message with
   'tool_calls'` 这个 400 错误，永远卡住。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.message import RemoveMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext

_DANGLING_TOOL_CALL_PLACEHOLDER = "[工具调用已中断，跳过]"


def _dangling_tool_call_fixes(messages: list) -> list:
    """扫描消息历史，返回悬空 tool_calls 需要补齐的占位 `ToolMessage` 列表。"""
    tool_call_ids_with_calls = {
        tool_call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in (message.tool_calls or [])
    }
    tool_call_ids_with_responses = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    dangling_ids = tool_call_ids_with_calls - tool_call_ids_with_responses
    if not dangling_ids:
        return []
    return [
        ToolMessage(content=_DANGLING_TOOL_CALL_PLACEHOLDER, tool_call_id=tool_call["id"])
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in (message.tool_calls or [])
        if tool_call["id"] in dangling_ids
    ]


def _orphaned_tool_message_fixes(messages: list) -> list:
    """扫描消息历史，返回孤儿 `ToolMessage`（无配对 tool_calls）需要删除的 `RemoveMessage` 列表。

    孤儿 `ToolMessage` 直接删除即可——它已经没有上下文（发起调用的 `AIMessage`
    已经不在了），留着也没有意义，只会继续触发模型供应商 API 的消息序列校验错误。
    """
    tool_call_ids_with_calls = {
        tool_call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in (message.tool_calls or [])
    }
    orphaned_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id not in tool_call_ids_with_calls
    ]
    return [RemoveMessage(id=message.id) for message in orphaned_messages]


class DanglingToolCallMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """每轮对话开始前修复两类不合法的 tool_calls/ToolMessage 配对状态。"""

    async def abefore_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """检查并修复本轮开始前的消息历史。

        Args:
            state: 当前 Agent 状态（本轮新消息已经合并进 `messages`，但这两类
                检查只看历史里已有的 `AIMessage.tool_calls`/`ToolMessage` 配对
                关系，跟本轮新加的消息无关，不影响判断结果）。
            runtime: 运行时上下文（本中间件不使用）。

        Returns:
            存在需要修复的消息时返回 state 更新（补齐的占位 `ToolMessage` +
            需要删除的 `RemoveMessage`）；历史本身合法时返回 None。
        """
        messages = state.get("messages") or []
        if not messages:
            return None

        fixes = [*_dangling_tool_call_fixes(messages), *_orphaned_tool_message_fixes(messages)]
        if not fixes:
            return None

        logger.warning(f"[DanglingToolCallMiddleware] 检测到 {len(fixes)} 处需要修复的 tool_calls/ToolMessage 配对问题")
        return {"messages": fixes}
