"""修复"悬空"工具调用：checkpoint 里存在 AIMessage.tool_calls 但没有对应 ToolMessage。

对应原项目 `chat_service.py::_sanitize_dangling_tool_calls`。触发场景：进程在
工具执行过程中崩溃/重启，checkpointer 里持久化了"模型决定调用某个工具"但
"工具执行结果从未写回"的半截状态；下一轮请求把这段历史原样喂给模型时，
大多数供应商的 API 会因为消息序列不合法（`INVALID_CHAT_HISTORY`）直接拒绝
整个请求。每次真正开始新一轮对话前都应该跑一次这个修复。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from loguru import logger

_DANGLING_TOOL_CALL_PLACEHOLDER = "[工具调用已中断，跳过]"


async def sanitize_dangling_tool_calls(agent: Any, config: dict) -> None:
    """扫描并修复 checkpoint 里悬空的工具调用。

    Args:
        agent: 已编译的 Lead Agent（`CompiledStateGraph`），需支持
            `aget_state`/`aupdate_state`。
        config: 本次调用的 `RunnableConfig`，须含 `configurable.thread_id`。
    """
    state = await agent.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    if not messages:
        return

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
        return

    fix_messages = [
        ToolMessage(content=_DANGLING_TOOL_CALL_PLACEHOLDER, tool_call_id=tool_call["id"])
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in (message.tool_calls or [])
        if tool_call["id"] in dangling_ids
    ]
    logger.warning(f"[dangling_tool_calls] 检测到 {len(fix_messages)} 个悬空工具调用，已补齐占位响应")
    await agent.aupdate_state(config, {"messages": fix_messages})
