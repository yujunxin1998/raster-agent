"""修复消息历史里不合法的 tool_calls/ToolMessage 配对/顺序状态（设计文档 4.2 节）。

原来是 `agent_core/agents/dangling_tool_calls.py` 里两个直接操作
`agent.aget_state`/`aupdate_state` 的普通函数，在 `chat_pipeline.py` 里手动
调用——这是从重构前的老架构（`chat_service.py::_sanitize_dangling_tool_calls`，
没有中间件概念的手写编排循环）原样迁移过来的历史遗留，迁移到中间件流水线后
没有跟着转换成中间件写法。现在改造成 `abefore_agent` 钩子：`abefore_agent` 在
整个 Agent 循环最开始、任何模型调用之前触发一次，纳入了标准中间件流水线，跟
`SummarizationMiddleware` 保持一致的写法。

三类问题，`_fix_message_order` 用同一遍扫描统一处理：

1. 悬空的 tool_calls（有调用，完全没有响应）：进程在工具执行过程中崩溃/重启，
   checkpointer 里持久化了"模型决定调用某个工具"但"工具执行结果从未写回"的
   半截状态。
2. 孤儿 ToolMessage（有响应，没有调用）：`MemoryCompressor.compress_messages()`
   在引入"避免切断 tool_calls/ToolMessage 配对"的修复之前，可能已经把发起
   某次工具调用的 `AIMessage` 压缩掉，但对应的 `ToolMessage` 留在了保留
   窗口里。
3. 错位的 ToolMessage（调用和响应都在，但响应没有紧跟在对应的 AIMessage
   后面）：例如工具调用发起后、结果落盘前，用户又发了一条新消息插在中间——
   这时"每个 tool_call_id 都有响应"这个检查会通过，但供应商 API 仍然会拒绝，
   因为它要求 ToolMessage 必须*紧跟*在发起调用的 AIMessage 后面，中间不能夹
   别的消息类型。只查"有没有响应"、不查"响应是否紧跟其后"的版本会漏掉这
   一类——这正是这个中间件曾经踩过的坑：第 1 类以为修好了（确实塞进去一个
   占位 `ToolMessage`），但顺序上还是被夹在了同一轮新插入的 `HumanMessage`
   之后，下一轮请求会继续触发同一个 400。

以上三类不管什么原因造成，最终都会触发同一类"assistant message with
tool_calls must be followed by tool messages"/"tool message must be a
response to a preceding tool_calls message"400 错误，永远卡住这个会话——所以
`_fix_message_order` 不区分成因，统一按"每个 AIMessage 的 tool_calls 后面必须
紧跟着对应的 ToolMessage（缺了补占位，位置不对就搬过来，没人认领就丢弃）"
这一条不变式重建整段历史。
"""
from __future__ import annotations

from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, RemoveMessage
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext

_DANGLING_TOOL_CALL_PLACEHOLDER = "[工具调用已中断，跳过]"


def _fix_message_order(messages: list[BaseMessage]) -> Optional[list[BaseMessage]]:
    """重建消息历史，让每个 AIMessage 的 tool_calls 后面都紧跟着对应的
    ToolMessage，返回重建后的完整消息列表；历史本身已经合法（顺序不变）时
    返回 None。

    不是"先判断有没有问题，有问题再修"——而是无条件重建一遍，末尾跟原始列表
    做逐位置比较来判断要不要返回 None。原因：只检查"每个 tool_call_id 是否
    存在对应 ToolMessage"（早期实现）查不出"响应存在但没有紧跟在 AIMessage
    后面"这一类（见模块文档第 3 类问题），必须实际按位置重排一遍才能确认
    顺序合法，判断和修复本来就是同一次扫描。

    也不能只返回"新增/删除的那几条"再指望 `add_messages` reducer 拼回去——
    `abefore_agent` 触发时本轮新的 `HumanMessage` 已经先合并进
    `state["messages"]`（模块文档有说明），而 `add_messages` 对不带已有 ID 的
    新消息只会追加到当前列表末尾，无法把占位/错位的 `ToolMessage` 插到中间
    正确的位置——必须重建整段顺序，配合 `REMOVE_ALL_MESSAGES` 整体替换（见
    `abefore_agent`）。
    """
    # 每个 tool_call_id 对应的响应只认第一个（正常情况下也只应该有一个）；
    # 后面出现的同 id 重复 ToolMessage 视为噪音，随原始位置一起被丢弃。
    responses_by_call_id: dict[str, ToolMessage] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id not in responses_by_call_id:
            responses_by_call_id[message.tool_call_id] = message

    fixed: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            # 有主的会在其 AIMessage 被处理到时一并放到紧随其后的正确位置
            # （不管它在原始列表里的位置在哪）；没有任何 AIMessage 认领的孤儿
            # ToolMessage 则不会被任何分支重新加入，等价于直接丢弃。
            continue
        fixed.append(message)
        if isinstance(message, AIMessage):
            for tool_call in (message.tool_calls or []):
                call_id = tool_call["id"]
                response = responses_by_call_id.get(call_id)
                fixed.append(
                    response if response is not None
                    else ToolMessage(content=_DANGLING_TOOL_CALL_PLACEHOLDER, tool_call_id=call_id)
                )

    if len(fixed) == len(messages) and all(a is b for a, b in zip(fixed, messages)):
        return None
    return fixed


class DanglingToolCallMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """每轮对话开始前修复不合法的 tool_calls/ToolMessage 配对/顺序状态。"""

    async def abefore_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """检查并修复本轮开始前的消息历史。

        Args:
            state: 当前 Agent 状态（本轮新消息已经合并进 `messages`，但重建
                逻辑只看历史里已有的 `AIMessage.tool_calls`/`ToolMessage`
                配对关系，跟本轮新加的消息无关，不影响判断结果）。
            runtime: 运行时上下文（本中间件不使用）。

        Returns:
            存在需要修复的消息时返回一次 `REMOVE_ALL_MESSAGES` + 重建后的完整
            消息列表（顺序已修复）；历史本身合法时返回 None。
        """
        messages = state.get("messages") or []
        if not messages:
            return None

        fixed = _fix_message_order(messages)
        if fixed is None:
            return None

        logger.warning(
            f"[DanglingToolCallMiddleware] 检测到 tool_calls/ToolMessage 配对问题，"
            f"已重建消息顺序（原 {len(messages)} 条 -> 修复后 {len(fixed)} 条）"
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *fixed]}
