"""按需现造一个"专精子 Agent"并执行一次性子任务——`task` 通用分发工具
（`delegation_tools.py::build_task_tool`）的执行内核。

对应设计文档 4.1 节 + DeerFlow 的 `SubagentExecutor`/`task()` 工具：`rag_agent`/
`web_search_agent` 这类有独立调优提示词、独立工具边界的专用能力，不是 Supervisor
图上的固定节点，也不是每个能力各自一个固定工具名（`delegate_to_xxx`，本模块早期
实现），而是统一通过 `run_subagent()` 现造一个小的 `create_agent` 子 Agent
（不带 checkpointer、不挂任何中间件，无状态，用完即弃），取子 Agent 最后一条
AIMessage 的文本内容作为结果返回。调用方（`task` 工具）只需要传入这次要用的
`system_prompt`/`tools`，不需要为每个专用能力单独写一个 builder 函数。

现造而不是缓存的原因与 `agent_core/model/model_factory.py` 一致：子 Agent 构造
成本不高，现造能保证每次调用都拿到最新的技能工具列表（`skill_manager.get_tools()`
本身就是"现取不缓存"的设计，见 `skill_manager.py` 的说明）。

**安全边界**：子 Agent 不挂任何中间件（`GuardrailMiddleware`/`LoopDetectionMiddleware`
都不在场），因此调用方传入的 `tools` 绝不能包含沙箱执行类工具
（`write_file`/`read_file`/`run_python`/`run_command`）——那几个工具必须留在 Lead
Agent 自己的 `base_tools` 里才能被中间件保护，塞进子 Agent 会让"写→跑"这类迭代循环
完全跑在死循环检测和权限校验之外，见 `lead_agent.py` 模块 docstring 的同一条说明。
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.model import create_chat_model

_NO_REPLY_TEMPLATE = "[{agent_name}] 未产生有效回复"


def _extract_final_text(messages: list) -> str | None:
    """从子 Agent 的最终消息列表里取最后一条有文本内容的 AIMessage。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return None


async def run_subagent(
    *,
    agent_name: str,
    system_prompt: str,
    tools: list[BaseTool],
    task: str,
    context: AgentRuntimeContext,
) -> str:
    """现造一个子 Agent，执行一次性子任务，返回其最终回复文本。

    Args:
        agent_name: 子 Agent 的标识名（用于日志和"未产生有效回复"兜底文案），
            对应 `SubagentProfile.name`，如 `"web-researcher"`。
        system_prompt: 子 Agent 的系统提示词。
        tools: 子 Agent 的工具集（见模块 docstring 的安全边界约束）。
        task: 交给子 Agent 处理的具体子任务描述。
        context: 外层 `task` 工具收到的 `runtime.context`，原样转发给子
            Agent——子 Agent 只接收 `context`，不接收外层的 `RunnableConfig`
            （不传 `config` 参数），因此天然不会带上外层的 `callbacks`：子
            Agent 自己的模型/工具调用事件不会经由回调传播机制泄漏进外层
            `astream_events` 流。委派在外层看来应该是一次不透明的工具调用。

    Returns:
        子 Agent 最终一条 `AIMessage` 的文本内容；子 Agent 未产生有效回复或执行
        异常时返回一段说明文本，不抛出异常。
    """
    sub_agent = create_agent(
        model=create_chat_model(), tools=tools, system_prompt=system_prompt, context_schema=AgentRuntimeContext,
    )
    try:
        result: dict[str, Any] = await sub_agent.ainvoke(
            {"messages": [HumanMessage(content=task)]}, context=context,
        )
    except Exception as exc:
        logger.error(f"[{agent_name}] 子 Agent 执行异常 task={task!r} error={exc}")
        return f"[{agent_name}] 执行失败: {exc}"

    final_text = _extract_final_text(result.get("messages", []))
    return final_text or _NO_REPLY_TEMPLATE.format(agent_name=agent_name)
