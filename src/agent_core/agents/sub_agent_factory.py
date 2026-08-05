"""把一个"专精子 Agent"包装成 Lead Agent 可调用的委派工具。

对应设计文档 4.1 节：`rag_agent`/`web_search_agent` 这类有独立调优提示词、独立
工具边界的专用能力，不再是 Supervisor 图上的固定节点，而是重新封装成 Lead Agent
工具集里的一个 `delegate_to_xxx` 工具——工具被调用时才现造一个小的 `create_agent`
子 Agent（不带 checkpointer，无状态，用完即弃），取子 Agent 最后一条 AIMessage
的文本内容作为委派结果返回。

现造而不是缓存的原因与 `agent_core/model/model_factory.py` 一致：子 Agent 构造
成本不高，现造能保证每次调用都拿到最新的技能工具列表（`skill_manager.get_tools()`
本身就是"现取不缓存"的设计，见 `skill_manager.py` 的说明）。
"""
from __future__ import annotations

from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from src.agent_core.model import create_chat_model

_NO_REPLY_TEMPLATE = "[{tool_name}] 未产生有效回复"


class _DelegateInput(BaseModel):
    """委派工具的统一入参：把子任务用一句自然语言描述清楚。"""

    task: str = Field(description="交给该子 Agent 处理的具体子任务描述，需包含完成任务所需的全部上下文")


def _extract_final_text(messages: list) -> str | None:
    """从子 Agent 的最终消息列表里取最后一条有文本内容的 AIMessage。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return None


def build_delegate_tool(
    *,
    tool_name: str,
    description: str,
    system_prompt: str,
    tools_factory: Callable[[], list[BaseTool]],
) -> BaseTool:
    """构建一个委派工具：调用时现造子 Agent 并执行给定子任务。

    Args:
        tool_name: 暴露给 Lead Agent 的工具名，如 `delegate_to_rag_agent`。
        description: 工具描述，决定 Lead Agent 何时选用该委派工具。
        system_prompt: 子 Agent 的系统提示词。
        tools_factory: 子 Agent 工具集的构建函数，调用时才执行（技能工具依赖
            `SkillManager` 单例，只有应用完成启动后才可用，不能在模块导入阶段
            就解析）。

    Returns:
        对应的 StructuredTool 实例。
    """

    async def _invoke(task: str, config: RunnableConfig) -> str:
        sub_agent = create_agent(
            model=create_chat_model(),
            tools=tools_factory(),
            system_prompt=system_prompt,
        )
        # 只透传 configurable，不透传外层 config 的 callbacks——否则子 Agent 自己的
        # 模型/工具调用事件会经由回调传播机制泄漏进外层 astream_events 流，
        # 污染用户可见的 token 流。委派工具在外层看来应该是一次不透明的工具调用。
        isolated_config: RunnableConfig = {"configurable": dict(config.get("configurable") or {})}
        try:
            result: dict[str, Any] = await sub_agent.ainvoke(
                {"messages": [HumanMessage(content=task)]}, isolated_config
            )
        except Exception as exc:
            logger.error(f"[{tool_name}] 子 Agent 执行异常 task={task!r} error={exc}")
            return f"[{tool_name}] 执行失败: {exc}"

        final_text = _extract_final_text(result.get("messages", []))
        return final_text or _NO_REPLY_TEMPLATE.format(tool_name=tool_name)

    return StructuredTool(
        name=tool_name,
        description=description,
        args_schema=_DelegateInput,
        coroutine=_invoke,
    )
