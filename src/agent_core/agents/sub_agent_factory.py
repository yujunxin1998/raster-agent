"""按需现造一个"专精子 Agent"并执行一次性子任务——`task` 通用分发工具
（`delegation_tools.py::build_task_tool`）的执行内核。

对应设计文档 4.1 节 + DeerFlow 的 `SubagentExecutor`/`task()` 工具：`rag_agent`/
`web_search_agent` 这类有独立调优提示词、独立工具边界的专用能力，不是 Supervisor
图上的固定节点，也不是每个能力各自一个固定工具名（`delegate_to_xxx`，本模块早期
实现），而是统一通过 `run_subagent()` 现造一个小的 `create_agent` 子 Agent
（不带 checkpointer，只挂错误处理和循环检测的最小治理中间件，无状态，用完即弃），取子 Agent 最后一条
AIMessage 的文本内容作为结果返回。调用方（`task` 工具）只需要传入这次要用的
`system_prompt`/`tools`，不需要为每个专用能力单独写一个 builder 函数。

现造而不是缓存的原因与 `agent_core/model/model_factory.py` 一致：子 Agent 构造
成本不高，现造能保证每次调用都拿到最新的技能工具列表（`skill_manager.get_tools()`
本身就是"现取不缓存"的设计，见 `skill_manager.py` 的说明）。

**安全边界**：子 Agent 默认只挂 `ToolErrorHandlingMiddleware` 和
`LoopDetectionMiddleware`，不挂 `GuardrailMiddleware`，因此调用方传入的 `tools` 绝不能包含
沙箱执行类工具（`write_file`/`read_file`/`run_python`/`run_command`）——那
几个工具必须留在 Lead Agent 自己的 `base_tools` 里才能被中间件保护，塞进
子 Agent 会让"写→跑"这类迭代循环完全跑在死循环检测和权限校验之外，见
`lead_agent.py` 模块 docstring 的同一条说明。**唯一的例外**是可选的
`middleware` 参数（`docs/Skill与Tool完全解耦重构设计.md` 第 12 节）：需要
Skill 能力的子 Agent（如 `web-researcher`）可以显式挂载
`SkillMiddleware`——它不提供任何沙箱执行类工具，只提供只读的
`load_skill`/`read_skill_resource`，不会重新引入上述"写→跑绕过中间件"的
风险，因此不违反这条安全边界；仍然不允许挂载会引入沙箱工具或改变工具执行
语义的其它中间件。

具体哪些工具/技能允许出现在这里、要不要做权限校验，现在统一由
`subagent_capability_resolver.py::resolve_subagent_capabilities()` 在派发前
（`delegation_tools.py`）解析决定——本函数只管拿到已经解析好的 `tools`/
`middleware` 执行，不自己做能力可用性判断。

**超时**：单个子 Agent 执行受 `SUBAGENT_TIMEOUT_SECONDS`（默认值见
`config/settings.py`）限制，超时返回说明文本而不是无限挂起——子 Agent 自己没有
`recursion_limit`/
`LoopDetectionMiddleware` 兜底，理论上可能在自己的 ReAct 循环里卡住。并发上限
（`SUBAGENT_MAX_CONCURRENCY`）不在这里控制，由调用方
`delegation_tools.py::build_task_tool()` 用 `asyncio.Semaphore` 包住每次
`run_subagent()` 调用——两者分层：本函数只管"单次执行不能无限久"，并发数量是
调用方的职责。
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from loguru import logger

from src.agent_core.agents.delegation_protocol import SubagentResult
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from src.agent_core.model import create_chat_model
from src.config.settings import get_settings

_NO_REPLY_TEMPLATE = "[{agent_name}] 未产生有效回复"
_TIMEOUT_TEMPLATE = "[{agent_name}] 执行超时（超过 {timeout} 秒），已中止"


def _extract_final_text(messages: list) -> str | None:
    """从子 Agent 的最终消息列表里取最后一条有文本内容的 AIMessage。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content
    return None


async def run_subagent_result(
    *,
    agent_name: str,
    system_prompt: str,
    tools: list[BaseTool],
    task: str,
    context: AgentRuntimeContext,
    middleware: list[AgentMiddleware] | None = None,
) -> SubagentResult:
    """现造一个子 Agent，执行一次性子任务，返回结构化终态。

    子 Agent 只接收派生后的业务 ``context``，不接收外层 ``RunnableConfig``，
    因而不会把内部模型和工具事件泄漏到外层 token/event 流。
    """

    task_id = context.task_id or f"subtask_{uuid4().hex[:24]}"
    sub_agent = create_agent(
        model=create_chat_model(), tools=tools, system_prompt=system_prompt, context_schema=AgentRuntimeContext,
        middleware=[
            ToolErrorHandlingMiddleware(),
            LoopDetectionMiddleware(),
            *(middleware or []),
        ],
    )
    timeout_seconds = get_settings().SUBAGENT_TIMEOUT_SECONDS
    try:
        result: dict[str, Any] = await asyncio.wait_for(
            sub_agent.ainvoke({"messages": [HumanMessage(content=task)]}, context=context),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(f"[{agent_name}] 子 Agent 执行超时 task={task!r} timeout={timeout_seconds}s")
        return SubagentResult(
            task_id=task_id,
            agent_name=agent_name,
            status="timed_out",
            error_code="subagent_timeout",
            error_message=_TIMEOUT_TEMPLATE.format(agent_name=agent_name, timeout=timeout_seconds),
            retryable=True,
        )
    except Exception as exc:
        logger.error(f"[{agent_name}] 子 Agent 执行异常 task={task!r} error={exc}")
        return SubagentResult(
            task_id=task_id,
            agent_name=agent_name,
            status="failed",
            error_code="subagent_execution_failed",
            error_message=str(exc),
            retryable=False,
        )

    final_text = _extract_final_text(result.get("messages", []))
    if final_text:
        return SubagentResult(
            task_id=task_id, agent_name=agent_name, status="succeeded", output=final_text,
        )
    return SubagentResult(
        task_id=task_id,
        agent_name=agent_name,
        status="failed",
        error_code="subagent_no_reply",
        error_message=_NO_REPLY_TEMPLATE.format(agent_name=agent_name),
        retryable=False,
    )


async def run_subagent(
    *,
    agent_name: str,
    system_prompt: str,
    tools: list[BaseTool],
    task: str,
    context: AgentRuntimeContext,
    middleware: list[AgentMiddleware] | None = None,
) -> str:
    """兼容现有 `task` 工具的文本接口；内部执行使用结构化终态。"""

    result = await run_subagent_result(
        agent_name=agent_name,
        system_prompt=system_prompt,
        tools=tools,
        task=task,
        context=context,
        middleware=middleware,
    )
    return result.to_text()
