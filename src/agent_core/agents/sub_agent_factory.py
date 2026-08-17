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

**安全边界**：子 Agent 默认不挂任何中间件（`GuardrailMiddleware`/
`LoopDetectionMiddleware` 都不在场），因此调用方传入的 `tools` 绝不能包含
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

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
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


async def run_subagent(
    *,
    agent_name: str,
    system_prompt: str,
    tools: list[BaseTool],
    task: str,
    context: AgentRuntimeContext,
    middleware: list[AgentMiddleware] | None = None,
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
        middleware: 可选的中间件列表，默认不挂任何中间件。见模块 docstring
            的安全边界说明——只允许挂载不引入沙箱执行类工具的中间件（如
            `SkillMiddleware`）。

    Returns:
        子 Agent 最终一条 `AIMessage` 的文本内容；子 Agent 未产生有效回复或执行
        异常时返回一段说明文本，不抛出异常。
    """
    sub_agent = create_agent(
        model=create_chat_model(), tools=tools, system_prompt=system_prompt, context_schema=AgentRuntimeContext,
        middleware=middleware or [],
    )
    timeout_seconds = get_settings().SUBAGENT_TIMEOUT_SECONDS
    try:
        result: dict[str, Any] = await asyncio.wait_for(
            sub_agent.ainvoke({"messages": [HumanMessage(content=task)]}, context=context),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(f"[{agent_name}] 子 Agent 执行超时 task={task!r} timeout={timeout_seconds}s")
        return _TIMEOUT_TEMPLATE.format(agent_name=agent_name, timeout=timeout_seconds)
    except Exception as exc:
        logger.error(f"[{agent_name}] 子 Agent 执行异常 task={task!r} error={exc}")
        return f"[{agent_name}] 执行失败: {exc}"

    final_text = _extract_final_text(result.get("messages", []))
    return final_text or _NO_REPLY_TEMPLATE.format(agent_name=agent_name)
