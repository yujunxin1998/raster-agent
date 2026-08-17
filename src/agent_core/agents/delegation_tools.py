"""通用 `task` 分发工具：把子任务派给 `subagent_profiles.py` 里注册的专用 subagent。

对应 DeerFlow 的 `task()` 工具（设计文档第二节表格里"作为二期演进方向"的那一条）。
取代早期版本里 `build_rag_delegate_tool()`/`build_web_search_delegate_tool()`/
`build_database_delegate_tool()` 三个各自固定工具名的 builder——rag/database 两个
能力已经改成直接挂在 Lead Agent 自己的工具集（`search_knowledge_base`/`query_database`
技能，见 `lead_agent.py`），只有仍然需要"子 Agent 做多轮搜索+提炼再回话"的 web_search
还留在这套委派机制里，通过 `subagent_type="web-researcher"` 访问。

不能加 `from __future__ import annotations`：`langchain_core.tools.structured.
StructuredTool._injected_args_keys` 用 `inspect.signature(fn)`（不带
`eval_str=True`）判断哪些参数需要注入，postponed evaluation 会让
`runtime: ToolRuntime[...]` 的标注在这里只是个没被求值的字符串，
`_is_injected_arg_type` 认不出来，`_parse_input` 就会把模型正确注入的
`runtime` 参数当成未声明字段丢弃，报
`_invoke() missing 1 required positional argument: 'runtime'`
（`@tool` 装饰器不受影响，因为它内部会重建一份带真实类型的 `__signature__`）。

**并发上限**：`build_task_tool()` 内部用一个 `asyncio.Semaphore(SUBAGENT_MAX_CONCURRENCY)`
包住每次 `run_subagent()` 调用——Semaphore 在 `build_task_tool()` 函数体内创建，
不是模块级单例，作用域天然是"这一次 `resolve_tools()`/Lead Agent 构建"（对应
一轮对话，见 `lead_agent.py` 模块文档"不做单例缓存，每次调用都现造"），不会让
不同会话互相排队等待。超过上限时排队等待，不是拒绝——`task` 工具本身没有
"部分失败"这个概念，模型也没必要为并发数操心。

**能力解析 + fail fast**：真正 `run_subagent()` 之前先调用
`subagent_capability_resolver.py::resolve_subagent_capabilities()`，把
Profile 声明的能力依赖解析成实际可用的 Tool/Middleware——必需能力缺失或被
Guardrail 拒绝时抛 `SubagentCapabilityUnavailable`，这里捕获后直接返回一段
说明文本给 Lead Agent，不创建子 Agent（不调用 `run_subagent()`）。可选能力
缺失时正常派发，降级信息记进 `SubagentRunMetadata` 日志。
"""

import asyncio

from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from src.agent_core.agents.sub_agent_factory import run_subagent
from src.agent_core.agents.subagent_capability_resolver import (
    SubagentCapabilityUnavailable,
    SubagentRunMetadata,
    resolve_subagent_capabilities,
)
from src.agent_core.agents.subagent_profiles import SUBAGENT_ONLY_TOOLS, get_profile, list_subagent_types
from src.agent_core.guardrail import get_guardrail_provider
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.skills import get_skill_manager
from src.config.settings import get_settings

# `get_tool_registry` 不能在模块顶层 import：`tools/registry/__init__.py` 会
# import `providers.py`，而 `providers.py` 需要 `build_task_tool`（本模块）来
# 登记 `subagent` 来源——顶层互相 import 会在 `tools.registry` 包初始化到一半
# 时炸出 ImportError（partially initialized module）。延迟到调用时才 import，
# 那时两个模块都已经完成初始化。


class _TaskInput(BaseModel):
    """`task` 工具的统一入参：选定 subagent 类型 + 一句自然语言描述子任务。"""

    subagent_type: str = Field(description="要调用的专用 subagent 类型，可选值见工具描述")
    task: str = Field(description="交给该 subagent 处理的具体子任务描述，需包含完成任务所需的全部上下文")


def _build_task_tool_description() -> str:
    lines = ["把子任务派给专用 subagent 执行。可用的 subagent_type："]
    for subagent_type in list_subagent_types():
        profile = get_profile(subagent_type)
        lines.append(f"- {profile.name}：{profile.description}")
    return "\n".join(lines)


def build_task_tool() -> BaseTool:
    """构建 `task` 工具：按 `subagent_type` 查表派发给对应的 subagent profile。"""

    max_concurrency = get_settings().SUBAGENT_MAX_CONCURRENCY
    concurrency_limit = asyncio.Semaphore(max_concurrency)
    # 纯诊断用的运行时计数器，不参与并发控制逻辑本身（`concurrency_limit`
    # 才是唯一的真实限流机制）——只是为了能从日志里直接看到"某一时刻真正
    # 同时在跑的 run_subagent() 数量有没有超过 max_concurrency"，而不用靠
    # 猜测/前端 loading 卡片数量倒推（那反映的是"已发起调用数"，不是"正在
    # 执行数"，两者本来就不该相等）。
    active_count = 0

    async def _invoke(subagent_type: str, task: str, runtime: ToolRuntime[AgentRuntimeContext]) -> str:
        nonlocal active_count
        profile = get_profile(subagent_type)
        if profile is None:
            return f"未知的 subagent_type: {subagent_type!r}，可用类型：{', '.join(list_subagent_types())}"
        waiting = concurrency_limit.locked()
        if waiting:
            logger.info(f"[task] 并发已达上限 {max_concurrency}，排队等待 subagent_type={subagent_type!r}")
        async with concurrency_limit:
            active_count += 1
            logger.info(f"[task] 开始执行 subagent_type={subagent_type!r} active={active_count}/{max_concurrency}")
            try:
                from src.agent_core.tools.registry import get_tool_registry

                try:
                    resolved = await resolve_subagent_capabilities(
                        profile=profile,
                        snapshot=get_tool_registry().current_snapshot(),
                        skill_manager=get_skill_manager(),
                        subagent_only_tools=SUBAGENT_ONLY_TOOLS,
                        guardrail_provider=get_guardrail_provider(),
                        user_id=runtime.context.user_id,
                        conversation_id=runtime.context.conversation_id,
                        thinking=runtime.context.thinking,
                    )
                except SubagentCapabilityUnavailable as exc:
                    logger.warning(
                        f"[task] subagent_type={exc.subagent_type!r} 能力缺失，拒绝派发: {exc.missing}"
                    )
                    return f"Sub-Agent {exc.subagent_type!r} 当前不可用，缺少必需能力：{', '.join(exc.missing)}"

                metadata = SubagentRunMetadata(
                    subagent_type=profile.name,
                    tool_registry_revision=resolved.tool_registry_revision,
                    skill_registry_revision=resolved.skill_registry_revision,
                    degraded_capabilities=resolved.missing_optional,
                )
                logger.info(f"[task] 能力解析完成 {metadata}")

                return await run_subagent(
                    agent_name=profile.name,
                    system_prompt=profile.system_prompt_factory(),
                    tools=resolved.tools,
                    task=task,
                    context=runtime.context,
                    middleware=resolved.middleware,
                )
            finally:
                active_count -= 1

    return StructuredTool(
        name="task",
        description=_build_task_tool_description(),
        args_schema=_TaskInput,
        coroutine=_invoke,
    )
