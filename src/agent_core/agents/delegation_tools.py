"""通用 `task` 分发工具：把子任务派给 `subagent_profiles.py` 里注册的专用 subagent。

对应 DeerFlow 的 `task()` 工具（设计文档第二节表格里"作为二期演进方向"的那一条）。
取代早期版本里 `build_rag_delegate_tool()`/`build_web_search_delegate_tool()`/
`build_database_delegate_tool()` 三个各自固定工具名的 builder——rag/database 两个
能力已经改成直接挂在 Lead Agent 自己的工具集（`search_knowledge_base`/`query_database`
技能，见 `lead_agent.py`），只有仍然需要"子 Agent 做多轮搜索+提炼再回话"的 web_search
还留在这套委派机制里，通过 `subagent_type="web-researcher"` 访问。
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.agent_core.agents.sub_agent_factory import run_subagent
from src.agent_core.agents.subagent_profiles import get_profile, list_subagent_types


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

    async def _invoke(subagent_type: str, task: str, config: RunnableConfig) -> str:
        profile = get_profile(subagent_type)
        if profile is None:
            return f"未知的 subagent_type: {subagent_type!r}，可用类型：{', '.join(list_subagent_types())}"
        return await run_subagent(
            agent_name=profile.name,
            system_prompt=profile.system_prompt_factory(),
            tools=profile.tools_factory(),
            task=task,
            config=config,
        )

    return StructuredTool(
        name="task",
        description=_build_task_tool_description(),
        args_schema=_TaskInput,
        coroutine=_invoke,
    )
