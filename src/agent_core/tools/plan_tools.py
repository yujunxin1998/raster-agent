"""Lead Agent 的轻量任务规划工具：`update_plan`。

对应设计文档"多 Agent 编排的规划/分发/收集/决策/调整"讨论的落地版本——不做
独立的规划-执行编排循环（那基本是把项目已经废弃过的 Supervisor 多图路由换个
说法重新引入，见 `docs/新一代AgentLoop工程设计方案.md` 记录的"每轮强制一次
结构化路由决策 LLM 调用"真实故障史），而是给 Lead Agent 一块可读写的"计划
草稿纸"：模型在自己现有的 ReAct 循环里自愿调用这个工具记录/更新计划，不新增
任何强制的决策节点。`PlanContextMiddleware`（`agent_core/agents/plan_middleware.py`）
负责把这里写入的计划动态注入进每次模型调用的 system_prompt，两者配合才是完整
的"计划可读可写"闭环。

只挂在 Lead Agent 上（`tools/registry/providers.py::_BUILTIN_TOOLS`）——
`sub_agent_factory.py` 现造的一次性子 Agent（`web-researcher` 等）没有"计划"
这个概念，不需要也不应该拿到这个工具。

**前端动态展示**：`update_plan` 是普通 `@tool`，天然经由 `chat_pipeline.py`
现成的通用 `tool_call`/`tool_response` 事件推给前端（跟截图里 `task` 工具的
"执行过程"卡片走的是同一条通用链路，不需要给它另开一种 WS 事件类型）。为了
让前端能把它跟其它工具区分开、渲染成一个实时更新的清单 UI（而不是一坨原始
JSON 参数），返回的 `ToolMessage` 正文里额外带一个 `<plan>{json}</plan>`
标签——跟 `sandbox_tool.py::save_output_file` 给图片产物加 `<image>{json}</image>`
标签、前端 `useChatStore.js::_extractImageBlocks` 识别渲染成图片卡片，是完全
一样的既有约定。前端需要照着写一个 `_extractPlanBlocks`（或类似命名）：
从 `tool_response`/`update_plan` 事件的正文里剥出 `<plan>...</plan>`，
JSON.parse 出 `[{content, status}, ...]` 数组渲染成带状态图标的清单；每次模型
再调用 `update_plan`，都会带上当前完整计划重新触发一次同名事件，前端整体替换
渲染状态即可（不是增量 patch）。这一步前端渲染代码本身不在这个仓库里
（`diit-agent-web` 是独立仓库），这里只保证后端契约稳定、可被消费。
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field

from src.agent_core.middlewares.context import AgentRuntimeContext

_EMPTY_PLAN_SUMMARY = "(空计划)"
_PLAN_TAG = "plan"


class PlanStep(BaseModel):
    """计划里的一个步骤。"""

    content: str = Field(description="这一步要做什么，简短清楚的一句话")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending", description="当前状态：pending 未开始 / in_progress 进行中 / completed 已完成",
    )


@tool
async def update_plan(plan: list[PlanStep], runtime: ToolRuntime[AgentRuntimeContext]) -> Command:
    """记录或更新当前任务的执行计划。

    每次调用都必须传入完整的计划列表（不是增量修改）——不需要计划时传空列表
    清空。已完成的步骤保留在列表里、status 标成 completed，不要直接从列表里
    删除（需要能看到完整进度，不只是剩余待办）。
    """
    serialized = [step.model_dump() for step in plan]
    summary = "\n".join(f"- [{step.status}] {step.content}" for step in plan) or _EMPTY_PLAN_SUMMARY
    context = runtime.context
    logger.info(
        f"[PlanTool] update_plan conversation_id={context.conversation_id if context else None} "
        f"user_id={context.user_id if context else None} steps={len(plan)}\n{summary}"
    )
    plan_tag = json.dumps(serialized, ensure_ascii=False)
    content = f"计划已更新：\n{summary}\n<{_PLAN_TAG}>{plan_tag}</{_PLAN_TAG}>"
    return Command(
        update={
            "plan": serialized,
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id),
            ],
        },
    )
