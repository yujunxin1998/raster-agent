"""Lead Agent 专属：把 `plan` state 动态注入每次模型调用的 system_prompt，
并在模型即将结束本轮回复时硬校验计划是否已经收尾。

跟 `plan_tools.py::update_plan` 是同一个功能的两面：`update_plan` 负责写，
本中间件负责读。不进 `agent_core/loop.py::build_middlewares()`——`web-researcher`
等 subagent（`sub_agent_factory.py::run_subagent()` 现造的子 Agent）没有
`update_plan` 工具，自然也不会有 `plan` state，本中间件只在 `lead_agent.py`
里随 Lead Agent 专属中间件追加，与 `SkillMiddleware`/`DatasourceRoutingMiddleware`
同一挂载方式。

必须动态注入而不是指望消息历史里 `update_plan` 那条 `ToolMessage` 让模型"自己
记得"：长任务跑很多轮之后，早期消息可能被 `MemoryCompressor`/
`SummarizationMiddleware` 压缩掉，届时如果计划只存在于已经被压缩的历史消息里，
等于形同虚设——`plan` 是独立的 state 字段，不受消息历史压缩影响，每次模型调用
都能拿到最新值。

**收尾硬校验**：`plan_system.md` 里"最后一步完成后也要调用一次 update_plan"
只是软提示（prompt），模型仍然可能跳过——实测出现过计划卡在"最后一步进行中"
但模型已经直接把最终答案写出来了的情况（步骤实际已完成，只是状态没同步）。
纯靠 prompt 不是确定性保证，照抄 `datasource_routing_middleware.py` 的"软提示
+ 硬拦截"两层防线：`handler(request)` 拿到响应后检查——如果这是本轮最终回复
（没有任何 tool_calls，说明模型准备直接结束）且当前 `plan` 里还有步骤没标成
completed，就注入一条纠正提示强制重试一次，让模型先调用 `update_plan` 把状态
收尾，再回来给最终答案（重试会让图多跑一轮"调用 update_plan → 再问一次模型"，
但保证了"计划展示的完成度"和"用户实际拿到的结果"不会不一致）。跟
`DatasourceRoutingMiddleware` 一样，重试只做一次、不管第二次结果如何都放行，
避免模型持续不配合导致死循环。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.state import PipelineState

_PLAN_BLOCK_TAG = "current_plan"
_COMPLETED_STATUS = "completed"
_CORRECTION_MESSAGE = (
    "检测到当前计划（<current_plan>）里还有步骤没有标成 completed，但本轮准备"
    "不做任何工具调用直接结束。如果这些步骤事实上已经完成，请先调用 update_plan"
    "把计划更新为准确的最终状态（已完成的标 completed，因故未完成/被跳过的如实"
    "反映），再给出最终回复；不要跳过这一步直接作答。"
)


def _format_plan_block(plan: list[dict]) -> str:
    lines = "\n".join(f"- [{step['status']}] {step['content']}" for step in plan)
    return f"<{_PLAN_BLOCK_TAG}>\n{lines}\n</{_PLAN_BLOCK_TAG}>"


def _has_incomplete_step(plan: list[dict]) -> bool:
    return any(step.get("status") != _COMPLETED_STATUS for step in plan)


def _is_final_response(result: list) -> bool:
    """本轮响应是否不再产生任何工具调用（即模型准备直接结束、给出最终回复）。"""
    return not any(
        isinstance(message, AIMessage) and message.tool_calls for message in result
    )


class PlanContextMiddleware(AgentMiddleware[PipelineState, AgentRuntimeContext]):
    """把当前 `plan`（如果有）注入进模型调用的 system_prompt，并硬校验收尾状态。"""

    state_schema = PipelineState

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """注入 `<current_plan>` 块；模型即将结束本轮但计划未收尾时强制重试一次。

        Args:
            request: 本次模型调用请求。
            handler: 执行模型调用的回调。

        Returns:
            模型调用结果（可能是重试后的结果）。
        """
        plan: list[dict[str, Any]] = request.state.get("plan") or []
        if not plan:
            return await handler(request)

        block = _format_plan_block(plan)
        merged_prompt = f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
        plan_request = request.override(system_prompt=merged_prompt)

        response = await handler(plan_request)
        if not (_is_final_response(response.result) and _has_incomplete_step(plan)):
            return response

        logger.warning(
            "[PlanContextMiddleware] 模型准备结束本轮但计划仍有未完成步骤，注入纠正提示并重试一次"
        )
        corrected_request = plan_request.override(
            system_prompt=f"{merged_prompt}\n\n{_CORRECTION_MESSAGE}"
        )
        return await handler(corrected_request)
