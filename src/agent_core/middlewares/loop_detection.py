"""死循环检测中间件（设计文档 4.2 节 #8）。

从旧 `supervisor.py` 里"同一 agent 连续被选中 3 次强制 FINISH"这类专属死循环
检测代码，升级为通用中间件：检测连续相同的工具调用（同名 + 同参数），超过
阈值时短路返回提示，而不是继续执行——可以顺带保护所有工具，不只是路由
这一种循环。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.state import PipelineState

_DEFAULT_THRESHOLD = 3  # 对应旧 supervisor 死循环判断的"连续 3 次"


def _tool_call_signature(tool_call: dict) -> str:
    """把一次工具调用（名称 + 参数）归一化为可比较的字符串签名。"""
    args = tool_call.get("args") or {}
    return f"{tool_call['name']}:{sorted(args.items())}"


def _consecutive_run_length(history: list[str], signature: str) -> int:
    """统计 `history` 末尾与 `signature` 连续相同的条数。"""
    count = 0
    for item in reversed(history):
        if item != signature:
            break
        count += 1
    return count


class LoopDetectionMiddleware(AgentMiddleware[PipelineState, AgentRuntimeContext]):
    """检测连续重复的工具调用，超过阈值时短路，不再实际执行。"""

    state_schema = PipelineState

    def __init__(self, threshold: int = _DEFAULT_THRESHOLD) -> None:
        """初始化中间件。

        Args:
            threshold: 连续相同调用达到该次数时短路（默认 3，对应旧
                supervisor 的死循环判断阈值）。
        """
        super().__init__()
        if threshold < 1:
            raise ValueError("threshold 必须 >= 1")
        self._threshold = threshold

    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        """检测本次调用是否与最近几次工具调用连续重复。

        Args:
            request: 工具调用请求，`request.state` 需含 `recent_tool_calls`
                （由本中间件自身维护，`PipelineState` 已声明该字段）。
            handler: 执行工具调用的回调。

        Returns:
            达到阈值时返回携带提示的 error `ToolMessage`（打包进 `Command`
            一并更新签名历史）；否则正常执行并更新签名历史。

        Note:
            返回的 `Command.update["recent_tool_calls"]` 只放本次这一条新签名
            （`[signature]`），不是"历史 + 本次"拼好的完整列表——同一步内 Lead
            Agent 可能并行发起多个工具调用，每个都会各自跑一次本方法；如果各自
            都基于同一份起始 `history` 拼出完整列表再整体写回，多个并行分支的
            写入会互相冲突（LangGraph 的默认 channel 一步只接受一次写入）。真正
            的拼接 + 窗口截断交给 `PipelineState` 上声明的 reducer
            （`state.py::_append_recent_tool_calls`）在合并阶段做，多个并行分支
            各自提交的单条签名可以安全叠加。
        """
        tool_call = request.tool_call
        signature = _tool_call_signature(tool_call)
        history: list[str] = list((request.state or {}).get("recent_tool_calls") or [])
        run_length = _consecutive_run_length(history, signature)

        if run_length + 1 >= self._threshold:
            logger.warning(f"[LoopDetectionMiddleware] 检测到连续重复调用 tool={tool_call['name']} 次数={run_length + 1}")
            message = ToolMessage(
                content=(
                    f"检测到已连续 {run_length + 1} 次以相同参数调用工具 [{tool_call['name']}]，"
                    "本次已跳过执行，请更换参数或采用其他方式推进。"
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
            return Command(update={"messages": [message], "recent_tool_calls": [signature]})

        result = await handler(request)

        if isinstance(result, Command):
            update: dict[str, Any] = dict(result.update or {})
            update["recent_tool_calls"] = [signature]
            return Command(graph=result.graph, update=update, resume=result.resume, goto=result.goto)

        return Command(update={"messages": [result], "recent_tool_calls": [signature]})
