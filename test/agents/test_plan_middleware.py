"""`PlanContextMiddleware` 单元测试：把 `plan` state 动态注入 system_prompt。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.agent_core.agents.plan_middleware import PlanContextMiddleware
from src.agent_core.middlewares.context import AgentRuntimeContext


def _request(*, state: dict, system_prompt: str = "你是 Lead Agent") -> ModelRequest:
    return ModelRequest(
        model=None, system_prompt=system_prompt, messages=[], tool_choice=None,
        tools=[], response_format=None, state=state,
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1")),
    )


async def test_injects_current_plan_block_when_plan_exists() -> None:
    middleware = PlanContextMiddleware()
    request = _request(state={"plan": [
        {"content": "搜索 AWS 信息", "status": "in_progress"},
        {"content": "搜索 Azure 信息", "status": "pending"},
    ]})
    # 响应里带一个工具调用（不是最终回复）：只验证注入逻辑本身，不触发下面
    # 专门测的"收尾硬校验"重试分支——否则 handler 会被调用两次，
    # `await_args`（最后一次调用）就不再是这里想断言的那个请求。
    handler = AsyncMock(return_value=ModelResponse(
        result=[AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])]
    ))

    await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()
    sent_request = handler.await_args.args[0]
    assert "<current_plan>" in sent_request.system_prompt
    assert "</current_plan>" in sent_request.system_prompt
    assert "[in_progress] 搜索 AWS 信息" in sent_request.system_prompt
    assert "[pending] 搜索 Azure 信息" in sent_request.system_prompt
    assert "你是 Lead Agent" in sent_request.system_prompt


async def test_skips_injection_when_no_plan_in_state() -> None:
    middleware = PlanContextMiddleware()
    request = _request(state={})
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(request, handler)

    sent_request = handler.await_args.args[0]
    assert sent_request.system_prompt == "你是 Lead Agent"


async def test_skips_injection_when_plan_is_empty_list() -> None:
    middleware = PlanContextMiddleware()
    request = _request(state={"plan": []})
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(request, handler)

    sent_request = handler.await_args.args[0]
    assert sent_request.system_prompt == "你是 Lead Agent"


async def test_retries_once_when_final_response_leaves_plan_incomplete() -> None:
    """模型准备直接结束（无 tool_calls）但计划最后一步还没标 completed：强制重试一次。"""
    middleware = PlanContextMiddleware()
    request = _request(state={"plan": [
        {"content": "步骤一", "status": "completed"},
        {"content": "步骤二", "status": "in_progress"},
    ]})
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="这是最终答案")]))

    await middleware.awrap_model_call(request, handler)

    assert handler.await_count == 2
    retry_request = handler.await_args.args[0]
    assert "检测到当前计划" in retry_request.system_prompt
    assert "[in_progress] 步骤二" in retry_request.system_prompt


async def test_no_retry_when_plan_already_fully_completed() -> None:
    middleware = PlanContextMiddleware()
    request = _request(state={"plan": [
        {"content": "步骤一", "status": "completed"},
        {"content": "步骤二", "status": "completed"},
    ]})
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="这是最终答案")]))

    await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()


async def test_no_retry_when_response_still_has_tool_calls() -> None:
    """模型这次响应还在调用其它工具（不是最终回复）：即使计划未收尾也不该打断。"""
    middleware = PlanContextMiddleware()
    request = _request(state={"plan": [
        {"content": "步骤一", "status": "in_progress"},
    ]})
    handler = AsyncMock(return_value=ModelResponse(
        result=[AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])]
    ))

    await middleware.awrap_model_call(request, handler)

    handler.assert_awaited_once()
