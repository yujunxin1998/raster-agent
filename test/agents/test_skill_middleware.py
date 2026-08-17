"""`SkillMiddleware` 单元测试。

覆盖两个钩子：`awrap_model_call` 预路由（forced 首次调用前注入正文 + 目录
注入 catalog_candidates），`awrap_tool_call` 拦截 `load_skill` 调用做校验 +
补充激活，短路不调用 `handler`；其余工具调用原样透传。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from src.agent_core.agents.skill_middleware import SkillMiddleware
from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.constants import SkillCategory


class _FakeGuardrailProvider:
    def __init__(self, decision: GuardrailDecision) -> None:
        self._decision = decision
        self.last_call_kwargs: dict | None = None

    async def check(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._decision


class _FakeActivationService:
    def __init__(self, *, result: str | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.last_call_args: tuple | None = None
        self.call_count = 0

    async def activate(self, skill_name: str, allowed_categories: frozenset[str]) -> str:
        self.call_count += 1
        self.last_call_args = (skill_name, allowed_categories)
        if self._error is not None:
            raise self._error
        return self._result


def _workflow_skill(name: str, category: str = "general", activation: str = "automatic") -> SkillDefinition:
    return SkillDefinition(
        name=name, description=f"{name} 描述",
        category=SkillCategory(category), skill_dir=Path(f"skills/public/{name}"),
        activation=activation,  # type: ignore[arg-type]
    )


def _registry(*skills: SkillDefinition) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def _middleware(registry: SkillRegistry, *, guardrail=None, activation_service=None) -> SkillMiddleware:
    return SkillMiddleware(
        SimpleNamespace(registry=registry),
        guardrail or _FakeGuardrailProvider(GuardrailDecision.allow()),
        activation_service or _FakeActivationService(result="正文"),
        allowed_categories=frozenset({"general", "tool", "database"}),
    )


def _model_request(state: dict | None = None, system_prompt: str = "你是 Lead Agent") -> ModelRequest:
    return ModelRequest(
        model=None, system_prompt=system_prompt, messages=[], tool_choice=None,
        tools=[], response_format=None, state=state or {},
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1")),
    )


def _tool_call_request(tool_name: str, args: dict, state: dict | None = None) -> ToolCallRequest:
    runtime = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", user_id="u1"))
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": "call-1"}, tool=None,
        state=state or {}, runtime=runtime,
    )


async def test_awrap_model_call_injects_catalog_for_automatic_skill() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    middleware = _middleware(registry)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(_model_request(), handler)

    sent_request = handler.await_args.args[0]
    assert "data-analysis" in sent_request.system_prompt
    assert "load_skill" in sent_request.system_prompt
    assert "你是 Lead Agent" in sent_request.system_prompt


async def test_awrap_model_call_skips_catalog_when_no_workflow_skills() -> None:
    middleware = _middleware(SkillRegistry())
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    result = await middleware.awrap_model_call(_model_request(), handler)

    sent_request = handler.await_args.args[0]
    assert sent_request.system_prompt == "你是 Lead Agent"
    assert isinstance(result, ModelResponse)


async def test_required_skill_is_activated_before_first_model_call_without_load_skill() -> None:
    """回归测试：`activation: required` 的技能不依赖模型调用 load_skill，
    首次模型调用前正文就已经在 system_prompt 里（重构文档 7.1 节核心诉求）。
    """
    registry = _registry(_workflow_skill("must-run", activation="required"))
    activation_service = _FakeActivationService(result="必须运行的正文")
    middleware = _middleware(registry, activation_service=activation_service)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    result = await middleware.awrap_model_call(_model_request(), handler)

    sent_request = handler.await_args.args[0]
    assert "必须运行的正文" in sent_request.system_prompt
    assert activation_service.call_count == 1
    assert isinstance(result, ExtendedModelResponse)
    assert result.command.update["activated_skills"]["must-run"]["activation_source"] == "required"


async def test_already_activated_skill_is_not_reinjected_or_reactivated() -> None:
    registry = _registry(_workflow_skill("must-run", activation="required"))
    activation_service = _FakeActivationService(result="正文")
    middleware = _middleware(registry, activation_service=activation_service)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    state = {"activated_skills": {"must-run": {"skill_version": None, "activation_source": "required", "activated_at": "t"}}}

    result = await middleware.awrap_model_call(_model_request(state=state), handler)

    assert activation_service.call_count == 0
    sent_request = handler.await_args.args[0]
    assert "正文" not in sent_request.system_prompt
    assert isinstance(result, ModelResponse)


async def test_already_activated_skill_excluded_from_catalog() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    middleware = _middleware(registry)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    state = {"activated_skills": {"data-analysis": {"skill_version": None, "activation_source": "load_skill", "activated_at": "t"}}}

    await middleware.awrap_model_call(_model_request(state=state), handler)

    sent_request = handler.await_args.args[0]
    assert "data-analysis" not in sent_request.system_prompt


async def test_explicit_skill_name_forces_activation_even_when_automatic() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    activation_service = _FakeActivationService(result="数据分析正文")
    middleware = _middleware(registry, activation_service=activation_service)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    runtime = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", explicit_skill_names=("data-analysis",)))
    request = ModelRequest(
        model=None, system_prompt="你是 Lead Agent", messages=[], tool_choice=None,
        tools=[], response_format=None, state={}, runtime=runtime,
    )

    await middleware.awrap_model_call(request, handler)

    sent_request = handler.await_args.args[0]
    assert "数据分析正文" in sent_request.system_prompt


async def test_forced_skill_denied_by_guardrail_is_not_activated() -> None:
    registry = _registry(_workflow_skill("must-run", activation="required"))
    guardrail = _FakeGuardrailProvider(GuardrailDecision.deny("禁用"))
    activation_service = _FakeActivationService(result="不应出现")
    middleware = _middleware(registry, guardrail=guardrail, activation_service=activation_service)
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    result = await middleware.awrap_model_call(_model_request(), handler)

    assert activation_service.call_count == 0
    sent_request = handler.await_args.args[0]
    assert "不应出现" not in sent_request.system_prompt
    assert isinstance(result, ModelResponse)


async def test_awrap_tool_call_passes_through_non_load_skill_calls() -> None:
    middleware = _middleware(SkillRegistry())
    request = _tool_call_request("run_python", {"code": "print(1)"})
    handler = AsyncMock(return_value=ToolMessage(content="1", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_awaited_once_with(request)
    assert result.content == "1"


async def test_awrap_tool_call_short_circuits_load_skill_without_calling_handler() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    activation_service = _FakeActivationService(result="技能正文")
    middleware = _middleware(registry, activation_service=activation_service)
    request = _tool_call_request("load_skill", {"skill_name": "data-analysis"})
    handler = AsyncMock(return_value=ToolMessage(content="不应该被返回", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_not_awaited()
    assert isinstance(result, Command)
    assert result.update["messages"][0].content == "技能正文"
    assert result.update["messages"][0].tool_call_id == "call-1"
    assert result.update["activated_skills"]["data-analysis"]["activation_source"] == "load_skill"
    assert activation_service.last_call_args[0] == "data-analysis"


async def test_awrap_tool_call_load_skill_already_activated_returns_lightweight_message() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    activation_service = _FakeActivationService(result="不应该被再次加载")
    middleware = _middleware(registry, activation_service=activation_service)
    state = {"activated_skills": {"data-analysis": {"skill_version": None, "activation_source": "required", "activated_at": "t"}}}
    request = _tool_call_request("load_skill", {"skill_name": "data-analysis"}, state=state)
    handler = AsyncMock()

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_not_awaited()
    assert activation_service.call_count == 0
    assert isinstance(result, ToolMessage)
    assert "已在本次对话中激活" in result.content


async def test_awrap_tool_call_checks_guardrail_by_skill_name_not_load_skill() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    guardrail = _FakeGuardrailProvider(GuardrailDecision.allow())
    middleware = _middleware(registry, guardrail=guardrail)
    request = _tool_call_request("load_skill", {"skill_name": "data-analysis"})
    handler = AsyncMock()

    await middleware.awrap_tool_call(request, handler)

    assert guardrail.last_call_kwargs["tool_name"] == "data-analysis"
    assert guardrail.last_call_kwargs["user_id"] == "u1"


async def test_awrap_tool_call_denied_returns_error_message_without_activating() -> None:
    guardrail = _FakeGuardrailProvider(GuardrailDecision.deny("该技能已被禁用"))
    activation_service = _FakeActivationService(result="不应该被加载")
    middleware = _middleware(SkillRegistry(), guardrail=guardrail, activation_service=activation_service)
    request = _tool_call_request("load_skill", {"skill_name": "data-analysis"})
    handler = AsyncMock()

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_not_awaited()
    assert result.status == "error"
    assert result.content == "该技能已被禁用"
    assert activation_service.last_call_args is None
