"""`SkillMiddleware` 单元测试（Skill 注入重构设计文档第四、五节）。

覆盖两个钩子：`awrap_model_call` 动态注入技能目录，`awrap_tool_call` 拦截
`load_skill` 调用做校验 + 加载，短路不调用 `handler`；其余工具调用原样透传。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

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

    async def activate(self, skill_name: str, allowed_categories: frozenset[str]) -> str:
        self.last_call_args = (skill_name, allowed_categories)
        if self._error is not None:
            raise self._error
        return self._result


def _workflow_skill(tool_name: str, category: str = "general") -> SkillDefinition:
    return SkillDefinition(
        name=tool_name, tool_name=tool_name, description=f"{tool_name} 描述",
        category=SkillCategory(category), skill_dir=Path(f"skills/public/{tool_name}"), parameters=[],
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


def _tool_call_request(tool_name: str, args: dict) -> ToolCallRequest:
    runtime = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1", user_id="u1"))
    return ToolCallRequest(tool_call={"name": tool_name, "args": args, "id": "call-1"}, tool=None, state={}, runtime=runtime)


async def test_awrap_model_call_injects_catalog_when_workflow_skills_exist() -> None:
    registry = _registry(_workflow_skill("data-analysis"))
    middleware = _middleware(registry)
    request = ModelRequest(
        model=None, system_prompt="你是 Lead Agent", messages=[], tool_choice=None,
        tools=[], response_format=None, state={},
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1")),
    )
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(request, handler)

    sent_request = handler.await_args.args[0]
    assert "data-analysis" in sent_request.system_prompt
    assert "load_skill" in sent_request.system_prompt
    assert "你是 Lead Agent" in sent_request.system_prompt


async def test_awrap_model_call_skips_catalog_when_no_workflow_skills() -> None:
    middleware = _middleware(SkillRegistry())
    request = ModelRequest(
        model=None, system_prompt="你是 Lead Agent", messages=[], tool_choice=None,
        tools=[], response_format=None, state={},
        runtime=SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1")),
    )
    handler = AsyncMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))

    await middleware.awrap_model_call(request, handler)

    sent_request = handler.await_args.args[0]
    assert sent_request.system_prompt == "你是 Lead Agent"


async def test_awrap_tool_call_passes_through_non_load_skill_calls() -> None:
    middleware = _middleware(SkillRegistry())
    request = _tool_call_request("run_python", {"code": "print(1)"})
    handler = AsyncMock(return_value=ToolMessage(content="1", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_awaited_once_with(request)
    assert result.content == "1"


async def test_awrap_tool_call_short_circuits_load_skill_without_calling_handler() -> None:
    activation_service = _FakeActivationService(result="技能正文")
    middleware = _middleware(SkillRegistry(), activation_service=activation_service)
    request = _tool_call_request("load_skill", {"skill_name": "data-analysis"})
    handler = AsyncMock(return_value=ToolMessage(content="不应该被返回", tool_call_id="call-1"))

    result = await middleware.awrap_tool_call(request, handler)

    handler.assert_not_awaited()
    assert result.content == "技能正文"
    assert result.tool_call_id == "call-1"
    assert activation_service.last_call_args[0] == "data-analysis"


async def test_awrap_tool_call_checks_guardrail_by_skill_name_not_load_skill() -> None:
    guardrail = _FakeGuardrailProvider(GuardrailDecision.allow())
    middleware = _middleware(SkillRegistry(), guardrail=guardrail)
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
