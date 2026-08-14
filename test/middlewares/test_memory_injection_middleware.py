"""`MemoryInjectionMiddleware` 单元测试。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.memory_injection import MemoryInjectionMiddleware


def _settings(**overrides) -> MagicMock:
    defaults = {"MEMORY_ENABLED": True, "MEMORY_INJECTION_ENABLED": True}
    defaults.update(overrides)
    return MagicMock(**defaults)


def _request(system_prompt: str | None, messages: list, context) -> MagicMock:
    request = MagicMock()
    request.system_prompt = system_prompt
    request.messages = messages
    request.runtime = SimpleNamespace(context=context)
    request.override = MagicMock(side_effect=lambda system_prompt: _request(system_prompt, messages, context))
    return request


async def test_injects_memory_context_into_system_prompt() -> None:
    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value="<memory>...</memory>")
    middleware = MemoryInjectionMiddleware(context_builder)
    context = AgentRuntimeContext(conversation_id="c1", user_id="u1")
    request = _request("原始 system prompt", [HumanMessage(content="记忆模块进展如何")], context)
    handler = AsyncMock(return_value="response")

    result = await middleware.awrap_model_call(request, handler)

    assert result == "response"
    handler.assert_awaited_once()
    called_request = handler.await_args.args[0]
    assert called_request.system_prompt == "原始 system prompt\n\n<memory>...</memory>"
    context_builder.build.assert_awaited_once_with(
        user_id="u1", query="记忆模块进展如何", cache=context.memory_cache,
    )


async def test_no_memory_context_leaves_system_prompt_unchanged() -> None:
    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value="")
    middleware = MemoryInjectionMiddleware(context_builder)
    context = AgentRuntimeContext(conversation_id="c1", user_id="u1")
    request = _request("原始 system prompt", [HumanMessage(content="你好")], context)
    handler = AsyncMock(return_value="response")

    await middleware.awrap_model_call(request, handler)

    called_request = handler.await_args.args[0]
    assert called_request.system_prompt == "原始 system prompt"


async def test_missing_user_id_skips_injection() -> None:
    context_builder = MagicMock()
    context_builder.build = AsyncMock()
    middleware = MemoryInjectionMiddleware(context_builder)
    context = AgentRuntimeContext(conversation_id="c1", user_id=None)
    request = _request("原始 system prompt", [HumanMessage(content="你好")], context)
    handler = AsyncMock(return_value="response")

    await middleware.awrap_model_call(request, handler)

    context_builder.build.assert_not_awaited()


async def test_memory_disabled_skips_injection() -> None:
    context_builder = MagicMock()
    context_builder.build = AsyncMock()
    middleware = MemoryInjectionMiddleware(context_builder)
    context = AgentRuntimeContext(conversation_id="c1", user_id="u1")
    request = _request("原始 system prompt", [HumanMessage(content="你好")], context)
    handler = AsyncMock(return_value="response")

    with patch("src.agent_core.middlewares.memory_injection.get_settings",
               return_value=_settings(MEMORY_ENABLED=False)):
        await middleware.awrap_model_call(request, handler)

    context_builder.build.assert_not_awaited()


async def test_passes_shared_memory_cache_across_calls_in_same_run() -> None:
    """同一次 Run 的多次模型调用应该复用同一个 context.memory_cache 实例。"""
    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value="")
    middleware = MemoryInjectionMiddleware(context_builder)
    context = AgentRuntimeContext(conversation_id="c1", user_id="u1")
    handler = AsyncMock(return_value="response")

    await middleware.awrap_model_call(_request(None, [HumanMessage(content="第一句")], context), handler)
    await middleware.awrap_model_call(_request(None, [HumanMessage(content="第二句")], context), handler)

    first_cache = context_builder.build.await_args_list[0].kwargs["cache"]
    second_cache = context_builder.build.await_args_list[1].kwargs["cache"]
    assert first_cache is second_cache is context.memory_cache
