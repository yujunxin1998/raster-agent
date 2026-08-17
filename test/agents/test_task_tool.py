"""`delegation_tools.build_task_tool` 的单元测试。

`task(subagent_type, task)` 是通用分发工具（DeerFlow 风格），按 `subagent_type`
查 `subagent_profiles._PROFILES` 拿到 profile，派发前先经
`resolve_subagent_capabilities()` 解析出实际可用的 Tool/Middleware（必需能力
缺失时 fail fast，不调用 `run_subagent`），再交给 `run_subagent` 执行——这里只测
分发这一层的逻辑（查到/查不到、能力解析结果如何路由、参数传递是否正确），不测
`resolve_subagent_capabilities()` 内部的解析细节（覆盖在
`test_subagent_capability_resolver.py`）也不测 `run_subagent` 本身（覆盖在
`test_sub_agent_config_isolation.py`）。
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent_core.agents.delegation_tools import build_task_tool
from src.agent_core.agents.subagent_capability_resolver import (
    ResolvedSubagentCapabilities,
    SubagentCapabilityUnavailable,
)
from src.agent_core.agents.subagent_profiles import SubagentProfile
from src.agent_core.middlewares.context import AgentRuntimeContext

_CONTEXT = AgentRuntimeContext(conversation_id="c1")


async def _invoke(tool, **kwargs) -> str:
    return await tool.coroutine(runtime=SimpleNamespace(context=_CONTEXT), **kwargs)


@contextmanager
def _patched_capability_deps(*, resolve_return_value=None, resolve_side_effect=None):
    """`_invoke()` 在真正调用 `resolve_subagent_capabilities()` 前，会把
    `get_tool_registry()`/`get_skill_manager()`/`get_guardrail_provider()`
    的返回值当作关键字参数表达式求值——即使 `resolve_subagent_capabilities`
    本身被 mock 掉，这三个 getter 依然会被真的调用一次，所以必须一起 patch，
    否则会因为对应的全局单例在测试里从未 `init_xxx()` 过而抛
    `RuntimeError`。收拢成一个 context manager，避免每个测试都重复四行 patch。
    """
    with patch("src.agent_core.tools.registry.get_tool_registry", return_value=MagicMock()), \
         patch("src.agent_core.agents.delegation_tools.get_skill_manager", return_value=MagicMock()), \
         patch("src.agent_core.agents.delegation_tools.get_guardrail_provider", return_value=MagicMock()), \
         patch(
             "src.agent_core.agents.delegation_tools.resolve_subagent_capabilities",
             new_callable=AsyncMock, return_value=resolve_return_value, side_effect=resolve_side_effect,
         ) as mock_resolve:
        yield mock_resolve


def _resolved(*, tools=None, middleware=None, missing_optional=()) -> ResolvedSubagentCapabilities:
    return ResolvedSubagentCapabilities(
        tools=tools or [], middleware=middleware or [],
        tool_registry_revision=1, skill_registry_revision=1, missing_optional=missing_optional,
    )


_FAKE_PROFILE = SubagentProfile(
    name="web-researcher", description="test profile",
    system_prompt_factory=lambda: "system prompt text",
)


async def test_unknown_subagent_type_returns_error_without_calling_run_subagent() -> None:
    tool = build_task_tool()

    with patch("src.agent_core.agents.delegation_tools.run_subagent") as mock_run:
        result = await _invoke(tool, subagent_type="does-not-exist", task="帮我查点东西")

    mock_run.assert_not_called()
    assert "does-not-exist" in result
    assert "web-researcher" in result  # 唯一已注册的类型，应该出现在可用列表里


async def test_known_subagent_type_dispatches_with_resolved_capabilities() -> None:
    tool = build_task_tool()
    fake_tools = [MagicMock(name="web_search")]
    fake_middleware = [MagicMock(name="SkillMiddleware instance")]

    with patch("src.agent_core.agents.delegation_tools.get_profile", return_value=_FAKE_PROFILE), \
         _patched_capability_deps(resolve_return_value=_resolved(tools=fake_tools, middleware=fake_middleware)), \
         patch("src.agent_core.agents.delegation_tools.run_subagent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "子 Agent 的回复"
        result = await _invoke(tool, subagent_type="web-researcher", task="查一下天气")

    assert result == "子 Agent 的回复"
    mock_run.assert_awaited_once()
    call_kwargs = mock_run.await_args.kwargs
    assert call_kwargs["agent_name"] == "web-researcher"
    assert call_kwargs["system_prompt"] == "system prompt text"
    assert call_kwargs["tools"] == fake_tools
    assert call_kwargs["middleware"] == fake_middleware
    assert call_kwargs["task"] == "查一下天气"
    assert call_kwargs["context"] is _CONTEXT


async def test_required_capability_missing_returns_error_without_calling_run_subagent() -> None:
    """`resolve_subagent_capabilities()` 认定必需能力缺失时，`task` 必须直接
    返回明确的失败文案，绝不能继续创建子 Agent——不能让缺了搜索工具的
    web-researcher 继续跑，凭模型自己的知识伪装成联网结果。
    """
    tool = build_task_tool()

    with patch("src.agent_core.agents.delegation_tools.get_profile", return_value=_FAKE_PROFILE), \
         _patched_capability_deps(
             resolve_side_effect=SubagentCapabilityUnavailable("web-researcher", ["web_search"])
         ), \
         patch("src.agent_core.agents.delegation_tools.run_subagent", new_callable=AsyncMock) as mock_run:
        result = await _invoke(tool, subagent_type="web-researcher", task="查一下天气")

    mock_run.assert_not_called()
    assert "web-researcher" in result
    assert "web_search" in result


async def test_degraded_optional_capability_still_dispatches() -> None:
    """可选能力缺失只降级，不阻止派发——`resolve_subagent_capabilities()` 已经
    把它从 `tools` 里剔除，这里只需要确认 `task` 仍然正常调用 `run_subagent`。
    """
    tool = build_task_tool()

    with patch("src.agent_core.agents.delegation_tools.get_profile", return_value=_FAKE_PROFILE), \
         _patched_capability_deps(resolve_return_value=_resolved(missing_optional=("some_optional_tool",))), \
         patch("src.agent_core.agents.delegation_tools.run_subagent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "ok"
        result = await _invoke(tool, subagent_type="web-researcher", task="任务")

    assert result == "ok"
    mock_run.assert_awaited_once()
    assert mock_run.await_args.kwargs["tools"] == []


def test_task_tool_description_lists_registered_subagent_types() -> None:
    tool = build_task_tool()
    assert "web-researcher" in tool.description


async def test_concurrent_calls_are_capped_by_settings() -> None:
    """回归测试：`SUBAGENT_MAX_CONCURRENCY` 生效——超过上限的调用排队等待，
    而不是无限制全部并发跑。Semaphore 在 `build_task_tool()` 内部按当时的
    `get_settings()` 取值创建，所以要在 `build_task_tool()` 调用前就把配置
    打好。
    """
    active = 0
    max_active = 0
    release_event = asyncio.Event()

    async def fake_run_subagent(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await release_event.wait()
        active -= 1
        return "ok"

    with patch("src.agent_core.agents.delegation_tools.get_settings",
               return_value=MagicMock(SUBAGENT_MAX_CONCURRENCY=2)):
        tool = build_task_tool()

    with patch("src.agent_core.agents.delegation_tools.get_profile", return_value=_FAKE_PROFILE), \
         _patched_capability_deps(resolve_return_value=_resolved()), \
         patch("src.agent_core.agents.delegation_tools.run_subagent", side_effect=fake_run_subagent):
        tasks = [
            asyncio.create_task(_invoke(tool, subagent_type="web-researcher", task=f"t{i}"))
            for i in range(3)
        ]
        await asyncio.sleep(0.05)  # 让前两个抢到 Semaphore 并卡在 release_event 上
        assert active == 2  # 第三个应该还在排队，没能进入 fake_run_subagent
        assert max_active == 2

        release_event.set()
        results = await asyncio.gather(*tasks)

    assert results == ["ok", "ok", "ok"]
