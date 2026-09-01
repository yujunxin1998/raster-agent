"""`sub_agent_factory.run_subagent` 的 context 隔离 + 超时单元测试，以及
"能力解析结果一旦拿到，就不再受后续 Registry 原子切换影响"的在途任务稳定性
测试。

`task` 工具内部现造的子 Agent 只接收 `context`（`AgentRuntimeContext`），不接收
外层的 `RunnableConfig`——因此天然拿不到外层的 `callbacks`：否则子 Agent 自己的
模型/工具调用事件会经由回调传播机制泄漏进外层 `astream_events` 流，污染用户
可见的 token 流（见模块 docstring）。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent_core.agents.sub_agent_factory import run_subagent
from src.agent_core.agents.subagent_capability_resolver import resolve_subagent_capabilities
from src.agent_core.agents.subagent_profiles import SubagentProfile
from src.agent_core.guardrail.guardrail_provider import GuardrailDecision, GuardrailProvider
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import ToolDefinition


class _AllowAllGuardrailProvider(GuardrailProvider):
    async def check(self, *, user_id, tool_name, tool_args, context) -> GuardrailDecision:
        return GuardrailDecision.allow()


def _tool_definition(name: str) -> ToolDefinition:
    """`build_tool` 用闭包捕获一个已经构造好的具体对象（跟
    `providers.py::BuiltinToolProvider.discover()` 的真实写法一致），不是
    每次调用现造——这正是"解析出的 Tool 对象跟后续 Registry 切换无关"的
    根本原因：拿到手的就是这一个持久对象，不会因为 Snapshot 被替换而失效。
    """
    tool_instance = MagicMock(name=f"tool:{name}")
    return ToolDefinition(
        canonical_name=f"builtin.{name}", model_name=name, description="",
        source_type="builtin", source_id="builtin", scope="application",
        permissions_key=name, build_tool=lambda: tool_instance,
    )


async def test_in_flight_task_keeps_using_capabilities_resolved_before_registry_swap() -> None:
    """解析一次拿到 `ResolvedSubagentCapabilities` 后，即使源头 `ToolRegistry`
    快照立刻被替换（工具被下线），已经持有的旧解析结果仍然可以正常喂给
    `run_subagent()` 执行——不需要引用计数，`RegistrySnapshot` 本身的
    copy-on-write 设计（`replace_source` 从不修改旧快照）加上"resolve 一次、
    不再回头查"的调用方式，天然满足"在途任务不受后续切换影响"。
    """
    profile = SubagentProfile(
        name="web-researcher", description="test", system_prompt_factory=lambda: "sp",
        required_tools=frozenset({"web_search"}),
    )
    snapshot_v1 = RegistrySnapshot.empty().replace_source("builtin", [_tool_definition("web_search")])

    resolved = await resolve_subagent_capabilities(
        profile=profile, snapshot=snapshot_v1, skill_manager=type("_M", (), {"registry": SkillRegistry()})(),
        subagent_only_tools={}, guardrail_provider=_AllowAllGuardrailProvider(), user_id="u1",
    )
    assert len(resolved.tools) == 1
    resolved_tool_instance = resolved.tools[0]

    # 模拟 web_search 被下线：新快照里这个来源发布空列表。
    snapshot_v2 = snapshot_v1.replace_source("builtin", [])
    assert snapshot_v2.by_model_name.get("web_search") is None  # 新快照里确实查不到了

    fake_sub_agent = MagicMock()
    fake_sub_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="子 Agent 的回复")]})
    with patch("src.agent_core.agents.sub_agent_factory.create_agent", return_value=fake_sub_agent) as mock_create, \
         patch("src.agent_core.agents.sub_agent_factory.create_chat_model", return_value=MagicMock()):
        result = await run_subagent(
            agent_name="web-researcher", system_prompt="sp", tools=resolved.tools,
            task="查一下天气", context=AgentRuntimeContext(conversation_id="c1"),
            middleware=resolved.middleware,
        )

    assert result == "子 Agent 的回复"
    assert mock_create.call_args.kwargs["tools"] == [resolved_tool_instance]


async def test_sub_agent_receives_context_without_outer_config() -> None:
    fake_sub_agent = MagicMock()
    fake_sub_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="子 Agent 的回复")]})

    with patch("src.agent_core.agents.sub_agent_factory.create_agent", return_value=fake_sub_agent) as mock_create, \
         patch("src.agent_core.agents.sub_agent_factory.create_chat_model", return_value=MagicMock()):
        context = AgentRuntimeContext(conversation_id="c1", user_id="u1")
        result = await run_subagent(
            agent_name="web-researcher", system_prompt="you are a test agent",
            tools=[], task="帮我查一下天气", context=context,
        )

    assert result == "子 Agent 的回复"
    assert mock_create.call_args.kwargs["context_schema"] is AgentRuntimeContext
    child_middlewares = mock_create.call_args.kwargs["middleware"]
    assert isinstance(child_middlewares[0], ToolErrorHandlingMiddleware)
    assert isinstance(child_middlewares[1], LoopDetectionMiddleware)
    fake_sub_agent.ainvoke.assert_awaited_once()
    _, call_kwargs = fake_sub_agent.ainvoke.await_args
    assert call_kwargs == {"context": context}


async def test_sub_agent_exception_returns_error_text_not_raise() -> None:
    fake_sub_agent = MagicMock()
    fake_sub_agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("src.agent_core.agents.sub_agent_factory.create_agent", return_value=fake_sub_agent), \
         patch("src.agent_core.agents.sub_agent_factory.create_chat_model", return_value=MagicMock()):
        result = await run_subagent(
            agent_name="web-researcher", system_prompt="you are a test agent",
            tools=[], task="task", context=AgentRuntimeContext(conversation_id="c1"),
        )

    assert "web-researcher" in result
    assert "boom" in result


async def test_sub_agent_no_reply_returns_fallback_text() -> None:
    fake_sub_agent = MagicMock()
    fake_sub_agent.ainvoke = AsyncMock(return_value={"messages": []})

    with patch("src.agent_core.agents.sub_agent_factory.create_agent", return_value=fake_sub_agent), \
         patch("src.agent_core.agents.sub_agent_factory.create_chat_model", return_value=MagicMock()):
        result = await run_subagent(
            agent_name="web-researcher", system_prompt="you are a test agent",
            tools=[], task="task", context=AgentRuntimeContext(conversation_id="c1"),
        )

    assert "web-researcher" in result
    assert "未产生有效回复" in result


async def test_sub_agent_timeout_returns_error_text_not_hang() -> None:
    async def _never_returns(*args, **kwargs):
        await asyncio.sleep(10)
        return {"messages": [AIMessage(content="不应该走到这里")]}

    fake_sub_agent = MagicMock()
    fake_sub_agent.ainvoke = AsyncMock(side_effect=_never_returns)

    with patch("src.agent_core.agents.sub_agent_factory.create_agent", return_value=fake_sub_agent), \
         patch("src.agent_core.agents.sub_agent_factory.create_chat_model", return_value=MagicMock()), \
         patch("src.agent_core.agents.sub_agent_factory.get_settings",
               return_value=MagicMock(SUBAGENT_TIMEOUT_SECONDS=0.05)):
        result = await run_subagent(
            agent_name="web-researcher", system_prompt="you are a test agent",
            tools=[], task="task", context=AgentRuntimeContext(conversation_id="c1"),
        )

    assert "web-researcher" in result
    assert "超时" in result
