"""`delegation_tools.build_task_tool` 的单元测试。

`task(subagent_type, task)` 是通用分发工具（DeerFlow 风格），按 `subagent_type`
查 `subagent_profiles._PROFILES` 拿到 profile 后交给 `run_subagent` 执行——这里
只测分发这一层的逻辑（查到/查不到、参数传递是否正确），不测 `run_subagent` 本身
（已经在 `test_sub_agent_config_isolation.py` 里覆盖）。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent_core.agents.delegation_tools import build_task_tool
from src.agent_core.agents.subagent_profiles import SubagentProfile
from src.agent_core.middlewares.context import AgentRuntimeContext

_CONTEXT = AgentRuntimeContext(conversation_id="c1")


async def _invoke(tool, **kwargs) -> str:
    return await tool.coroutine(runtime=SimpleNamespace(context=_CONTEXT), **kwargs)


async def test_unknown_subagent_type_returns_error_without_calling_run_subagent() -> None:
    tool = build_task_tool()

    with patch("src.agent_core.agents.delegation_tools.run_subagent") as mock_run:
        result = await _invoke(tool, subagent_type="does-not-exist", task="帮我查点东西")

    mock_run.assert_not_called()
    assert "does-not-exist" in result
    assert "web-researcher" in result  # 唯一已注册的类型，应该出现在可用列表里


async def test_known_subagent_type_dispatches_with_profile_config() -> None:
    tool = build_task_tool()
    fake_tools = [MagicMock(name="web_search")]
    fake_profile = SubagentProfile(
        name="web-researcher", description="test profile",
        system_prompt_factory=lambda: "system prompt text",
        tools_factory=lambda: fake_tools,
    )

    with patch("src.agent_core.agents.delegation_tools.get_profile", return_value=fake_profile), \
         patch("src.agent_core.agents.delegation_tools.run_subagent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "子 Agent 的回复"
        result = await _invoke(tool, subagent_type="web-researcher", task="查一下天气")

    assert result == "子 Agent 的回复"
    mock_run.assert_awaited_once()
    call_kwargs = mock_run.await_args.kwargs
    assert call_kwargs["agent_name"] == "web-researcher"
    assert call_kwargs["system_prompt"] == "system prompt text"
    assert call_kwargs["tools"] == fake_tools
    assert call_kwargs["task"] == "查一下天气"
    assert call_kwargs["context"] is _CONTEXT


def test_task_tool_description_lists_registered_subagent_types() -> None:
    tool = build_task_tool()
    assert "web-researcher" in tool.description
