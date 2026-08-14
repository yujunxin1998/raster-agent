"""`sub_agent_factory.run_subagent` 的 context 隔离单元测试。

`task` 工具内部现造的子 Agent 只接收 `context`（`AgentRuntimeContext`），不接收
外层的 `RunnableConfig`——因此天然拿不到外层的 `callbacks`：否则子 Agent 自己的
模型/工具调用事件会经由回调传播机制泄漏进外层 `astream_events` 流，污染用户
可见的 token 流（见模块 docstring）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent_core.agents.sub_agent_factory import run_subagent
from src.agent_core.middlewares.context import AgentRuntimeContext


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
