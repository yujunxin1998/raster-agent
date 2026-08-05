"""`TitleMiddleware` 单元测试。

核心回归点：判断"是不是第一轮"不能直接数消息条数——第一轮里只要触发过一次工具
调用（哪怕是 `search_knowledge_base`/`query_database` 这类不需要好几轮的直接挂载
技能），就会多出 `AIMessage(tool_calls=[...])`/`ToolMessage` 这一对，把消息数顶到
2 条以上；改成数 `HumanMessage` 条数后，第一轮无论中间发生多少次工具调用，标题
生成都应该正常触发。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.middlewares.title import TitleMiddleware


def _make_middleware(on_title_generated=None) -> TitleMiddleware:
    return TitleMiddleware(
        prompt_factory=MagicMock(), model_name="m", provider="p", api_key="k", base_url="b",
        on_title_generated=on_title_generated,
    )


def _make_runtime(*, conversation_id: str | None = "c1") -> SimpleNamespace:
    context = AgentRuntimeContext(conversation_id=conversation_id) if conversation_id is not None else None
    return SimpleNamespace(context=context)


async def test_first_turn_without_tool_calls_triggers_generation() -> None:
    middleware = _make_middleware()
    state = {"messages": [HumanMessage(content="你好"), AIMessage(content="你好，有什么可以帮你")]}

    with patch("src.agent_core.middlewares.title.asyncio.create_task") as mock_create_task:
        await middleware.aafter_agent(state, _make_runtime())

    mock_create_task.assert_called_once()


async def test_first_turn_with_tool_calls_still_triggers_generation() -> None:
    """回归测试：第一轮里带工具调用，消息数会超过 2 条，标题生成依然要触发。"""
    middleware = _make_middleware()
    state = {"messages": [
        HumanMessage(content="今天上证指数收盘涨跌幅是多少？"),
        AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "call-1"}]),
        ToolMessage(content="搜索结果...", tool_call_id="call-1"),
        AIMessage(content="今天上证指数收盘上涨 0.96%"),
    ]}

    with patch("src.agent_core.middlewares.title.asyncio.create_task") as mock_create_task:
        await middleware.aafter_agent(state, _make_runtime())

    mock_create_task.assert_called_once()


async def test_second_turn_does_not_retrigger_generation() -> None:
    middleware = _make_middleware()
    state = {"messages": [
        HumanMessage(content="你好"), AIMessage(content="你好"),
        HumanMessage(content="再问一个问题"), AIMessage(content="回答"),
    ]}

    with patch("src.agent_core.middlewares.title.asyncio.create_task") as mock_create_task:
        await middleware.aafter_agent(state, _make_runtime())

    mock_create_task.assert_not_called()


async def test_missing_conversation_id_skips_generation() -> None:
    middleware = _make_middleware()
    state = {"messages": [HumanMessage(content="你好"), AIMessage(content="你好")]}

    with patch("src.agent_core.middlewares.title.asyncio.create_task") as mock_create_task:
        await middleware.aafter_agent(state, _make_runtime(conversation_id=None))

    mock_create_task.assert_not_called()


async def test_generate_calls_on_title_generated_callback_with_trimmed_title() -> None:
    on_title_generated = AsyncMock()
    middleware = _make_middleware(on_title_generated=on_title_generated)
    middleware._prompt_factory.render = MagicMock(return_value="prompt text")
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=AIMessage(content='"《智能助手功能介绍》"'))

    with patch("src.agent_core.middlewares.title.create_chat_model", return_value=fake_llm):
        await middleware._generate("c1", "你好", "你好，有什么可以帮你")

    on_title_generated.assert_awaited_once_with("c1", "智能助手功能介绍")
