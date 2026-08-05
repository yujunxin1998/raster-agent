"""MemoryCompressor.compress_messages 单元测试（graph-free 压缩核心逻辑）。

`_summarize` 会调用真实 LLM，测试里统一 mock 掉，只验证阈值判断 +
`state_update` 的组装是否正确。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import RemoveMessage

from src.agent_core.memory.memory_compressor import MemoryCompressor

_SUMMARY_FAILURE_PREFIX = "[摘要生成失败，"


def _make_compressor(threshold: int = 2, keep_recent: int = 1) -> MemoryCompressor:
    return MemoryCompressor(
        model_name="test-model", provider="deepseek", api_key="k", base_url="u",
        default_threshold=threshold, default_keep_recent=keep_recent,
    )


def _messages(count: int) -> list:
    messages = []
    for index in range(count):
        cls = HumanMessage if index % 2 == 0 else AIMessage
        messages.append(cls(content=f"消息{index}", id=str(index)))
    return messages


async def test_below_threshold_returns_none() -> None:
    compressor = _make_compressor(threshold=2)
    messages = _messages(2)

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_above_threshold_compresses_and_builds_state_update() -> None:
    compressor = _make_compressor(threshold=2, keep_recent=1)
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = _messages(3)  # 超过阈值 2

    outcome = await compressor.compress_messages(messages)

    assert outcome is not None
    assert outcome.summary == "摘要内容"
    assert outcome.compressed_count == 2  # 保留最近 1 条，压缩前面 2 条

    update_messages = outcome.state_update["messages"]
    assert len(update_messages) == 3  # 2 条 RemoveMessage + 1 条摘要 SystemMessage
    assert all(isinstance(message, RemoveMessage) for message in update_messages[:2])
    assert {message.id for message in update_messages[:2]} == {"0", "1"}
    assert isinstance(update_messages[2], SystemMessage)
    assert update_messages[2].content == "摘要内容"


async def test_summary_generation_failure_returns_none() -> None:
    compressor = _make_compressor(threshold=2, keep_recent=1)
    compressor._summarize = AsyncMock(return_value=f"{_SUMMARY_FAILURE_PREFIX}原始消息 2 条已归档]")
    messages = _messages(3)

    outcome = await compressor.compress_messages(messages)

    assert outcome is None
