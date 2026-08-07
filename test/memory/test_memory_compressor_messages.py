"""MemoryCompressor.compress_messages 单元测试（graph-free 压缩核心逻辑）。

`_summarize` 会调用真实 LLM，测试里统一 mock 掉，只验证触发条件判断 +
`state_update` 的组装是否正确。
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import RemoveMessage

from src.agent_core.memory.memory_compressor import MemoryCompressor

_SUMMARY_FAILURE_PREFIX = "[摘要生成失败，"


def _make_compressor(
    trigger_type: str = "messages",
    trigger_value: float = 2,
    keep_recent: int = 1,
    model_max_input_tokens: int = 0,
) -> MemoryCompressor:
    return MemoryCompressor(
        model_name="test-model", provider="deepseek", api_key="k", base_url="u",
        trigger_type=trigger_type, trigger_value=trigger_value,
        default_keep_recent=keep_recent, model_max_input_tokens=model_max_input_tokens,
    )


def _messages(count: int) -> list:
    messages = []
    for index in range(count):
        cls = HumanMessage if index % 2 == 0 else AIMessage
        messages.append(cls(content=f"消息{index}", id=str(index)))
    return messages


def test_constructor_rejects_unknown_trigger_type() -> None:
    try:
        _make_compressor(trigger_type="bytes")
        raised = False
    except ValueError:
        raised = True
    assert raised


async def test_below_threshold_returns_none() -> None:
    compressor = _make_compressor(trigger_value=2)
    messages = _messages(2)

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_above_threshold_compresses_and_builds_state_update() -> None:
    compressor = _make_compressor(trigger_value=2, keep_recent=1)
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = _messages(3)  # 超过阈值 2

    outcome = await compressor.compress_messages(messages)

    assert outcome is not None
    assert outcome.summary == "摘要内容"
    assert outcome.compressed_count == 2  # 保留最近 1 条，压缩前面 2 条

    update_messages = outcome.state_update["messages"]
    assert len(update_messages) == 3  # 2 条 RemoveMessage + 1 条摘要 AIMessage
    assert all(isinstance(message, RemoveMessage) for message in update_messages[:2])
    assert {message.id for message in update_messages[:2]} == {"0", "1"}
    # 摘要必须是 AIMessage 而非 SystemMessage：压缩后它停留在消息历史中间，
    # Qwen/vLLM 等模型的 chat template 要求 system 消息必须在最前面，
    # 非开头位置的 SystemMessage 会触发 400 "System message must be at the
    # beginning" 错误（见 memory_compressor.py 内注释）。
    assert isinstance(update_messages[2], AIMessage)
    assert update_messages[2].content == "[历史摘要]\n摘要内容"


async def test_summary_generation_failure_returns_none() -> None:
    compressor = _make_compressor(trigger_value=2, keep_recent=1)
    compressor._summarize = AsyncMock(return_value=f"{_SUMMARY_FAILURE_PREFIX}原始消息 2 条已归档]")
    messages = _messages(3)

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_split_point_on_tool_message_rewinds_to_include_its_ai_message() -> None:
    """回归测试：切分点如果恰好落在 ToolMessage 上，必须往前退到发起这次工具
    调用的 AIMessage(tool_calls)，让这一对消息一起留在保留窗口里——不能把
    ToolMessage 孤零零地留在保留窗口最前面而把它的 AIMessage 压缩掉，否则下一轮
    请求会被模型供应商 API 拒绝（"Messages with role 'tool' must be a response
    to a preceding message with 'tool_calls'"）。
    """
    compressor = _make_compressor(trigger_value=2, keep_recent=2)
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = [
        HumanMessage(content="帮我查一下", id="0"),
        AIMessage(content="", id="1", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]),
        ToolMessage(content="搜索结果", id="2", tool_call_id="call-1"),
        AIMessage(content="根据结果...", id="3"),
    ]
    # keep_recent=2 时初始切分点会落在 messages[2]（ToolMessage）上，
    # 必须往前退到 messages[1]（发起 tool_calls 的 AIMessage），让它和
    # ToolMessage 一起留在保留窗口，而不是被压缩掉。

    outcome = await compressor.compress_messages(messages)

    assert outcome is not None
    removed_ids = {
        message.id for message in outcome.state_update["messages"] if isinstance(message, RemoveMessage)
    }
    assert removed_ids == {"0"}
    assert outcome.compressed_count == 1


async def test_split_point_rewinding_to_empty_skips_compression() -> None:
    """如果往前退到底（所有消息都被 tool_calls 配对关系"拉"进保留窗口），
    这一轮没有可压缩的旧消息，应该跳过而不是压缩空列表。
    """
    compressor = _make_compressor(trigger_value=1, keep_recent=1)
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = [
        AIMessage(content="", id="0", tool_calls=[{"name": "search", "args": {}, "id": "call-1"}]),
        ToolMessage(content="搜索结果", id="1", tool_call_id="call-1"),
    ]

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_tokens_trigger_type_uses_approximate_token_count() -> None:
    """trigger_type="tokens" 时按近似 token 数而不是消息条数判断。"""
    compressor = _make_compressor(trigger_type="tokens", trigger_value=1000, keep_recent=1)
    messages = _messages(3)  # 3 条短消息，token 数远低于 1000

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_tokens_trigger_type_fires_when_over_value() -> None:
    compressor = _make_compressor(trigger_type="tokens", trigger_value=5, keep_recent=1)
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = _messages(3)  # 消息数不多，但把阈值设得很低，确保能超过

    outcome = await compressor.compress_messages(messages)

    assert outcome is not None


async def test_fraction_trigger_type_without_model_max_tokens_never_fires() -> None:
    """没配置 model_max_input_tokens（默认 0）时没法算比例，应该视为永不触发，
    而不是除零报错。
    """
    compressor = _make_compressor(trigger_type="fraction", trigger_value=0.1, keep_recent=1)
    messages = _messages(50)

    outcome = await compressor.compress_messages(messages)

    assert outcome is None


async def test_fraction_trigger_type_fires_when_ratio_exceeds_value() -> None:
    compressor = _make_compressor(
        trigger_type="fraction", trigger_value=0.01, keep_recent=1, model_max_input_tokens=100,
    )
    compressor._summarize = AsyncMock(return_value="摘要内容")
    messages = _messages(20)

    outcome = await compressor.compress_messages(messages)

    assert outcome is not None
