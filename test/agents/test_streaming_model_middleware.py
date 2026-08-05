"""`StreamingModelMiddleware` 的 tool_calls 补发逻辑单元测试。

复现真实 bug 场景：`stream_mode="messages"` 通道里的增量 `AIMessageChunk`
在参数较长时（如 `write_file` 的 `content`）要跨多个 chunk 才拼出完整
`tool_calls`，中途每个 chunk 的 `tool_calls` 都是半成品（`args={}`，第二块
起 `name`/`id` 还是空）。`chat_pipeline.py` 曾经直接读这条通道触发
`tool_call` 事件，参数越长，推给前端的半成品事件越多——这是"任务卡在写文件
这一步不动"的真实根因。修复后 `StreamingModelMiddleware` 应该在自己已经拼完
`final_message` 之后，把解析完整的 `tool_calls` 通过 `stream_writer` 补发一次。
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from src.agent_core.agents.streaming_model_middleware import StreamingModelMiddleware


class _FakeBoundModel:
    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self._chunks = chunks

    async def astream(self, messages):
        for chunk in self._chunks:
            yield chunk


def _fake_request(chunks: list[AIMessageChunk], stream_writer) -> MagicMock:
    model = MagicMock()
    model.bind.return_value = _FakeBoundModel(chunks)
    model.bind_tools.return_value = _FakeBoundModel(chunks)

    request = MagicMock()
    request.model = model
    request.tools = []
    request.tool_choice = None
    request.model_settings = {}
    request.messages = []
    request.system_message = None
    request.runtime.stream_writer = stream_writer
    return request


async def test_final_tool_calls_are_rebroadcast_once_after_accumulation() -> None:
    # 模拟一次跨两块才拼完的 tool_call（第一块只有 name + 半截 JSON 参数）。
    chunk1 = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "write_file", "args": '{"path": ', "id": "call_1", "index": 0}]
    )
    chunk2 = AIMessageChunk(
        content="", tool_call_chunks=[{"name": None, "args": '"a.py"}', "id": None, "index": 0}]
    )
    written: list[Any] = []
    request = _fake_request([chunk1, chunk2], stream_writer=written.append)

    response = await StreamingModelMiddleware().awrap_model_call(request, handler=MagicMock())

    # 原始 chunk 各推一次（打字机效果）；name+id 在 chunk1 就齐了，紧跟着补发一次
    # "仅名字" pending 标记；最后再补发一次完整 tool_calls。
    assert len(written) == 4
    assert written[0] is chunk1
    assert written[1] == {"tool_call_pending": {"id": "call_1", "name": "write_file"}}
    assert written[2] is chunk2
    assert written[3] == {
        "tool_calls": [{"name": "write_file", "args": {"path": "a.py"}, "id": "call_1", "type": "tool_call"}]
    }

    final_message = response.result[0]
    assert final_message.tool_calls == [
        {"name": "write_file", "args": {"path": "a.py"}, "id": "call_1", "type": "tool_call"}
    ]


async def test_pending_marker_broadcast_only_once_per_tool_call_id() -> None:
    # 三块都属于同一个 tool_call_id：第一块之后 id/name 不会再出现，
    # pending 标记只该广播一次，不能每块都发。
    chunk1 = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "write_file", "args": "{", "id": "call_1", "index": 0}]
    )
    chunk2 = AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": '"a"', "id": None, "index": 0}])
    chunk3 = AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": "}", "id": None, "index": 0}])
    written: list[Any] = []
    request = _fake_request([chunk1, chunk2, chunk3], stream_writer=written.append)

    await StreamingModelMiddleware().awrap_model_call(request, handler=MagicMock())

    pending_writes = [w for w in written if isinstance(w, dict) and "tool_call_pending" in w]
    assert pending_writes == [{"tool_call_pending": {"id": "call_1", "name": "write_file"}}]


async def test_no_extra_broadcast_when_turn_has_no_tool_calls() -> None:
    chunk = AIMessageChunk(content="你好")
    written: list[Any] = []
    request = _fake_request([chunk], stream_writer=written.append)

    await StreamingModelMiddleware().awrap_model_call(request, handler=MagicMock())

    assert written == [chunk]  # 没有工具调用，不应该补发任何东西
