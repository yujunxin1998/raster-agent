"""`ConnectionManager` 单元测试：不依赖真实 WebSocket 连接。"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.websocket.connection_manager import ConnectionManager


def _make_websocket() -> MagicMock:
    websocket = MagicMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    return websocket


async def test_connect_registers_connection_and_queue() -> None:
    manager = ConnectionManager()
    websocket = _make_websocket()

    await manager.connect(websocket, "client-1")

    websocket.accept.assert_awaited_once()
    assert "client-1" in manager._connections
    assert "client-1" in manager._tool_queues


async def test_disconnect_removes_connection_without_error_if_missing() -> None:
    manager = ConnectionManager()
    manager.disconnect("never-connected")  # 不应抛异常


async def test_send_json_serializes_and_sends() -> None:
    manager = ConnectionManager()
    websocket = _make_websocket()
    await manager.connect(websocket, "client-1")

    await manager.send_json("client-1", {"type": "token", "content": "你好"})

    websocket.send_text.assert_awaited_once()
    sent_raw = websocket.send_text.await_args.args[0]
    assert json.loads(sent_raw) == {"type": "token", "content": "你好"}


async def test_send_tool_call_and_wait_response_returns_frontend_result() -> None:
    manager = ConnectionManager()
    websocket = _make_websocket()
    await manager.connect(websocket, "client-1")

    async def _respond_after_delay() -> None:
        await asyncio.sleep(0.01)
        await manager.put_tool_response("client-1", {"isError": False, "result": {"ok": True}})

    asyncio.create_task(_respond_after_delay())
    result = await manager.send_tool_call_and_wait_response(
        "client-1", "map_add_layer", {"url": "x"}, "conv-1", timeout=1,
    )

    assert result == {"isError": False, "result": {"ok": True}}


async def test_send_tool_call_times_out_when_no_response() -> None:
    manager = ConnectionManager()
    websocket = _make_websocket()
    await manager.connect(websocket, "client-1")

    with pytest.raises(asyncio.TimeoutError):
        await manager.send_tool_call_and_wait_response("client-1", "map_add_layer", {}, "conv-1", timeout=0.05)


async def test_stale_queue_response_is_drained_before_new_call() -> None:
    manager = ConnectionManager()
    websocket = _make_websocket()
    await manager.connect(websocket, "client-1")

    # 上一次调用遗留的响应
    await manager.put_tool_response("client-1", {"isError": False, "result": "stale"})

    async def _respond_after_delay() -> None:
        await asyncio.sleep(0.01)
        await manager.put_tool_response("client-1", {"isError": False, "result": "fresh"})

    asyncio.create_task(_respond_after_delay())
    result = await manager.send_tool_call_and_wait_response("client-1", "tool", {}, "conv-1", timeout=1)

    assert result == {"isError": False, "result": "fresh"}


async def test_unknown_client_raises_runtime_error() -> None:
    manager = ConnectionManager()
    with pytest.raises(RuntimeError):
        await manager.send_tool_call_and_wait_response("ghost-client", "tool", {}, "conv-1")
