"""管理 WebSocket 连接生命周期，以及前后端工具调用的异步队列。

原样迁移自 `diit-agent-server` 的 `src/api/websocket/connection_manager.py`，
逻辑不变。工具调用流程：

    1. Agent 调用 `send_tool_call_and_wait_response()` 发送指令给前端。
    2. 主循环收到前端 tool response 后调用 `put_tool_response()` 放入队列。
    3. `send_tool_call_and_wait_response()` 从队列取出结果并返回给 Agent。
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import WebSocket
from loguru import logger

_DEFAULT_TOOL_RESPONSE_TIMEOUT_SECONDS = 15


class ConnectionManager:
    """WebSocket 连接 + 前端工具调用回环的管理器。"""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._tool_queues: dict[str, asyncio.Queue] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """接受一个新连接，注册连接和工具响应队列。"""
        await websocket.accept()
        self._connections[client_id] = websocket
        self._tool_queues[client_id] = asyncio.Queue()
        logger.info(f"[WS] 连接建立 client_id={client_id}")

    def disconnect(self, client_id: str) -> None:
        """注销一个连接（不存在也不报错）。"""
        self._connections.pop(client_id, None)
        self._tool_queues.pop(client_id, None)
        logger.info(f"[WS] 连接断开 client_id={client_id}")

    async def send_json(self, client_id: str, data: dict) -> None:
        """向指定客户端发送一条 JSON 消息；客户端不存在时静默跳过。"""
        websocket = self._connections.get(client_id)
        if websocket:
            await websocket.send_text(json.dumps(data, ensure_ascii=False))

    async def put_tool_response(self, client_id: str, tool_result: dict) -> None:
        """前端工具执行完毕后，将结果放入队列。"""
        queue = self._tool_queues.get(client_id)
        if queue:
            await queue.put(tool_result)

    async def send_tool_call_and_wait_response(
        self,
        client_id: str,
        tool_name: str,
        tool_args: dict,
        conversation_id: str,
        timeout: int = _DEFAULT_TOOL_RESPONSE_TIMEOUT_SECONDS,
    ) -> dict:
        """发送工具调用指令给前端，阻塞等待前端返回结果。

        由 `CustomToolConverter` 生成的 `tool_execute` 协程调用。

        Args:
            client_id: 目标客户端连接 ID。
            tool_name: 工具名。
            tool_args: 工具调用参数。
            conversation_id: 归属会话 ID。
            timeout: 等待前端响应的超时秒数，默认 15s。

        Returns:
            前端返回的结果字典（`{"isError", "message", "result"}`）。

        Raises:
            RuntimeError: 客户端不存在或已断开。
            TimeoutError: 超过 `timeout` 秒未收到前端响应。
        """
        queue = self._tool_queues.get(client_id)
        if not queue:
            raise RuntimeError(f"客户端 {client_id} 不存在或已断开")

        # 清空旧队列，防止上一次残留响应干扰
        while not queue.empty():
            queue.get_nowait()

        request_id = f"{conversation_id}_{tool_name}_{int(time.time() * 1000)}"
        await self.send_json(client_id, {
            "role": "agent",
            "type": "tool",
            "action": "call",
            "conversation_id": conversation_id,
            "content": {"tool_name": tool_name, "tool_args": tool_args, "request_id": request_id},
        })
        logger.info(f"[ToolCall] 已发送 tool={tool_name} request_id={request_id}")

        result = await asyncio.wait_for(queue.get(), timeout=timeout)
        logger.info(f"[ToolCall] 收到响应 tool={tool_name} isError={result.get('isError')}")
        return result


# 全局单例，由 chat_ws.py 和 CustomToolConverter 共同使用。
manager = ConnectionManager()
