"""WebSocket 流式对话接口：`/ws/chat`。

协议对齐原项目 `diit-agent-server` 的 `src/api/websocket/chat.py`（客户端→服务端
的 `chat`/`tool response` 两类消息，服务端→客户端的
`start/token/thinking/tool_call/skill_call/tool_response/skill_response/
reference/done/error` 事件），驱动逻辑改为共用
`agent_core/agents/chat_pipeline.py::run_chat_turn()`——本文件只负责协议解析、
转发事件、落库，不再手写事件消费循环。
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from src.agent_core.agents.chat_pipeline import ChatTurnResult, run_chat_turn, run_regenerate_turn
from src.agent_core.eval.emitter import EvalEventEmitter
from src.agent_core.eval.http_middleware import emit_ws_chat_record
from src.agent_core.tools.custom_tool_converter import CustomToolConverter
from src.api.websocket.connection_manager import manager
from src.storage.conversation_store import get_conversation_store
from src.storage.message_store import get_message_store
from src.utils.uuid_utils import generate_uuid

router = APIRouter()

_DEFAULT_USER_ID = "default"


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket 流式对话接口，支持前端自定义工具。

    客户端 → 服务端：
        对话消息 `{"role":"agent","type":"chat","action":"send","user_id":?,
        "conversation_id":?,"content":{"message":str,"thinking":bool,
        "tool_list":[...]?,"datasource_id":str?}}`
        重新生成 `{"role":"agent","type":"chat","action":"regenerate","user_id":?,
        "conversation_id":"uuid","content":{"thinking":bool}}`——撤回该会话
        最后一轮 AI 回复并重新生成，不追加新的用户发言（见
        `chat_pipeline.py::run_regenerate_turn`）。
        工具执行结果 `{"role":"agent","type":"tool","action":"response",
        "conversation_id":"uuid","content":{"tool_result":{"isError":bool,
        "message":str,"result":{...}}}}`

    服务端 → 客户端：`start`/`token`/`thinking`/`tool_call`/`skill_call`/
        `tool_response`/`skill_response`/`reference`/`done`/`error`。`done`
        额外携带 `message_id`（本轮 assistant 消息在 `conversation_messages`
        表里的自增 ID），供前端关联点赞/点踩反馈。
    """
    client_id = generate_uuid()
    await manager.connect(websocket, client_id)

    user_id = _DEFAULT_USER_ID

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(client_id, {"type": "error", "message": "消息须为合法 JSON"})
                continue

            msg_type = payload.get("type", "")
            msg_action = payload.get("action", "")

            if msg_type == "tool" and msg_action == "response":
                tool_result = payload.get("content", {}).get("tool_result", {})
                await manager.put_tool_response(client_id, tool_result)
                continue

            if msg_type == "chat" and msg_action == "regenerate":
                content = payload.get("content", {})
                conversation_id = payload.get("conversation_id") or content.get("conversation_id")
                if not conversation_id:
                    await manager.send_json(client_id, {"type": "error", "message": "conversation_id 不能为空"})
                    continue
                thinking = bool(content.get("thinking", False))
                datasource_id = content.get("datasource_id")
                user_id = payload.get("user_id") or content.get("user_id") or user_id
                asyncio.create_task(_handle_regenerate(client_id, conversation_id, thinking, datasource_id, user_id))
                continue

            content = payload.get("content", {})
            message = content.get("message", "").strip()
            if not message:
                await manager.send_json(client_id, {"type": "error", "message": "message 不能为空"})
                continue

            thinking = bool(content.get("thinking", False))
            tool_list = content.get("tool_list")
            datasource_id = content.get("datasource_id")
            file_ids = content.get("file_ids")

            conversation_id = payload.get("conversation_id") or content.get("conversation_id") or generate_uuid()
            user_id = payload.get("user_id") or content.get("user_id") or user_id

            extra_tools = None
            if tool_list:
                extra_tools = CustomToolConverter.merge_tools(
                    original_tools=[], tool_list=tool_list, manager=manager,
                    client_id=client_id, conversation_id=conversation_id,
                )

            asyncio.create_task(
                _handle_chat(
                    client_id, conversation_id, message, thinking, extra_tools, user_id, datasource_id, file_ids,
                )
            )

    except WebSocketDisconnect:
        logger.info(f"[WS] 客户端主动断开 client_id={client_id}")
    except Exception as exc:
        logger.error(f"[WS] 未预期异常 client_id={client_id}: {exc}")
    finally:
        manager.disconnect(client_id)


async def _handle_chat(
    client_id: str,
    conversation_id: str,
    message: str,
    thinking: bool,
    extra_tools,
    user_id: str,
    datasource_id: str | None,
    file_ids: list[str] | None = None,
) -> None:
    await manager.send_json(client_id, {"type": "start", "conversation_id": conversation_id})

    conversation_store = get_conversation_store()
    message_store = get_message_store()

    is_new = not await conversation_store.exists(conversation_id)
    if is_new:
        await conversation_store.create(conversation_id, user_id)

    trace_id = generate_uuid()
    start_ms = int(time.time() * 1000)
    emitter = EvalEventEmitter(trace_id=trace_id, conversation_id=conversation_id, user_id=user_id, user_input=message)

    try:
        turn_result = ChatTurnResult()
        async for event in run_chat_turn(
            conversation_id=conversation_id, user_id=user_id, message=message, thinking=thinking,
            datasource_id=datasource_id, file_ids=file_ids, extra_tools=extra_tools,
            emitter=emitter, result=turn_result,
        ):
            # 前端现在允许多个会话同时流式生成（同一个 WS 连接上跑多个
            # `_handle_chat` 任务），每个事件必须带上归属的 conversation_id，
            # 否则前端没法区分这条 token 该写进哪个会话的消息列表——这是支持
            # "切到另一个会话继续发消息、原会话在后台继续生成"的前提。
            await manager.send_json(client_id, {**event, "conversation_id": conversation_id})

        await message_store.add_message(conversation_id, "user", message)
        assistant_message_id = await message_store.add_message(
            conversation_id, "assistant", turn_result.ai_response, turn_result.thinking_content,
            turn_result.tool_call_records or None, turn_result.references or None,
        )
        await conversation_store.touch(conversation_id)

        await manager.send_json(
            client_id, {"type": "done", "conversation_id": conversation_id, "message_id": assistant_message_id},
        )

        asyncio.create_task(emitter.flush_trace(turn_result.ai_response, task_status="success"))
        await emit_ws_chat_record(
            trace_id=trace_id, conversation_id=conversation_id, latency_ms=int(time.time() * 1000) - start_ms,
        )

    except Exception as exc:
        logger.exception(f"[WS] 流式响应异常 client_id={client_id}: {exc}")
        asyncio.create_task(emitter.flush_trace("", task_status="failed", error_message=str(exc)))
        await manager.send_json(client_id, {"type": "error", "message": str(exc), "conversation_id": conversation_id})


async def _handle_regenerate(
    client_id: str,
    conversation_id: str,
    thinking: bool,
    datasource_id: str | None,
    user_id: str,
) -> None:
    """撤回某会话最后一轮 AI 回复并重新生成，驱动逻辑见
    `chat_pipeline.py::run_regenerate_turn()`。
    """
    await manager.send_json(client_id, {"type": "start", "conversation_id": conversation_id})

    conversation_store = get_conversation_store()
    message_store = get_message_store()

    if not await conversation_store.exists(conversation_id):
        await manager.send_json(
            client_id, {"type": "error", "message": "会话不存在，无法重新生成", "conversation_id": conversation_id},
        )
        return

    trace_id = generate_uuid()
    start_ms = int(time.time() * 1000)
    emitter = EvalEventEmitter(
        trace_id=trace_id, conversation_id=conversation_id, user_id=user_id, user_input="[regenerate]",
    )

    try:
        turn_result = ChatTurnResult()
        async for event in run_regenerate_turn(
            conversation_id=conversation_id, user_id=user_id, thinking=thinking,
            datasource_id=datasource_id, emitter=emitter, result=turn_result,
        ):
            await manager.send_json(client_id, {**event, "conversation_id": conversation_id})

        if not turn_result.ai_response:
            # 没有可重新生成的内容（会话里还没有用户发言），直接收口，不落库。
            await manager.send_json(client_id, {"type": "done", "conversation_id": conversation_id})
            return

        await message_store.delete_last_assistant_message(conversation_id)
        assistant_message_id = await message_store.add_message(
            conversation_id, "assistant", turn_result.ai_response, turn_result.thinking_content,
            turn_result.tool_call_records or None, turn_result.references or None,
        )
        await conversation_store.touch(conversation_id)

        await manager.send_json(
            client_id, {"type": "done", "conversation_id": conversation_id, "message_id": assistant_message_id},
        )

        asyncio.create_task(emitter.flush_trace(turn_result.ai_response, task_status="success"))
        await emit_ws_chat_record(
            trace_id=trace_id, conversation_id=conversation_id, latency_ms=int(time.time() * 1000) - start_ms,
        )

    except Exception as exc:
        logger.exception(f"[WS] 重新生成异常 client_id={client_id}: {exc}")
        asyncio.create_task(emitter.flush_trace("", task_status="failed", error_message=str(exc)))
        await manager.send_json(client_id, {"type": "error", "message": str(exc), "conversation_id": conversation_id})
