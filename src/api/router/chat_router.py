"""`/chat` REST 接口（设计文档 4.1 节 Lead Agent 落地的第一个入口）。

非流式：内部消费共享的 `chat_pipeline.run_chat_turn()`（与 WebSocket
`/ws/chat` 共用同一条驱动逻辑，见该模块说明），丢弃逐条 yield 的事件，只读
最终累计的 `ChatTurnResult`。记忆注入/提取/压缩/标题生成已经是 Lead Agent
中间件流水线的职责，这里只需要：解析会话归属、驱动一次对话轮次、把结果落库。
"""
from __future__ import annotations

from fastapi import APIRouter, Response
from loguru import logger

from src.agent_core.agents.chat_pipeline import ChatTurnResult, run_chat_turn
from src.agent_core.eval.emitter import EvalEventEmitter
from src.common.response import ApiResponse, success
from src.schema.chat_schema import ChatRequest, ChatResponse
from src.storage.conversation_store import get_conversation_store
from src.storage.message_store import get_message_store
from src.utils.user_utils import resolve_user_id
from src.utils.uuid_utils import generate_uuid

router = APIRouter(prefix="/chat")


@router.post("/", response_model=ApiResponse[ChatResponse], summary="发送一条对话消息")
async def send_chat(chat_request: ChatRequest, response: Response) -> ApiResponse:
    """处理一次非流式对话请求。

    Args:
        chat_request: 请求体，见 `ChatRequest`。
        response: FastAPI 响应对象，用于写入 `X-Conversation-Id`/`X-Trace-Id` 响应头
            （`EvalMiddleware` 靠这两个头做关联，见 `agent_core/eval/http_middleware.py`）。

    Returns:
        `ApiResponse[ChatResponse]`。
    """
    conversation_id = chat_request.conversation_id or generate_uuid()
    user_id = resolve_user_id(chat_request.user_id)
    trace_id = generate_uuid()

    conversation_store = get_conversation_store()
    message_store = get_message_store()

    if not await conversation_store.exists(conversation_id):
        await conversation_store.create(conversation_id, user_id)

    emitter = EvalEventEmitter(
        trace_id=trace_id, conversation_id=conversation_id, user_id=user_id, user_input=chat_request.message,
    )

    turn_result = ChatTurnResult()
    async for _event in run_chat_turn(
        conversation_id=conversation_id, user_id=user_id, message=chat_request.message,
        thinking=chat_request.thinking, datasource_id=chat_request.datasource_id,
        file_ids=chat_request.file_ids, emitter=emitter, result=turn_result,
    ):
        pass  # REST 接口不流式返回，丢弃逐条事件，只读最终的 turn_result

    await emitter.flush_trace(turn_result.ai_response, task_status="success")

    await message_store.add_message(conversation_id, "user", chat_request.message)
    await message_store.add_message(
        conversation_id, "assistant", turn_result.ai_response, turn_result.thinking_content,
        turn_result.tool_call_records or None, turn_result.references or None,
    )
    await conversation_store.touch(conversation_id)

    response.headers["X-Conversation-Id"] = conversation_id
    response.headers["X-Trace-Id"] = trace_id

    logger.info(f"[chat_router] 对话完成 conversation_id={conversation_id} user_id={user_id}")

    return success(
        ChatResponse(
            conversation_id=conversation_id, message=turn_result.ai_response,
            thinking_content=turn_result.thinking_content,
        )
    )
