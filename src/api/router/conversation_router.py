"""`/conversations` REST 接口：会话 CRUD + 历史消息查询。

逻辑对齐原项目 `api/router/conversation.py`，DAO 调用换成本仓库新写的
`conversation_store`/`message_store`。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.common.response import ApiResponse, success
from src.schema.conversation_schema import (
    ConversationCreate,
    ConversationHistory,
    ConversationResponse,
    ConversationUpdate,
    MessageItem,
)
from src.storage.conversation_store import get_conversation_store
from src.storage.message_store import get_message_store
from src.utils.uuid_utils import generate_uuid

router = APIRouter(prefix="/conversations")

_NOT_FOUND_MESSAGE = "对话不存在或无权访问"
_VISIBLE_MESSAGE_ROLES = ("user", "assistant")


@router.post("/", response_model=ApiResponse[ConversationResponse], summary="创建会话")
async def create_conversation(body: ConversationCreate) -> ApiResponse:
    """新建一条会话记录。"""
    conversation_id = generate_uuid()
    title = body.title or "新对话"
    row = await get_conversation_store().create(conversation_id, body.user_id, title)
    return success(ConversationResponse(**row))


@router.get("/", response_model=ApiResponse[list[ConversationResponse]], summary="列出用户的全部会话")
async def list_conversations(user_id: str) -> ApiResponse:
    """按用户列出全部会话，按最近更新时间倒序。"""
    rows = await get_conversation_store().list_by_user(user_id)
    return success([ConversationResponse(**row) for row in rows])


@router.get(
    "/{conversation_id}/messages",
    response_model=ApiResponse[ConversationHistory],
    summary="查询会话历史消息",
)
async def get_messages(conversation_id: str, user_id: str) -> ApiResponse:
    """查询某会话的历史消息，只返回 user/assistant 角色的消息。"""
    conversation = await get_conversation_store().get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    rows = await get_message_store().get_messages(conversation_id)
    messages = [
        MessageItem(
            role=row["role"], content=row["content"], thinking_content=row["thinking_content"],
            tool_calls=row["tool_calls"], references=row["references"],
        )
        for row in rows
        if row["role"] in _VISIBLE_MESSAGE_ROLES
    ]
    return success(ConversationHistory(conversation_id=conversation_id, messages=messages))


@router.put("/{conversation_id}", response_model=ApiResponse[ConversationResponse], summary="修改会话标题")
async def update_conversation(conversation_id: str, user_id: str, body: ConversationUpdate) -> ApiResponse:
    """修改会话标题。"""
    conversation_store = get_conversation_store()
    conversation = await conversation_store.get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    await conversation_store.update_title(conversation_id, user_id, body.title)
    updated = await conversation_store.get(conversation_id, user_id)
    return success(ConversationResponse(**updated))


@router.delete("/{conversation_id}", response_model=ApiResponse[None], summary="删除会话")
async def delete_conversation(conversation_id: str, user_id: str) -> ApiResponse:
    """删除会话及其历史消息。"""
    conversation_store = get_conversation_store()
    conversation = await conversation_store.get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    await conversation_store.delete(conversation_id, user_id)
    await get_message_store().delete_messages(conversation_id)
    return success(None)
