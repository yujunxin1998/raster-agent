"""`/conversations` REST 接口：会话 CRUD + 历史消息查询。

逻辑对齐原项目 `api/router/conversation.py`，DAO 调用换成本仓库新写的
`conversation_store`/`message_store`。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.agent_core.workspace import get_thread_workspace_manager
from src.common.constants import WorkspaceDirectory
from src.common.exceptions import PathTraversalError
from src.common.response import ApiResponse, success
from src.config.settings import get_settings
from src.schema.conversation_schema import (
    ConversationCreate,
    ConversationHistory,
    ConversationResponse,
    ConversationUpdate,
    FileItem,
    MessageItem,
)
from src.storage.checkpoint_cleanup import get_checkpoint_cleanup
from src.storage.conversation_store import get_conversation_store
from src.storage.file_store import get_file_store
from src.storage.message_store import get_message_store
from src.utils.uuid_utils import generate_uuid

router = APIRouter(prefix="/conversations")

_NOT_FOUND_MESSAGE = "对话不存在或无权访问"
_VISIBLE_MESSAGE_ROLES = ("user", "assistant")
_FILE_NOT_FOUND_MESSAGE = "文件不存在或无权访问"


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
    await get_checkpoint_cleanup().delete_for_thread(conversation_id)
    return success(None)


@router.get("/{conversation_id}/outputs/{file_path:path}", summary="下载会话产物文件")
async def download_output_file(conversation_id: str, user_id: str, file_path: str) -> FileResponse:
    """下载 `save_output_file` 工具保存到本次会话 outputs 目录下的产物文件。

    对应设计文档 5.2 节"outputs/ 下的产物需要有对应的静态文件服务/下载接口
    暴露给前端"——按"生成 URL 挂进现有 Markdown 正文"的方式兼容，不需要前端
    改动：工具返回文本里直接带这个接口的相对路径，前端 Markdown 渲染器本来
    就会把它渲成可点击链接。
    """
    conversation = await get_conversation_store().get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    workspace = get_thread_workspace_manager().get_or_create(conversation_id, user_id)
    try:
        real_path = workspace.resolve(file_path, WorkspaceDirectory.OUTPUTS)
    except PathTraversalError:
        raise HTTPException(status_code=400, detail="非法路径")

    if not real_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(real_path, filename=real_path.name)


@router.post(
    "/{conversation_id}/uploads",
    response_model=ApiResponse[FileItem],
    summary="上传一个文件到会话附件区",
)
async def upload_file(conversation_id: str, user_id: str, file: UploadFile = File(...)) -> ApiResponse:
    """接收 multipart 上传，落盘到该会话的 uploads/ 目录并记录元数据。

    会话不存在时按 chat 接口同样的"首次访问即建会话"约定自动创建，
    不强制要求先调用 `POST /conversations/` ——聊天附件场景下前端往往是
    "选完文件立刻上传"，可能先于会话正式创建。
    """
    settings = get_settings()
    original_name = file.filename or "未命名文件"
    extension = Path(original_name).suffix.lower()
    allowed = {ext.strip().lower() for ext in settings.UPLOAD_ALLOWED_EXTENSIONS.split(",") if ext.strip()}
    if allowed and extension not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {extension or '(无扩展名)'}")

    content = await file.read()
    if len(content) > settings.UPLOAD_MAX_FILE_BYTES:
        max_mb = settings.UPLOAD_MAX_FILE_BYTES / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"文件过大，单个文件不能超过 {max_mb:.0f}MB")

    conversation_store = get_conversation_store()
    if not await conversation_store.exists(conversation_id):
        await conversation_store.create(conversation_id, user_id)

    file_id = generate_uuid()
    stored_name = f"{file_id}_{original_name}"

    workspace = get_thread_workspace_manager().get_or_create(conversation_id, user_id)
    workspace.write_upload(stored_name, content)

    row = await get_file_store().add_file(
        file_id=file_id, conversation_id=conversation_id, user_id=user_id,
        original_name=original_name, stored_name=stored_name,
        content_type=file.content_type or "application/octet-stream", size_bytes=len(content),
    )
    return success(FileItem(**row))


@router.get(
    "/{conversation_id}/uploads",
    response_model=ApiResponse[list[FileItem]],
    summary="列出会话已上传的全部文件",
)
async def list_uploaded_files(conversation_id: str, user_id: str) -> ApiResponse:
    """列出某会话下的全部上传文件。"""
    conversation = await get_conversation_store().get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    rows = await get_file_store().list_files(conversation_id)
    return success([FileItem(**row) for row in rows])


@router.get("/{conversation_id}/uploads/{file_id}", summary="下载会话中已上传的文件")
async def download_uploaded_file(conversation_id: str, user_id: str, file_id: str) -> FileResponse:
    """下载用户此前上传到该会话的文件（按原始文件名回传）。"""
    conversation = await get_conversation_store().get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_MESSAGE)

    row = await get_file_store().get_file(file_id, conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail=_FILE_NOT_FOUND_MESSAGE)

    workspace = get_thread_workspace_manager().get_or_create(conversation_id, user_id)
    try:
        real_path = workspace.resolve(row["stored_name"], WorkspaceDirectory.UPLOADS)
    except PathTraversalError:
        raise HTTPException(status_code=400, detail="非法路径")

    if not real_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(real_path, filename=row["original_name"], media_type=row["content_type"])
