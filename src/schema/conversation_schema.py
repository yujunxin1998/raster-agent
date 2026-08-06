"""`/conversations` REST API 的请求/响应模型。

字段与原项目 `src/schema/conversation.py` 保持一致，含 `ReferenceSource.id`/
`.desc` 默认空字符串的历史兼容处理（原注释：早期 web_search 来源的消息从未
填充过这两个字段，收紧校验会导致读取历史消息时报 500）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    """`POST /conversations/` 请求体。"""

    user_id: str = Field(..., description="用户ID")
    title: Optional[str] = Field(None, description="会话标题，不传则使用默认标题")


class ConversationResponse(BaseModel):
    """会话详情响应体，字段与 `conversations` 表列一一对应。"""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationUpdate(BaseModel):
    """`PUT /conversations/{id}` 请求体。"""

    title: str = Field(..., min_length=1, max_length=50, description="新标题")


class ToolCallRecord(BaseModel):
    """一次工具调用的持久化记录（调用 + 响应合并成一条）。"""

    tool_name: str
    tool_args: dict
    request_id: str
    tool_type: str = ""
    tool_response: Optional[str] = None


class ReferenceSource(BaseModel):
    """一条引用来源。"""

    index: Optional[int] = None
    id: str = ""
    type: str = "file"
    name: str
    desc: str = ""
    url: str = ""
    score: Optional[float] = None


class MessageItem(BaseModel):
    """会话历史里的一条消息。"""

    role: str = Field(..., description="user | assistant | system")
    content: str
    thinking_content: Optional[str] = None
    tool_calls: Optional[list[ToolCallRecord]] = None
    references: Optional[list[ReferenceSource]] = None


class ConversationHistory(BaseModel):
    """`GET /conversations/{id}/messages` 响应体。"""

    conversation_id: str
    messages: list[MessageItem]


class FileItem(BaseModel):
    """一条上传文件记录，字段与 `conversation_files` 表列一一对应。"""

    id: str
    original_name: str
    content_type: str
    size_bytes: int
    created_at: datetime
