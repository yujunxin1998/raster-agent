"""`/chat` REST + WebSocket API 的请求/响应模型。

字段与原项目 `src/schema/chat.py` 保持一致，确保 `diit-agent-web` 相关 api
模块日后对接时无需改动字段名。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

_MAX_MESSAGE_LENGTH = 65536


class ChatRequest(BaseModel):
    """`POST /chat/` 请求体。"""

    conversation_id: Optional[str] = Field(None, description="会话ID，为空时服务端生成新会话")
    user_id: Optional[str] = Field(None, description="用户ID，为空时回退默认用户")
    message: str = Field("", max_length=_MAX_MESSAGE_LENGTH, description="用户发言内容")
    thinking: bool = Field(False, description="是否开启深度思考模式")
    datasource_id: Optional[str] = Field(None, description="数据源ID")


class ChatResponse(BaseModel):
    """`POST /chat/` 响应体（`ApiResponse.data`）。"""

    conversation_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="AI 回复正文")
    thinking_content: Optional[str] = Field(None, description="深度思考模式下的推理过程文本")


class WSChatRequest(BaseModel):
    """WS `content` 字段的载荷（`type=="chat"`）。"""

    conversation_id: Optional[str] = Field(None, description="会话ID，为空时服务端生成新会话")
    user_id: Optional[str] = Field(None, description="用户ID，为空时沿用连接上一次的值")
    message: str = Field(..., max_length=_MAX_MESSAGE_LENGTH, description="用户发言内容")
    thinking: bool = Field(False, description="是否开启深度思考模式")
    tool_list: Optional[list[dict[str, Any]]] = Field(None, description="前端注入的自定义工具列表（MCP JSON Schema 格式）")
    datasource_id: Optional[str] = Field(None, description="数据源ID")


class WSEvent(BaseModel):
    """服务端推给客户端的事件（`start | token | thinking | tool_call | tool_response | reference | done | error`）。"""

    type: str = Field(..., description="事件类型")
    content: Optional[Any] = Field(None, description="事件载荷，形状随 type 而变")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    message: Optional[str] = Field(None, description="错误说明（type=='error' 时使用）")
