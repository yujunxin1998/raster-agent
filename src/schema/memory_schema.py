"""长期记忆管理 REST API 的请求/响应模型。

字段与原项目 `src/schema/memory.py` 保持完全一致，确保
`diit-agent-web/src/api/memoryApi.js` 无需任何改动即可对接。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MemoryCreate(BaseModel):
    """新增一条长期记忆的请求体。"""

    user_id: str = Field(..., description="用户ID")
    content: str = Field(..., min_length=1, description="记忆内容")
    memory_type: str = Field("fact", description="fact | preference | decision | instruction | correction")
    importance: int = Field(5, ge=1, le=10, description="重要度 1-10")
    conversation_id: Optional[str] = Field(None, description="来源对话ID（可选）")
    expires_at: Optional[datetime] = Field(None, description="过期时间（可选），过期后不再被召回")


class MemoryUpdate(BaseModel):
    """编辑一条长期记忆的请求体，字段均为可选（不传即不修改）。"""

    content: Optional[str] = Field(None, min_length=1, description="新内容，不传则不修改")
    memory_type: Optional[str] = Field(None, description="新分类，不传则不修改")
    importance: Optional[int] = Field(None, ge=1, le=10, description="新重要度，不传则不修改")
    status: Optional[str] = Field(None, description="active | archived | pending，不传则不修改")
    expires_at: Optional[datetime] = Field(None, description="过期时间，不传则不修改")


class MemoryResponse(BaseModel):
    """长期记忆详情响应体。"""

    id: str = Field(..., description="记忆ID")
    content: str
    memory_type: str
    importance: int
    created_at: str
    status: str = Field("active", description="active | pending | archived | expired")
    expires_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    source: Optional[str] = Field(None, description="创建来源：extractor | tool | api | compressor")
    conversation_id: Optional[str] = Field(None, description="来源对话ID")
    scope_type: str = Field("user", description="记忆作用域，预留字段，当前固定为 user")
    scope_id: Optional[str] = Field(None, description="作用域 ID，预留字段")
    agent_name: Optional[str] = Field(None, description="专属 Agent 名，预留字段")


class MemorySearchQuery(BaseModel):
    """按语义检索长期记忆的请求体。"""

    query: str = Field(..., min_length=1, description="检索语句")
    top_n: int = Field(5, ge=1, le=20, description="返回条数")


class MemoryAuditLogResponse(BaseModel):
    """记忆审计日志响应体。"""

    id: int
    memory_id: Optional[str]
    user_id: str
    conversation_id: Optional[str]
    trace_id: Optional[str]
    action: str
    source: str
    detail: Optional[str]
    created_at: datetime
