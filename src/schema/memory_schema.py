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
    memory_type: str = Field("context", description="preference | knowledge | context | behavior | goal")
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


class UserProfileResponse(BaseModel):
    """用户画像与时间线响应体（三层记忆架构的 L1/L2，只读）。"""

    work_context: str = Field("", description="职业角色、公司、关键项目、主力技术栈")
    personal_context: str = Field("", description="语言能力、沟通偏好、兴趣领域")
    top_of_mind: str = Field("", description="当前关注的多个并行焦点，更新频率最高")
    recent_months: str = Field("", description="近 1-3 个月的详细活动摘要")
    earlier_context: str = Field("", description="3-12 个月前的重要模式")
    long_term_background: str = Field("", description="长期不变的基础背景")
    updated_at: Optional[str] = Field(None, description="最近一次更新时间，还没生成过画像时为 None")


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


class ProfileFieldSourceResponse(BaseModel):
    """画像单个字段的最近更新来源（Memory v2 新增，人工闭环 §9.2）。"""

    field_name: str
    updated_at: datetime
    source_event_id: Optional[str] = Field(None, description="触发这次更新的 memory_event ID")
    confidence: float
