"""长期记忆管理 REST API。

路径、参数、响应结构与原项目 `src/api/router/memory.py` 完全一致，确保
`diit-agent-web` 的 `memoryApi.js` 无需任何改动即可对接本工程。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.agent_core.memory import get_memory_manager, get_memory_with_retry
from src.common.response import ApiResponse, success
from src.schema.memory_schema import (
    MemoryAuditLogResponse,
    MemoryCreate,
    MemoryResponse,
    MemorySearchQuery,
    MemoryUpdate,
)
from src.storage.memory_audit_store import get_memory_audit_store

router = APIRouter(prefix="/memories")

_DEFAULT_UNAVAILABLE_MESSAGE = "记忆服务暂不可用，请稍后重试"


def _dependency_unavailable(detail: str = _DEFAULT_UNAVAILABLE_MESSAGE) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


@router.get("/", response_model=ApiResponse[list[MemoryResponse]], summary="列出用户所有长期记忆")
async def list_memories(user_id: str) -> ApiResponse:
    """列出指定用户的全部长期记忆。"""
    try:
        rows = await get_memory_manager().store.list_all(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] list memories failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([MemoryResponse(**row) for row in rows])


@router.post("/search", response_model=ApiResponse[list[MemoryResponse]], summary="按语义检索长期记忆")
async def search_memories(user_id: str, body: MemorySearchQuery) -> ApiResponse:
    """按语义相似度检索指定用户的长期记忆。"""
    try:
        rows = await get_memory_manager().store.search(query=body.query, user_id=user_id, top_n=body.top_n)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] search memories failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([MemoryResponse(**row) for row in rows])


@router.post("/", response_model=ApiResponse[MemoryResponse], summary="新增一条长期记忆")
async def create_memory(body: MemoryCreate) -> ApiResponse:
    """通过管理 API 新增一条长期记忆。"""
    manager = get_memory_manager()
    try:
        memory_id = await manager.store.save(
            content=body.content, user_id=body.user_id, conversation_id=body.conversation_id,
            memory_type=body.memory_type, importance=body.importance, source="api",
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
        )
    except Exception as exc:
        logger.warning(f"[MemoryAPI] create memory failed user_id={body.user_id}: {exc}")
        raise _dependency_unavailable() from exc

    if not memory_id:
        if manager.is_sensitive_content(body.content):
            raise HTTPException(status_code=400, detail="记忆保存失败：内容包含敏感信息")
        raise _dependency_unavailable()

    try:
        row = await get_memory_with_retry(manager.store, memory_id, body.user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get created memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if row is None:
        raise _dependency_unavailable("记忆已写入，但暂时无法读取，请稍后重试")
    return success(MemoryResponse(**row))


@router.put("/{memory_id}", response_model=ApiResponse[MemoryResponse], summary="编辑一条长期记忆")
async def update_memory(memory_id: str, user_id: str, body: MemoryUpdate) -> ApiResponse:
    """编辑一条长期记忆。"""
    manager = get_memory_manager()
    try:
        existing = await manager.store.get(memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get memory before update failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if existing is None:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")

    try:
        ok = await manager.store.update(
            memory_id, user_id=user_id, content=body.content, memory_type=body.memory_type,
            importance=body.importance, source="api", status=body.status,
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
            # expires_at 显式传 null（而不是不传）才代表"清除过期时间"，而不是"不修改"
            clear_expires_at="expires_at" in body.model_fields_set and body.expires_at is None,
        )
    except Exception as exc:
        logger.warning(f"[MemoryAPI] update memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc

    if not ok:
        if manager.is_sensitive_content(body.content):
            raise HTTPException(status_code=400, detail="记忆更新失败：新内容包含敏感信息")
        raise _dependency_unavailable("记忆更新失败：记忆服务暂不可用")

    try:
        updated = await get_memory_with_retry(manager.store, memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get updated memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if updated is None:
        raise _dependency_unavailable("记忆已更新，但暂时无法读取，请稍后重试")
    return success(MemoryResponse(**updated))


@router.delete("/{memory_id}", response_model=ApiResponse[None], summary="删除一条长期记忆")
async def delete_memory(memory_id: str, user_id: str) -> ApiResponse:
    """删除一条长期记忆。"""
    manager = get_memory_manager()
    try:
        existing = await manager.store.get(memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get memory before delete failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if existing is None:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")

    try:
        ok = await manager.store.delete(memory_id, user_id=user_id, source="api")
    except Exception as exc:
        logger.warning(f"[MemoryAPI] delete memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if not ok:
        raise _dependency_unavailable("记忆删除失败：记忆服务暂不可用")
    return success(msg="已删除")


@router.get("/{memory_id}/audit-logs", response_model=ApiResponse[list[MemoryAuditLogResponse]], summary="查看一条记忆的操作审计记录")
async def get_memory_audit_logs(memory_id: str) -> ApiResponse:
    """查询一条记忆的全部操作审计记录。"""
    audit_store = get_memory_audit_store()
    if audit_store is None:
        raise _dependency_unavailable("记忆审计服务暂不可用，请稍后重试")
    rows = await audit_store.list_by_memory(memory_id)
    return success([MemoryAuditLogResponse(**row) for row in rows])
