"""长期记忆管理 REST API。

Memory v2：现有端点的路径、参数、响应结构保持与原项目 `src/api/router/memory.py`
完全一致（确保 `diit-agent-web` 的 `memoryApi.js` 无需任何改动即可对接），内部
实现改为对接 `UserMemoryFactStore`（PostgreSQL 规范化主存，写入立即强一致，
不再需要旧版本因 ES 最终一致性而引入的 `get_memory_with_retry` 重试读）。
新增的 pending 审核、画像字段来源、导出/彻底删除是新增端点，不影响旧端点行为。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.agent_core.memory import get_memory_manager
from src.agent_core.memory.elasticsearch_memory_store import get_elasticsearch_memory_store
from src.common.response import ApiResponse, success
from src.schema.memory_schema import (
    MemoryAuditLogResponse,
    MemoryCreate,
    MemoryResponse,
    MemorySearchQuery,
    MemoryUpdate,
    ProfileFieldSourceResponse,
    UserProfileResponse,
)
from src.storage.memory_audit_store import get_memory_audit_store
from src.storage.user_memory_fact_store import get_user_memory_fact_store
from src.storage.user_profile_store import get_user_profile_store

router = APIRouter(prefix="/memories")

_DEFAULT_UNAVAILABLE_MESSAGE = "记忆服务暂不可用，请稍后重试"


def _dependency_unavailable(detail: str = _DEFAULT_UNAVAILABLE_MESSAGE) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


def _to_memory_response(fact: dict) -> MemoryResponse:
    """把 `user_memory_fact` 行映射为对外的 `MemoryResponse`（字段名保持旧契约）。"""
    return MemoryResponse(
        id=str(fact["id"] if "id" in fact else fact["fact_id"]),
        content=fact["content"],
        memory_type=fact.get("memory_type") or fact.get("category"),
        importance=fact["importance"],
        created_at=_iso(fact.get("created_at")),
        status=fact.get("status", "active"),
        expires_at=_iso(fact.get("expires_at")),
        last_accessed_at=_iso(fact.get("last_accessed_at")),
        access_count=fact.get("access_count") or 0,
        source="extractor" if fact.get("source_event_id") else "api",
        conversation_id=fact.get("conversation_id") or fact.get("source_conversation_id"),
        agent_name=fact.get("agent_name"),
    )


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


@router.get("/profile", response_model=ApiResponse[UserProfileResponse], summary="查询用户画像与时间线")
async def get_user_profile(user_id: str) -> ApiResponse:
    """查询用户画像与时间线（三层记忆架构的 L1/L2，只读）。

    还没生成过画像时返回全空字段而不是 404——这是新用户的正常状态，不是异常。
    """
    try:
        profile = await get_memory_manager().get_profile(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get user profile failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc

    if profile is None:
        return success(UserProfileResponse())

    updated_at = profile.get("updated_at")
    return success(UserProfileResponse(
        work_context=profile.get("work_context", ""),
        personal_context=profile.get("personal_context", ""),
        top_of_mind=profile.get("top_of_mind", ""),
        recent_months=profile.get("recent_months", ""),
        earlier_context=profile.get("earlier_context", ""),
        long_term_background=profile.get("long_term_background", ""),
        updated_at=updated_at.isoformat() if updated_at else None,
    ))


@router.get(
    "/profile/field-sources", response_model=ApiResponse[list[ProfileFieldSourceResponse]],
    summary="查询画像各字段的最近更新来源（Memory v2 新增）",
)
async def get_profile_field_sources(user_id: str) -> ApiResponse:
    """查询用户画像各字段最近一次更新的来源事件与置信度，供人工审核追溯依据。"""
    try:
        rows = await get_user_profile_store().get_field_sources(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] get profile field sources failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([ProfileFieldSourceResponse(**row) for row in rows])


@router.get("/pending", response_model=ApiResponse[list[MemoryResponse]], summary="列出待人工确认的记忆（Memory v2 新增）")
async def list_pending_memories(user_id: str) -> ApiResponse:
    """列出指定用户所有 `status=pending` 的 Fact，供人工确认/拒绝。"""
    try:
        rows = await get_user_memory_fact_store().list_pending(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] list pending memories failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([_to_memory_response(row) for row in rows])


@router.post("/{memory_id}/confirm", response_model=ApiResponse[None], summary="确认一条待审核记忆（Memory v2 新增）")
async def confirm_pending_memory(memory_id: str, user_id: str) -> ApiResponse:
    """把一条 `pending` Fact 转为 `active`。"""
    try:
        ok = await get_user_memory_fact_store().confirm_pending(memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] confirm memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if not ok:
        raise HTTPException(status_code=404, detail="记忆不存在、无权访问或不处于待确认状态")
    return success(msg="已确认")


@router.post("/{memory_id}/reject", response_model=ApiResponse[None], summary="拒绝一条待审核记忆（Memory v2 新增）")
async def reject_pending_memory(memory_id: str, user_id: str) -> ApiResponse:
    """把一条 `pending` Fact 直接归档。"""
    try:
        ok = await get_user_memory_fact_store().reject_pending(memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] reject memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if not ok:
        raise HTTPException(status_code=404, detail="记忆不存在、无权访问或不处于待确认状态")
    return success(msg="已拒绝")


@router.get("/", response_model=ApiResponse[list[MemoryResponse]], summary="列出用户所有长期记忆")
async def list_memories(user_id: str) -> ApiResponse:
    """列出指定用户的全部长期记忆。"""
    try:
        rows = await get_user_memory_fact_store().list_by_user(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] list memories failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([_to_memory_response(row) for row in rows])


@router.post("/search", response_model=ApiResponse[list[MemoryResponse]], summary="按语义检索长期记忆")
async def search_memories(user_id: str, body: MemorySearchQuery) -> ApiResponse:
    """按语义相似度检索指定用户的长期记忆（走 ES 检索投影）。"""
    try:
        rows = await get_elasticsearch_memory_store().search(query=body.query, user_id=user_id, top_n=body.top_n)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] search memories failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success([_to_memory_response(row) for row in rows])


@router.post("/", response_model=ApiResponse[MemoryResponse], summary="新增一条长期记忆")
async def create_memory(body: MemoryCreate) -> ApiResponse:
    """通过管理 API 新增一条长期记忆（人工直接写入，跳过 Delta/LLM 判断）。"""
    manager = get_memory_manager()
    if manager.is_sensitive_content(body.content):
        raise HTTPException(status_code=400, detail="记忆保存失败：内容包含敏感信息")

    try:
        result = await get_user_memory_fact_store().add_or_reinforce(
            user_id=body.user_id, content=body.content, category=body.memory_type,
            importance=body.importance, confidence=1.0, status="active",
            source_conversation_id=body.conversation_id,
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
        )
    except Exception as exc:
        logger.warning(f"[MemoryAPI] create memory failed user_id={body.user_id}: {exc}")
        raise _dependency_unavailable() from exc

    row = await get_user_memory_fact_store().get(result["fact_id"], body.user_id)
    if row is None:
        raise _dependency_unavailable("记忆已写入，但暂时无法读取，请稍后重试")
    return success(_to_memory_response(row))


@router.put("/{memory_id}", response_model=ApiResponse[MemoryResponse], summary="编辑一条长期记忆")
async def update_memory(memory_id: str, user_id: str, body: MemoryUpdate) -> ApiResponse:
    """编辑一条长期记忆（管理 API 可以把任意状态互相改写，`allow_any_status=True`）。"""
    fact_store = get_user_memory_fact_store()
    existing = await fact_store.get(memory_id, user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")

    manager = get_memory_manager()
    if body.content and manager.is_sensitive_content(body.content):
        raise HTTPException(status_code=400, detail="记忆更新失败：新内容包含敏感信息")

    try:
        ok = await fact_store.update(
            memory_id, user_id, content=body.content, category=body.memory_type,
            importance=body.importance, status=body.status,
            expires_at=body.expires_at.isoformat() if body.expires_at else None,
            # expires_at 显式传 null（而不是不传）才代表"清除过期时间"，而不是"不修改"
            clear_expires_at="expires_at" in body.model_fields_set and body.expires_at is None,
            allow_any_status=True,
        )
    except Exception as exc:
        logger.warning(f"[MemoryAPI] update memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc

    if not ok:
        raise _dependency_unavailable("记忆更新失败：记忆服务暂不可用")

    updated = await fact_store.get(memory_id, user_id)
    if updated is None:
        raise _dependency_unavailable("记忆已更新，但暂时无法读取，请稍后重试")
    return success(_to_memory_response(updated))


@router.delete("/purge", response_model=ApiResponse[None], summary="彻底删除用户全部长期记忆（Memory v2 新增）")
async def purge_user_memory(user_id: str) -> ApiResponse:
    """彻底删除用户的全部 Facts + 画像（不可恢复，设计文档 §9.2）。

    注册顺序注意：必须排在 `DELETE /{memory_id}` 之前——两者都是 DELETE +
    单段路径，FastAPI 按注册顺序匹配，`/{memory_id}` 在前会把字面量 "purge"
    当成 memory_id 吞掉，导致这个端点永远不可达。
    """
    try:
        deleted = await get_user_memory_fact_store().purge_user(user_id)
        await get_user_profile_store().delete(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] purge memory failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc
    return success(msg=f"已彻底删除 {deleted} 条记忆及画像数据")


@router.delete("/{memory_id}", response_model=ApiResponse[None], summary="删除一条长期记忆")
async def delete_memory(memory_id: str, user_id: str) -> ApiResponse:
    """彻底删除一条长期记忆。"""
    fact_store = get_user_memory_fact_store()
    try:
        ok = await fact_store.delete(memory_id, user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] delete memory failed id={memory_id}: {exc}")
        raise _dependency_unavailable() from exc
    if not ok:
        raise HTTPException(status_code=404, detail="记忆不存在或无权访问")
    return success(msg="已删除")


@router.get("/{memory_id}/audit-logs", response_model=ApiResponse[list[MemoryAuditLogResponse]], summary="查看一条记忆的操作审计记录")
async def get_memory_audit_logs(memory_id: str) -> ApiResponse:
    """查询一条记忆的全部操作审计记录。"""
    audit_store = get_memory_audit_store()
    if audit_store is None:
        raise _dependency_unavailable("记忆审计服务暂不可用，请稍后重试")
    rows = await audit_store.list_by_memory(memory_id)
    return success([MemoryAuditLogResponse(**row) for row in rows])


@router.post("/export", response_model=ApiResponse[dict], summary="导出用户全部长期记忆（Memory v2 新增）")
async def export_user_memory(user_id: str) -> ApiResponse:
    """导出用户的全部 Facts + 画像，供数据导出/合规场景使用（设计文档 §9.2）。"""
    try:
        facts = await get_user_memory_fact_store().list_by_user(user_id, limit=10000)
        profile = await get_memory_manager().get_profile(user_id)
    except Exception as exc:
        logger.warning(f"[MemoryAPI] export memory failed user_id={user_id}: {exc}")
        raise _dependency_unavailable() from exc

    return success({
        "facts": [_to_memory_response(fact).model_dump() for fact in facts],
        "profile": profile or {},
    })
