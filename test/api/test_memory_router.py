"""`memory_router` 单元测试：mock 各 Store 单例，不依赖真实数据库/ES 连接。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import memory_router

_FACT_ROW = {
    "fact_id": "f-1", "user_id": "user-1", "agent_name": None,
    "content": "用户偏好使用 Vim", "category": "preference", "importance": 7,
    "confidence": 0.9, "status": "active", "source_event_id": None,
    "source_conversation_id": "conv-1", "created_at": "2026-08-13T00:00:00",
    "expires_at": None, "last_accessed_at": None, "access_count": 0,
}


def _client(**stores):
    app = FastAPI()
    app.include_router(memory_router.router)
    patches = []
    getter_map = {
        "memory_manager": "get_memory_manager",
        "fact_store": "get_user_memory_fact_store",
        "profile_store": "get_user_profile_store",
        "es_store": "get_elasticsearch_memory_store",
        "audit_store": "get_memory_audit_store",
    }
    for key, getter in getter_map.items():
        if key in stores:
            patches.append(patch(f"src.api.router.memory_router.{getter}", return_value=stores[key]))
    return TestClient(app), patches


def _run(client, patches, method, path, **kwargs):
    for p in patches:
        p.start()
    try:
        return getattr(client, method)(path, **kwargs)
    finally:
        for p in patches:
            p.stop()


def test_get_user_profile_returns_empty_for_new_user() -> None:
    manager = MagicMock()
    manager.get_profile = AsyncMock(return_value=None)
    client, patches = _client(memory_manager=manager)

    response = _run(client, patches, "get", "/memories/profile", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.json()["data"]["work_context"] == ""


def test_get_profile_field_sources() -> None:
    profile_store = MagicMock()
    profile_store.get_field_sources = AsyncMock(return_value=[
        {"field_name": "work_context", "updated_at": "2026-08-13T00:00:00", "source_event_id": "evt-1", "confidence": 0.9},
    ])
    client, patches = _client(profile_store=profile_store)

    response = _run(client, patches, "get", "/memories/profile/field-sources", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.json()["data"][0]["field_name"] == "work_context"


def test_list_pending_memories() -> None:
    fact_store = MagicMock()
    fact_store.list_pending = AsyncMock(return_value=[{**_FACT_ROW, "status": "pending"}])
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "get", "/memories/pending", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.json()["data"][0]["status"] == "pending"


def test_confirm_pending_memory_success() -> None:
    fact_store = MagicMock()
    fact_store.confirm_pending = AsyncMock(return_value=True)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "post", "/memories/f-1/confirm", params={"user_id": "user-1"})

    assert response.status_code == 200
    fact_store.confirm_pending.assert_awaited_once_with("f-1", "user-1")


def test_confirm_pending_memory_not_found() -> None:
    fact_store = MagicMock()
    fact_store.confirm_pending = AsyncMock(return_value=False)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "post", "/memories/f-1/confirm", params={"user_id": "user-1"})

    assert response.status_code == 404


def test_reject_pending_memory_success() -> None:
    fact_store = MagicMock()
    fact_store.reject_pending = AsyncMock(return_value=True)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "post", "/memories/f-1/reject", params={"user_id": "user-1"})

    assert response.status_code == 200


def test_list_memories() -> None:
    fact_store = MagicMock()
    fact_store.list_by_user = AsyncMock(return_value=[_FACT_ROW])
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "get", "/memories/", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.json()["data"][0]["memory_type"] == "preference"


def test_search_memories_uses_es_store() -> None:
    es_store = MagicMock()
    es_store.search = AsyncMock(return_value=[
        {"id": "f-1", "content": "用户偏好使用 Vim", "memory_type": "preference", "importance": 7,
         "created_at": "2026-08-13T00:00:00", "status": "active"},
    ])
    client, patches = _client(es_store=es_store)

    response = _run(client, patches, "post", "/memories/search", params={"user_id": "user-1"}, json={"query": "编辑器"})

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "f-1"


def test_create_memory_rejects_sensitive_content() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=True)
    client, patches = _client(memory_manager=manager)

    response = _run(client, patches, "post", "/memories/", json={"user_id": "user-1", "content": "sk-abcdefghijklmnop1234"})

    assert response.status_code == 400


def test_create_memory_success() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=False)
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "active"})
    fact_store.get = AsyncMock(return_value=_FACT_ROW)
    client, patches = _client(memory_manager=manager, fact_store=fact_store)

    response = _run(client, patches, "post", "/memories/", json={
        "user_id": "user-1", "content": "用户偏好使用 Vim", "memory_type": "preference", "importance": 7,
    })

    assert response.status_code == 200
    assert response.json()["data"]["content"] == "用户偏好使用 Vim"


def test_update_memory_not_found() -> None:
    fact_store = MagicMock()
    fact_store.get = AsyncMock(return_value=None)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "put", "/memories/f-1", params={"user_id": "user-1"}, json={"importance": 9})

    assert response.status_code == 404


def test_update_memory_success() -> None:
    fact_store = MagicMock()
    fact_store.get = AsyncMock(side_effect=[_FACT_ROW, {**_FACT_ROW, "importance": 9}])
    fact_store.update = AsyncMock(return_value=True)
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=False)
    client, patches = _client(fact_store=fact_store, memory_manager=manager)

    response = _run(client, patches, "put", "/memories/f-1", params={"user_id": "user-1"}, json={"importance": 9})

    assert response.status_code == 200
    assert response.json()["data"]["importance"] == 9
    assert fact_store.update.await_args.kwargs["allow_any_status"] is True


def test_delete_memory_not_found() -> None:
    fact_store = MagicMock()
    fact_store.delete = AsyncMock(return_value=False)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "delete", "/memories/f-1", params={"user_id": "user-1"})

    assert response.status_code == 404


def test_delete_memory_success() -> None:
    fact_store = MagicMock()
    fact_store.delete = AsyncMock(return_value=True)
    client, patches = _client(fact_store=fact_store)

    response = _run(client, patches, "delete", "/memories/f-1", params={"user_id": "user-1"})

    assert response.status_code == 200


def test_purge_route_is_not_shadowed_by_dynamic_memory_id_route() -> None:
    """回归测试：`DELETE /purge` 必须先于 `DELETE /{memory_id}` 注册，否则 "purge"
    会被当成 memory_id 吞掉，永远匹配到 delete_memory 而不是 purge_user_memory。
    """
    fact_store = MagicMock()
    fact_store.purge_user = AsyncMock(return_value=3)
    fact_store.delete = AsyncMock(return_value=True)  # 如果路由错误命中这里，delete 会被调用
    profile_store = MagicMock()
    profile_store.delete = AsyncMock()
    client, patches = _client(fact_store=fact_store, profile_store=profile_store)

    response = _run(client, patches, "delete", "/memories/purge", params={"user_id": "user-1"})

    assert response.status_code == 200
    fact_store.purge_user.assert_awaited_once_with("user-1")
    fact_store.delete.assert_not_awaited()
    profile_store.delete.assert_awaited_once_with("user-1")


def test_export_user_memory() -> None:
    fact_store = MagicMock()
    fact_store.list_by_user = AsyncMock(return_value=[_FACT_ROW])
    manager = MagicMock()
    manager.get_profile = AsyncMock(return_value={"work_context": "后端工程师"})
    client, patches = _client(fact_store=fact_store, memory_manager=manager)

    response = _run(client, patches, "post", "/memories/export", params={"user_id": "user-1"})

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["facts"]) == 1
    assert data["profile"]["work_context"] == "后端工程师"


def test_get_memory_audit_logs() -> None:
    audit_store = MagicMock()
    audit_store.list_by_memory = AsyncMock(return_value=[{
        "id": 1, "memory_id": "f-1", "user_id": "user-1", "conversation_id": None,
        "trace_id": None, "action": "create", "source": "api", "detail": None,
        "created_at": "2026-08-13T00:00:00",
    }])
    client, patches = _client(audit_store=audit_store)

    response = _run(client, patches, "get", "/memories/f-1/audit-logs")

    assert response.status_code == 200
    assert response.json()["data"][0]["action"] == "create"


def test_get_memory_audit_logs_unavailable_when_store_missing() -> None:
    client, patches = _client(audit_store=None)

    response = _run(client, patches, "get", "/memories/f-1/audit-logs")

    assert response.status_code == 503
