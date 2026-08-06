"""`/conversations/{id}/uploads` 上传/列表/下载接口单元测试。

与 `test_conversation_router_outputs.py` 同样的策略：真实
`ThreadWorkspaceManager`（tmp_path）+ mock `conversation_store`/`file_store`，
不依赖真实数据库连接。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager
from src.api.router import conversation_router


def _patched(manager, conversation_store, file_store):
    return (
        patch("src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager),
        patch("src.api.router.conversation_router.get_conversation_store", return_value=conversation_store),
        patch("src.api.router.conversation_router.get_file_store", return_value=file_store),
    )


def test_upload_file_saves_to_uploads_dir_and_returns_metadata(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.exists = AsyncMock(return_value=True)
    file_store = AsyncMock()
    file_store.add_file = AsyncMock(
        side_effect=lambda **kwargs: {
            "id": "file-1", "conversation_id": kwargs["conversation_id"], "user_id": kwargs["user_id"],
            "original_name": kwargs["original_name"], "stored_name": kwargs["stored_name"],
            "content_type": kwargs["content_type"], "size_bytes": kwargs["size_bytes"],
            "created_at": "2026-08-06T00:00:00+00:00",
        }
    )

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ), patch(
        "src.api.router.conversation_router.get_file_store", return_value=file_store
    ):
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.post(
            "/conversations/conv-1/uploads",
            params={"user_id": "user-1"},
            files={"file": ("report.txt", b"hello world", "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["original_name"] == "report.txt"
    assert body["data"]["size_bytes"] == len(b"hello world")

    workspace = manager.get_or_create("conv-1", "user-1")
    uploaded = list(workspace.uploads_dir.iterdir())
    assert len(uploaded) == 1
    assert uploaded[0].read_bytes() == b"hello world"

    file_store.add_file.assert_awaited_once()


def test_upload_file_rejects_disallowed_extension(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.exists = AsyncMock(return_value=True)
    file_store = AsyncMock()

    p1, p2, p3 = _patched(manager, conversation_store, file_store)
    with p1, p2, p3:
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.post(
            "/conversations/conv-1/uploads",
            params={"user_id": "user-1"},
            files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
        )

    assert response.status_code == 400
    file_store.add_file.assert_not_called()


def test_upload_file_rejects_oversize(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.exists = AsyncMock(return_value=True)
    file_store = AsyncMock()

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ), patch(
        "src.api.router.conversation_router.get_file_store", return_value=file_store
    ), patch(
        "src.api.router.conversation_router.get_settings"
    ) as mock_settings:
        mock_settings.return_value.UPLOAD_MAX_FILE_BYTES = 4
        mock_settings.return_value.UPLOAD_ALLOWED_EXTENSIONS = ".txt"

        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.post(
            "/conversations/conv-1/uploads",
            params={"user_id": "user-1"},
            files={"file": ("report.txt", b"this is too long", "text/plain")},
        )

    assert response.status_code == 400
    file_store.add_file.assert_not_called()


def test_list_uploaded_files_returns_records(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    file_store = AsyncMock()
    file_store.list_files = AsyncMock(return_value=[{
        "id": "file-1", "original_name": "a.txt", "content_type": "text/plain",
        "size_bytes": 5, "created_at": "2026-08-06T00:00:00+00:00",
    }])

    p1, p2, p3 = _patched(manager, conversation_store, file_store)
    with p1, p2, p3:
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/uploads", params={"user_id": "user-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["id"] == "file-1"


def test_list_uploaded_files_404_when_conversation_not_owned(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value=None)
    file_store = AsyncMock()

    p1, p2, p3 = _patched(manager, conversation_store, file_store)
    with p1, p2, p3:
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/uploads", params={"user_id": "user-1"})

    assert response.status_code == 404


def test_download_uploaded_file_returns_content(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    workspace = manager.get_or_create("conv-1", "user-1")
    workspace.write_upload("file-1_report.txt", b"downloaded content")

    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    file_store = AsyncMock()
    file_store.get_file = AsyncMock(return_value={
        "id": "file-1", "stored_name": "file-1_report.txt",
        "original_name": "report.txt", "content_type": "text/plain",
    })

    p1, p2, p3 = _patched(manager, conversation_store, file_store)
    with p1, p2, p3:
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/uploads/file-1", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.text == "downloaded content"


def test_download_uploaded_file_404_when_not_found(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    file_store = AsyncMock()
    file_store.get_file = AsyncMock(return_value=None)

    p1, p2, p3 = _patched(manager, conversation_store, file_store)
    with p1, p2, p3:
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/uploads/missing-id", params={"user_id": "user-1"})

    assert response.status_code == 404
