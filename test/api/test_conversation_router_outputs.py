"""`/conversations/{id}/outputs/{file_path}` 下载接口单元测试。

对应设计文档 5.2 节"outputs/ 下的产物需要有对应的静态文件服务/下载接口暴露
给前端"——用真实 `ThreadWorkspaceManager` + tmp_path 落一个真实文件，只 mock
`conversation_store`（owner 校验）,不依赖真实数据库连接。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager
from src.api.router import conversation_router


def test_download_output_file_returns_content(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    workspace = manager.get_or_create("conv-1", "user-1")
    (workspace.outputs_dir / "report.md").write_text("# 结果", encoding="utf-8")

    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ):
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/outputs/report.md", params={"user_id": "user-1"})

    assert response.status_code == 200
    assert response.text == "# 结果"


def test_download_output_file_404_when_conversation_not_owned(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value=None)

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ):
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/outputs/report.md", params={"user_id": "user-1"})

    assert response.status_code == 404


def test_download_output_file_404_when_file_missing(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    manager.get_or_create("conv-1", "user-1")
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ):
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get("/conversations/conv-1/outputs/missing.md", params={"user_id": "user-1"})

    assert response.status_code == 404


def test_download_output_file_400_on_path_traversal(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))
    manager.get_or_create("conv-1", "user-1")
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})

    with patch(
        "src.api.router.conversation_router.get_thread_workspace_manager", return_value=manager
    ), patch(
        "src.api.router.conversation_router.get_conversation_store", return_value=conversation_store
    ):
        app = FastAPI()
        app.include_router(conversation_router.router)
        client = TestClient(app)
        response = client.get(
            "/conversations/conv-1/outputs/../../workspace/secret.txt", params={"user_id": "user-1"}
        )

    assert response.status_code in (400, 404)
