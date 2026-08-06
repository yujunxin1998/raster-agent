"""`chat_pipeline._build_message_with_attachments` 单元测试。

对应文件上传功能：本轮携带 `file_ids` 时，把附件文件名/大小/下载路径拼进
用户发言末尾，供模型感知本轮上传了哪些文件。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agent_core.agents.chat_pipeline import _build_message_with_attachments


@pytest.mark.asyncio
async def test_no_file_ids_returns_original_message() -> None:
    result = await _build_message_with_attachments("conv-1", "你好", None)
    assert result == "你好"


@pytest.mark.asyncio
async def test_empty_file_ids_returns_original_message() -> None:
    result = await _build_message_with_attachments("conv-1", "你好", [])
    assert result == "你好"


@pytest.mark.asyncio
async def test_file_ids_appends_attachment_summary() -> None:
    file_store = AsyncMock()
    file_store.get_files_by_ids = AsyncMock(return_value=[{
        "id": "file-1", "original_name": "report.pdf", "content_type": "application/pdf",
        "size_bytes": 2048,
    }])

    with patch("src.agent_core.agents.chat_pipeline.get_file_store", return_value=file_store):
        result = await _build_message_with_attachments("conv-1", "帮我看看这个文件", ["file-1"])

    assert result.startswith("帮我看看这个文件")
    assert "report.pdf" in result
    assert "2.0KB" in result
    assert "/conversations/conv-1/uploads/file-1" in result
    file_store.get_files_by_ids.assert_awaited_once_with(["file-1"], "conv-1")


@pytest.mark.asyncio
async def test_unknown_file_ids_returns_original_message() -> None:
    file_store = AsyncMock()
    file_store.get_files_by_ids = AsyncMock(return_value=[])

    with patch("src.agent_core.agents.chat_pipeline.get_file_store", return_value=file_store):
        result = await _build_message_with_attachments("conv-1", "你好", ["missing-id"])

    assert result == "你好"
