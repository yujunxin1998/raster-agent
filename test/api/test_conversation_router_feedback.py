"""`PUT /conversations/{id}/messages/{message_id}/feedback` 单元测试。

策略与 `test_conversation_router_uploads.py` 一致：mock `conversation_store`/
`message_store`，不依赖真实数据库连接。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import conversation_router


def _client(conversation_store, message_store):
    app = FastAPI()
    app.include_router(conversation_router.router)
    p1 = patch("src.api.router.conversation_router.get_conversation_store", return_value=conversation_store)
    p2 = patch("src.api.router.conversation_router.get_message_store", return_value=message_store)
    return TestClient(app), p1, p2


def test_set_feedback_success() -> None:
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    message_store = AsyncMock()
    message_store.set_feedback = AsyncMock(return_value=True)

    client, p1, p2 = _client(conversation_store, message_store)
    with p1, p2:
        response = client.put(
            "/conversations/conv-1/messages/42/feedback",
            params={"user_id": "user-1"},
            json={"feedback": "like"},
        )

    assert response.status_code == 200
    message_store.set_feedback.assert_awaited_once_with(42, "conv-1", "like")


def test_set_feedback_null_clears_reaction() -> None:
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    message_store = AsyncMock()
    message_store.set_feedback = AsyncMock(return_value=True)

    client, p1, p2 = _client(conversation_store, message_store)
    with p1, p2:
        response = client.put(
            "/conversations/conv-1/messages/42/feedback",
            params={"user_id": "user-1"},
            json={"feedback": None},
        )

    assert response.status_code == 200
    message_store.set_feedback.assert_awaited_once_with(42, "conv-1", None)


def test_set_feedback_rejects_invalid_value() -> None:
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    message_store = AsyncMock()

    client, p1, p2 = _client(conversation_store, message_store)
    with p1, p2:
        response = client.put(
            "/conversations/conv-1/messages/42/feedback",
            params={"user_id": "user-1"},
            json={"feedback": "love-it"},
        )

    assert response.status_code == 400
    message_store.set_feedback.assert_not_called()


def test_set_feedback_404_when_conversation_not_owned() -> None:
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value=None)
    message_store = AsyncMock()

    client, p1, p2 = _client(conversation_store, message_store)
    with p1, p2:
        response = client.put(
            "/conversations/conv-1/messages/42/feedback",
            params={"user_id": "user-1"},
            json={"feedback": "like"},
        )

    assert response.status_code == 404
    message_store.set_feedback.assert_not_called()


def test_set_feedback_404_when_message_not_found() -> None:
    conversation_store = AsyncMock()
    conversation_store.get = AsyncMock(return_value={"id": "conv-1", "user_id": "user-1"})
    message_store = AsyncMock()
    message_store.set_feedback = AsyncMock(return_value=False)

    client, p1, p2 = _client(conversation_store, message_store)
    with p1, p2:
        response = client.put(
            "/conversations/conv-1/messages/999/feedback",
            params={"user_id": "user-1"},
            json={"feedback": "dislike"},
        )

    assert response.status_code == 404
