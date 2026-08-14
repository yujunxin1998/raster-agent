"""`save_memory`/`recall_memory` 工具单元测试。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.memory_tools import recall_memory, save_memory

_RUNTIME = SimpleNamespace(context=AgentRuntimeContext(conversation_id="conv-1", user_id="user-1"))


def _settings(**overrides) -> MagicMock:
    defaults = {"MEMORY_ENABLED": True, "MEMORY_TOOL_ENABLED": True, "MEMORY_MIN_RECALL_SCORE": 0.3}
    defaults.update(overrides)
    return MagicMock(**defaults)


async def test_save_memory_disabled_returns_message() -> None:
    with patch("src.agent_core.tools.memory_tools.get_settings",
               return_value=_settings(MEMORY_ENABLED=False)):
        result = await save_memory.coroutine(content="用户偏好使用 Vim", runtime=_RUNTIME)

    assert "已关闭" in result


async def test_save_memory_rejects_sensitive_content() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=True)
    audit_store = MagicMock()
    audit_store.record = AsyncMock()

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_memory_manager", return_value=manager), \
         patch("src.agent_core.tools.memory_tools.get_memory_audit_store", return_value=audit_store):
        result = await save_memory.coroutine(content="api_key=sk-abcdefghijklmnop1234", runtime=_RUNTIME)

    assert "敏感信息" in result
    audit_store.record.assert_awaited_once()
    assert audit_store.record.await_args.kwargs["action"].value == "save_rejected"


async def test_save_memory_writes_via_fact_store() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=False)
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "added", "status": "active"})
    audit_store = MagicMock()
    audit_store.record = AsyncMock()

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_memory_manager", return_value=manager), \
         patch("src.agent_core.tools.memory_tools.get_user_memory_fact_store", return_value=fact_store), \
         patch("src.agent_core.tools.memory_tools.get_memory_audit_store", return_value=audit_store):
        result = await save_memory.coroutine(
            content="用户偏好使用 Vim", memory_type="preference", importance=6, runtime=_RUNTIME,
        )

    assert "已保存记忆" in result
    kwargs = fact_store.add_or_reinforce.await_args.kwargs
    assert kwargs["user_id"] == "user-1"
    assert kwargs["category"] == "preference"
    assert kwargs["source_conversation_id"] == "conv-1"
    assert audit_store.record.await_args.kwargs["action"].value == "create"


async def test_save_memory_reports_reinforced_action_in_audit() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=False)
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(return_value={"fact_id": "f-1", "action": "reinforced", "status": "active"})
    audit_store = MagicMock()
    audit_store.record = AsyncMock()

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_memory_manager", return_value=manager), \
         patch("src.agent_core.tools.memory_tools.get_user_memory_fact_store", return_value=fact_store), \
         patch("src.agent_core.tools.memory_tools.get_memory_audit_store", return_value=audit_store):
        await save_memory.coroutine(content="用户偏好使用 Vim", runtime=_RUNTIME)

    assert audit_store.record.await_args.kwargs["action"].value == "fact_reinforced"


async def test_save_memory_handles_store_failure() -> None:
    manager = MagicMock()
    manager.is_sensitive_content = MagicMock(return_value=False)
    fact_store = MagicMock()
    fact_store.add_or_reinforce = AsyncMock(side_effect=Exception("db down"))

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_memory_manager", return_value=manager), \
         patch("src.agent_core.tools.memory_tools.get_user_memory_fact_store", return_value=fact_store):
        result = await save_memory.coroutine(content="用户偏好使用 Vim", runtime=_RUNTIME)

    assert "记忆服务暂不可用" in result


async def test_recall_memory_disabled_returns_message() -> None:
    with patch("src.agent_core.tools.memory_tools.get_settings",
               return_value=_settings(MEMORY_TOOL_ENABLED=False)):
        result = await recall_memory.coroutine(query="喜欢什么编辑器", runtime=_RUNTIME)

    assert "已关闭" in result


async def test_recall_memory_returns_no_result_message_when_empty() -> None:
    memory_store = MagicMock()
    memory_store.search = AsyncMock(return_value=[])

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_elasticsearch_memory_store", return_value=memory_store):
        result = await recall_memory.coroutine(query="喜欢什么编辑器", runtime=_RUNTIME)

    assert "未找到" in result


async def test_recall_memory_filters_by_min_score_and_records_audit() -> None:
    memory_store = MagicMock()
    memory_store.search = AsyncMock(return_value=[
        {"id": "f-1", "memory_type": "preference", "content": "用户偏好使用 Vim", "score": 0.9, "created_at": "2026-08-13"},
        {"id": "f-2", "memory_type": "context", "content": "低相关", "score": 0.1, "created_at": "2026-08-13"},
    ])
    audit_store = MagicMock()
    audit_store.record = AsyncMock()

    with patch("src.agent_core.tools.memory_tools.get_settings", return_value=_settings()), \
         patch("src.agent_core.tools.memory_tools.get_elasticsearch_memory_store", return_value=memory_store), \
         patch("src.agent_core.tools.memory_tools.get_memory_audit_store", return_value=audit_store):
        result = await recall_memory.coroutine(query="喜欢什么编辑器", runtime=_RUNTIME)

    assert "用户偏好使用 Vim" in result
    assert "低相关" not in result
    audit_store.record.assert_awaited_once()
    assert audit_store.record.await_args.kwargs["action"].value == "recall"
