"""`UserProfileUpdater` 单元测试：mock LLM 调用，不依赖真实模型服务。"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent_core.memory.user_profile_updater import UserProfileUpdater

_FULL_PROFILE = {
    "work_context": "后端工程师", "personal_context": "偏好中文", "top_of_mind": "记忆模块重构",
    "recent_months": "", "earlier_context": "", "long_term_background": "",
}


def _updater() -> UserProfileUpdater:
    return UserProfileUpdater(model_name="test-model", provider="deepseek", api_key="k", base_url="u")


def _fake_store(existing: dict | None) -> MagicMock:
    store = MagicMock()
    store.get = AsyncMock(return_value=existing)
    store.upsert = AsyncMock()
    return store


async def test_update_after_chat_upserts_parsed_result() -> None:
    updater = _updater()
    updater._call_llm_update = AsyncMock(return_value={
        "work_context": "高级后端工程师", "personal_context": "偏好中文", "top_of_mind": "记忆模块重构上线",
        "recent_months": "重构了记忆模块", "earlier_context": "", "long_term_background": "",
    })
    store = _fake_store(_FULL_PROFILE)

    await updater.update_after_chat(store, user_message="我升职了", ai_response="恭喜", user_id="u1")

    store.upsert.assert_awaited_once_with(
        "u1",
        work_context="高级后端工程师", personal_context="偏好中文", top_of_mind="记忆模块重构上线",
        recent_months="重构了记忆模块", earlier_context="", long_term_background="",
    )


async def test_update_after_chat_uses_empty_defaults_for_new_user() -> None:
    updater = _updater()
    captured_prompt = {}

    async def _capture(prompt):
        captured_prompt["value"] = prompt
        return dict(_FULL_PROFILE)

    updater._call_llm_update = _capture
    store = _fake_store(None)  # 全新用户，还没有画像记录

    await updater.update_after_chat(store, user_message="你好", ai_response="你好", user_id="u1")

    store.get.assert_awaited_once_with("u1")
    assert "职业背景（work_context）：" in captured_prompt["value"]


async def test_update_after_chat_skips_upsert_when_llm_fails() -> None:
    updater = _updater()
    updater._call_llm_update = AsyncMock(return_value=None)
    store = _fake_store(_FULL_PROFILE)

    await updater.update_after_chat(store, user_message="你好", ai_response="你好", user_id="u1")

    store.upsert.assert_not_awaited()


async def test_call_llm_update_parses_plain_json() -> None:
    updater = _updater()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"work_context": "wc"}'))

    with patch("src.agent_core.memory.user_profile_updater.create_chat_model", return_value=llm):
        result = await updater._call_llm_update("prompt")

    assert result["work_context"] == "wc"
    assert result["top_of_mind"] == ""  # 缺失字段兜底为空字符串


async def test_call_llm_update_strips_code_fence() -> None:
    updater = _updater()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='```json\n{"work_context": "wc"}\n```'))

    with patch("src.agent_core.memory.user_profile_updater.create_chat_model", return_value=llm):
        result = await updater._call_llm_update("prompt")

    assert result["work_context"] == "wc"


async def test_call_llm_update_returns_none_on_llm_exception() -> None:
    updater = _updater()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=Exception("timeout"))

    with patch("src.agent_core.memory.user_profile_updater.create_chat_model", return_value=llm):
        result = await updater._call_llm_update("prompt")

    assert result is None


async def test_call_llm_update_returns_none_on_non_dict_json() -> None:
    updater = _updater()
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='["not", "a", "dict"]'))

    with patch("src.agent_core.memory.user_profile_updater.create_chat_model", return_value=llm):
        result = await updater._call_llm_update("prompt")

    assert result is None
