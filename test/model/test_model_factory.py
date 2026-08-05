"""`create_chat_model` 的供应商分支单元测试。

不发起真实网络请求——`init_chat_model`/`PatchedChatDeepSeek` 的构造阶段只是
纯 Python 对象组装，不会连接供应商 API。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent_core.model.model_factory import create_chat_model
from src.agent_core.model.patched_deepseek import PatchedChatDeepSeek


def test_deepseek_provider_builds_patched_class_with_thinking_enabled() -> None:
    model = create_chat_model(
        model="deepseek-v4-pro", provider="deepseek", api_key="k", base_url="https://api.deepseek.com",
        thinking_enabled=True,
    )

    assert isinstance(model, PatchedChatDeepSeek)
    assert model.extra_body == {"thinking": {"type": "enabled"}}


def test_deepseek_provider_thinking_disabled_by_default() -> None:
    model = create_chat_model(
        model="deepseek-v4-pro", provider="deepseek", api_key="k", base_url="https://api.deepseek.com",
    )

    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_non_deepseek_provider_falls_back_to_init_chat_model() -> None:
    with patch("src.agent_core.model.model_factory.init_chat_model") as mock_init_chat_model:
        mock_init_chat_model.return_value = MagicMock(name="fake_openai_model")

        result = create_chat_model(model="gpt-4o-mini", provider="openai", api_key="k", base_url="https://api.openai.com/v1")

        mock_init_chat_model.assert_called_once_with(
            model="gpt-4o-mini", model_provider="openai", api_key="k", base_url="https://api.openai.com/v1",
        )
        assert result is mock_init_chat_model.return_value


def test_falls_back_to_settings_when_params_omitted() -> None:
    fake_settings = MagicMock(
        DEFAULT_MODEL="deepseek-v4-pro", PROVIDER="deepseek",
        API_KEY="settings-key", BASE_URL="https://api.deepseek.com",
    )
    with patch("src.agent_core.model.model_factory.get_settings", return_value=fake_settings):
        model = create_chat_model()

    assert isinstance(model, PatchedChatDeepSeek)
    assert model.api_key.get_secret_value() == "settings-key"
