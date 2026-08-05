"""Chat Model 工厂：统一的模型构建入口。

对应设计文档"参考 DeerFlow 封装 LLM 工厂"的决策：一个无状态工厂函数
`create_chat_model()`，DeepSeek 走 `PatchedChatDeepSeek`（reasoning_content
多轮续接修复），其余供应商走 LangChain 通用的 `init_chat_model`。取代原项目
`LLMFactory` 按 `(purpose, kwargs)` 缓存单例的写法——模型实例构造成本不高，
现造一个的开销可以忽略，缓存带来的"同一个实例被多处意外共享 callbacks/状态"
的心智负担反而更值得避免。
"""
from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.agent_core.model.patched_deepseek import PatchedChatDeepSeek
from src.config.settings import get_settings

_DEEPSEEK_PROVIDER = "deepseek"


def create_chat_model(
    *,
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    thinking_enabled: bool = False,
    **overrides: Any,
) -> BaseChatModel:
    """构建一个 Chat Model 实例。

    Args:
        model: 模型名称，为空时回落到 `settings.DEFAULT_MODEL`。
        provider: 模型供应商标识，为空时回落到 `settings.PROVIDER`。
        api_key: 供应商 API Key，为空时回落到 `settings.API_KEY`。
        base_url: 供应商 Base URL，为空时回落到 `settings.BASE_URL`。
        thinking_enabled: 是否开启扩展推理模式。DeepSeek 下会显式写入
            `extra_body.thinking.type`（不传等于什么都没做，模型可能沿用
            供应商默认值，导致关闭 thinking 时仍然产生推理 token）。
        **overrides: 透传给模型构造函数的其余参数（如 `temperature`）。

    Returns:
        对应供应商的 Chat Model 实例。
    """
    settings = get_settings()
    resolved_model = model or settings.DEFAULT_MODEL
    resolved_provider = provider or settings.PROVIDER
    resolved_api_key = api_key or settings.API_KEY
    resolved_base_url = base_url or settings.BASE_URL

    if resolved_provider == _DEEPSEEK_PROVIDER:
        return PatchedChatDeepSeek(
            model=resolved_model,
            api_key=resolved_api_key,
            api_base=resolved_base_url,
            extra_body={"thinking": {"type": "enabled" if thinking_enabled else "disabled"}},
            **overrides,
        )

    return init_chat_model(
        model=resolved_model,
        model_provider=resolved_provider,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        **overrides,
    )
