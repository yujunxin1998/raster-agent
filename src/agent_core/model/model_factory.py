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

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.agent_core.model.patched_deepseek import PatchedChatDeepSeek
from src.agent_core.model.patched_vllm import PatchedChatOpenAI
from src.config.settings import get_settings

_DEEPSEEK_PROVIDER = "deepseek"
_VLLM_PROVIDERS = frozenset({"vllm", "qwen"})


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
    resolved_api_key = api_key or settings.OPENAI_API_KEY or settings.API_KEY
    resolved_base_url = base_url or settings.OPENAI_BASE_URL or settings.BASE_URL

    # DeerFlow-style model selection: the thinking switch can select a
    # separately advertised model (for example qwen3-thinking). If no separate
    # model is configured, the provider-specific request body below toggles the
    # same model's thinking mode.
    if model is None and thinking_enabled and settings.OPENAI_MODEL_THINKING:
        resolved_model = settings.OPENAI_MODEL_THINKING

    if resolved_provider == _DEEPSEEK_PROVIDER:
        # DeepSeek is reachable directly in the deployment environment.  Pass
        # clients with trust_env=False so httpx does not silently pick up
        # ALL_PROXY/HTTP(S)_PROXY from the shell and route model requests
        # through an unavailable local proxy.
        direct_http_client = httpx.Client(trust_env=False)
        direct_async_http_client = httpx.AsyncClient(trust_env=False)
        return PatchedChatDeepSeek(
            model=resolved_model,
            api_key=resolved_api_key,
            api_base=resolved_base_url,
            extra_body={"thinking": {"type": "enabled" if thinking_enabled else "disabled"}},
            http_client=direct_http_client,
            http_async_client=direct_async_http_client,
            **overrides,
        )

    is_qwen_compatible_openai = (
        resolved_provider == "openai"
        and "qwen" in resolved_model.lower()
    )
    if resolved_provider in _VLLM_PROVIDERS or is_qwen_compatible_openai:
        extra_body = dict(overrides.pop("extra_body", {}) or {})
        model_name = resolved_model.lower()
        is_qwen = resolved_provider == "qwen" or "qwen" in model_name
        has_separate_thinking_model = bool(
            settings.OPENAI_MODEL_THINKING
            and settings.OPENAI_MODEL_THINKING != settings.DEFAULT_MODEL
        )

        # Qwen served by vLLM uses the chat template switch. When a distinct
        # thinking model is configured, keep the model selection as the source
        # of truth but still send the explicit flag for compatible servers.
        if is_qwen and (not has_separate_thinking_model or thinking_enabled):
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
            chat_template_kwargs["enable_thinking"] = thinking_enabled
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        return PatchedChatOpenAI(
            model=resolved_model,
            api_key=resolved_api_key or "EMPTY",
            base_url=resolved_base_url,
            extra_body=extra_body or None,
            **overrides,
        )

    return init_chat_model(
        model=resolved_model,
        model_provider=resolved_provider,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        **overrides,
    )
