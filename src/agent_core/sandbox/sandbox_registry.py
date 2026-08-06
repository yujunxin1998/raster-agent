"""沙箱提供者的全局单例登记表。

与具体 Provider 实现（Local/Docker）解耦：原来 `_provider`/`init_*`/
`get_sandbox_provider()` 这一套全局单例逻辑直接写在 `local_sandbox_provider.py`
里，只要接入第二种实现（`DockerSandboxProvider`）就必须挪出来，否则两个模块
会各自维护一份不一致的全局状态。调用方（`sandbox_tool.py`/`sandbox_middleware.py`/
`skill_tool_factory.py` 等）统一从 `src.agent_core.sandbox` 包导入
`get_sandbox_provider`，不关心背后登记的是哪种实现。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.sandbox.sandbox_provider import SandboxProvider

_provider: SandboxProvider | None = None


def init_sandbox_provider(provider: SandboxProvider) -> None:
    """应用启动时调用一次，登记全局唯一的沙箱提供者实例。

    Args:
        provider: 已完成构造的 SandboxProvider 实现（Local 或 Docker）。
    """
    global _provider
    _provider = provider
    logger.info(f"[SandboxRegistry] 已登记 {type(provider).__name__}")


def get_sandbox_provider() -> SandboxProvider:
    """返回全局唯一的沙箱提供者实例。

    Raises:
        RuntimeError: 尚未调用过 init_sandbox_provider()。
    """
    if _provider is None:
        raise RuntimeError("SandboxProvider 尚未初始化，请确认应用已完成启动")
    return _provider
