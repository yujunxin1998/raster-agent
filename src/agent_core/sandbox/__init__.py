"""沙箱模块：技能脚本与未来的代码执行类工具统一经此执行（对应设计文档 5.3 节）。

`get_sandbox_provider()`/`init_sandbox_provider()` 是与具体实现无关的全局单例
登记表（见 `sandbox_registry.py`），应用启动时按 `settings.SANDBOX_PROVIDER`
调用 `init_local_sandbox_provider()` 或 `init_docker_sandbox_provider()` 其中
之一即可，调用方统一用 `get_sandbox_provider()` 取用，不关心背后是哪种实现。
"""
from src.agent_core.sandbox.docker_sandbox import DockerSandbox
from src.agent_core.sandbox.docker_sandbox_provider import (
    DockerSandboxProvider,
    init_docker_sandbox_provider,
)
from src.agent_core.sandbox.local_sandbox import LocalSandbox
from src.agent_core.sandbox.local_sandbox_provider import LocalSandboxProvider, init_local_sandbox_provider
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.sandbox.sandbox_registry import get_sandbox_provider, init_sandbox_provider

__all__ = [
    "CommandResult",
    "Sandbox",
    "SandboxProvider",
    "LocalSandbox",
    "LocalSandboxProvider",
    "init_local_sandbox_provider",
    "DockerSandbox",
    "DockerSandboxProvider",
    "init_docker_sandbox_provider",
    "init_sandbox_provider",
    "get_sandbox_provider",
]
