"""沙箱模块：技能脚本与未来的代码执行类工具统一经此执行（对应设计文档 5.3 节）。"""
from src.agent_core.sandbox.local_sandbox import LocalSandbox
from src.agent_core.sandbox.local_sandbox_provider import (
    LocalSandboxProvider,
    get_sandbox_provider,
    init_local_sandbox_provider,
)
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox
from src.agent_core.sandbox.sandbox_provider import SandboxProvider

__all__ = [
    "CommandResult",
    "Sandbox",
    "SandboxProvider",
    "LocalSandbox",
    "LocalSandboxProvider",
    "init_local_sandbox_provider",
    "get_sandbox_provider",
]
