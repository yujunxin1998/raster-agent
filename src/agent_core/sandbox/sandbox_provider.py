"""沙箱提供者抽象接口。

`SandboxProvider` 负责 Sandbox 实例的生命周期（获取/释放），与 `Sandbox` 本身
的执行接口分离，是为了让"如何拿到一个可用的沙箱"这件事可以独立于"沙箱怎么
执行命令"演进——本轮只有 LocalSandboxProvider，未来接入容器化时只需新增
DockerSandboxProvider，上层调用方（中间件）代码不用改。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.agent_core.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
    """Sandbox 生命周期管理的抽象接口。"""

    @abstractmethod
    async def acquire(self, conversation_id: str, user_id: str | None = None) -> Sandbox:
        """获取（必要时创建）指定会话的沙箱实例。

        Args:
            conversation_id: 会话 ID，用作沙箱隔离粒度。
            user_id: 归属用户 ID，用于目录路径的用户级隔离。

        Returns:
            可直接使用的 Sandbox 实例。
        """

    @abstractmethod
    async def release(self, sandbox: Sandbox) -> None:
        """释放一个沙箱实例。

        对本地进程模式的实现而言通常是空操作（沙箱状态即宿主机目录，
        无需额外清理）；容器化实现会在这里做容器停止/资源回收。

        Args:
            sandbox: 待释放的 Sandbox 实例。
        """
