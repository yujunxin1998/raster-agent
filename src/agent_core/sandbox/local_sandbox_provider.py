"""本地沙箱的提供者实现。"""
from __future__ import annotations

from src.agent_core.sandbox.local_sandbox import LocalSandbox
from src.agent_core.sandbox.sandbox import Sandbox
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.sandbox.sandbox_registry import init_sandbox_provider
from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager


class LocalSandboxProvider(SandboxProvider):
    """基于 ThreadWorkspaceManager 的本地沙箱提供者。

    每次 acquire() 都会返回一个绑定到对应会话工作区的新 LocalSandbox 实例，
    实例本身无状态（不持有子进程句柄等需要清理的资源），因此 release() 是
    空操作——这一点会在未来的 DockerSandboxProvider 中不同（需要真正停止
    容器/归还连接池）。
    """

    def __init__(
        self,
        workspace_manager: ThreadWorkspaceManager,
        *,
        default_timeout_seconds: int,
        max_output_bytes: int,
        max_memory_mb: int,
    ) -> None:
        """初始化本地沙箱提供者。

        Args:
            workspace_manager: 会话隔离工作区管理器，用于按 conversation_id 定位目录。
            default_timeout_seconds: 沙箱内命令的默认超时秒数。
            max_output_bytes: 沙箱内命令 stdout 的截断上限。
            max_memory_mb: POSIX 下子进程的地址空间上限。
        """
        self._workspace_manager = workspace_manager
        self._default_timeout_seconds = default_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_memory_mb = max_memory_mb

    async def acquire(self, conversation_id: str, user_id: str | None = None) -> Sandbox:
        workspace = self._workspace_manager.get_or_create(conversation_id, user_id)
        return LocalSandbox(
            workspace,
            default_timeout_seconds=self._default_timeout_seconds,
            max_output_bytes=self._max_output_bytes,
            max_memory_mb=self._max_memory_mb,
        )

    async def release(self, sandbox: Sandbox) -> None:
        # 本地进程沙箱无需释放任何资源，保留方法体以满足接口契约并便于未来扩展审计埋点。
        pass


def init_local_sandbox_provider(
    workspace_manager: ThreadWorkspaceManager,
    *,
    default_timeout_seconds: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> None:
    """应用启动时调用一次，构造 LocalSandboxProvider 并登记为全局单例。

    全局单例登记表由 `sandbox_registry` 统一持有（不再由本模块自己持有），
    这样 `get_sandbox_provider()` 无论应用启动时选择了 Local 还是 Docker
    提供者都能拿到正确的实例。
    """
    init_sandbox_provider(
        LocalSandboxProvider(
            workspace_manager,
            default_timeout_seconds=default_timeout_seconds,
            max_output_bytes=max_output_bytes,
            max_memory_mb=max_memory_mb,
        )
    )


__all__ = ["LocalSandboxProvider", "init_local_sandbox_provider"]
