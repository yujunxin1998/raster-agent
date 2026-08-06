"""Docker 沙箱的提供者实现（设计文档 5.3 节"二期预留 DockerSandboxProvider"）。"""
from __future__ import annotations

import asyncio

from loguru import logger

from src.agent_core.sandbox.docker_sandbox import DockerSandbox
from src.agent_core.sandbox.sandbox import Sandbox
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.sandbox.sandbox_registry import init_sandbox_provider
from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager


class DockerSandboxProvider(SandboxProvider):
    """基于 Docker 容器执行命令的沙箱提供者。

    `acquire()` 依然是无状态的（只是构造一个绑定到同一会话工作区目录的新
    `DockerSandbox` 对象），真正的容器创建/删除发生在每次
    `DockerSandbox.execute_command()` 调用内部（一次性容器，用完即删），
    因此 `release()` 同样保持无操作，不需要维护"容器和 conversation_id 的
    映射表"这类额外状态。
    """

    def __init__(
        self,
        workspace_manager: ThreadWorkspaceManager,
        docker_client,
        *,
        image: str,
        project_root: str,
        default_timeout_seconds: int,
        max_output_bytes: int,
        max_memory_mb: int,
    ) -> None:
        """初始化 Docker 沙箱提供者。

        Args:
            workspace_manager: 会话隔离工作区管理器，用于按 conversation_id 定位目录。
            docker_client: 已完成握手的 `docker.DockerClient` 实例。
            image: 执行命令使用的默认镜像名。
            project_root: 项目根目录的宿主机绝对路径。
            default_timeout_seconds: 沙箱内命令的默认超时秒数。
            max_output_bytes: 沙箱内命令 stdout 的截断上限。
            max_memory_mb: 容器内存上限（cgroup 强制）。
        """
        self._workspace_manager = workspace_manager
        self._docker_client = docker_client
        self._image = image
        self._project_root = project_root
        self._default_timeout_seconds = default_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_memory_mb = max_memory_mb

    async def acquire(self, conversation_id: str, user_id: str | None = None) -> Sandbox:
        workspace = self._workspace_manager.get_or_create(conversation_id, user_id)
        return DockerSandbox(
            workspace,
            self._docker_client,
            image=self._image,
            project_root=self._project_root,
            default_timeout_seconds=self._default_timeout_seconds,
            max_output_bytes=self._max_output_bytes,
            max_memory_mb=self._max_memory_mb,
        )

    async def release(self, sandbox: Sandbox) -> None:
        # 容器已经在 DockerSandbox.execute_command() 内部用完即删，这里无需处理。
        pass


async def init_docker_sandbox_provider(
    workspace_manager: ThreadWorkspaceManager,
    *,
    image: str,
    project_root: str,
    default_timeout_seconds: int,
    max_output_bytes: int,
    max_memory_mb: int,
) -> None:
    """应用启动时调用一次，构造 DockerSandboxProvider 并登记为全局单例。

    `docker.from_env()` 会同步做一次 daemon 握手，放进线程池执行避免阻塞
    事件循环（和 `LocalSandbox` 里子进程执行放线程池是同一个考虑）。

    Raises:
        RuntimeError: 无法连接到 Docker daemon（比如本机没有启动 Docker Desktop）。
    """
    import docker

    try:
        client = await asyncio.to_thread(docker.from_env)
    except Exception as exc:
        raise RuntimeError(
            f"无法连接到 Docker daemon，请确认已启动 Docker（或改用 SANDBOX_PROVIDER=local）: {exc}"
        ) from exc

    init_sandbox_provider(
        DockerSandboxProvider(
            workspace_manager,
            client,
            image=image,
            project_root=project_root,
            default_timeout_seconds=default_timeout_seconds,
            max_output_bytes=max_output_bytes,
            max_memory_mb=max_memory_mb,
        )
    )
    logger.info(f"[DockerSandboxProvider] 初始化完成 image={image}")
