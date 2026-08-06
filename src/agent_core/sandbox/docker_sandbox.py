"""基于 Docker 容器的沙箱实现（设计文档 5.3 节"二期预留 DockerSandboxProvider"）。

只有 `execute_command` 真正需要容器级隔离：`read_file`/`write_file`/`list_dir`
本质是宿主机文件系统操作（沙箱和宿主机共享同一个按 `conversation_id` 隔离的
workspace 目录），所以这三个方法直接组合一个 `LocalSandbox` 实例复用其实现，
不重复写一遍路径解析逻辑。`execute_command` 每次调用起一个 `--rm` 语义的一次性
容器（用完即删），不维护长期存活的容器生命周期——这样 `release()` 可以继续
保持无操作，不需要引入容器 reaping/孤儿容器清理这类额外复杂度，和 `LocalSandbox`
"无状态 acquire/release" 的语义完全一致。

两个需要"翻译"的地方（容器内文件系统布局和宿主机不同）：

1. **解释器路径**：调用方（`sandbox_tool.py`/`skill_content_reader.py`）用
   `sys.executable` 拼 Python 命令——这是宿主机解释器的绝对路径，在容器里没有
   意义，`_translate_command()` 检测到时替换成容器镜像自带的 `python3`。
2. **技能脚本的绝对路径**：`SkillContentReader.run_script()` 传入的是脚本在
   宿主机上的绝对路径（如 `{PROJECT_ROOT}/skills/core/xxx/scripts/main.py`），
   不在按会话隔离的 workspace 目录下，因此额外把整个项目根目录只读挂载到容器
   内固定路径 `/app`，`_translate_command()` 把落在 `PROJECT_ROOT` 前缀下的
   绝对路径参数等价替换成 `/app` 前缀——这样技能脚本不需要感知自己是被容器
   还是宿主机进程执行。

已知限制（显式记录而非假装解决）：
- 默认镜像 `python:3.11-slim` 不含 Node.js，涉及 `.js` 脚本的技能在 Docker
  模式下会失败，需要通过 `DOCKER_SANDBOX_IMAGE` 换成自带 node 的镜像。
- 暂不支持 `execute_command(stdin=...)` 管道输入（`SkillContentReader` 用它
  传 JSON 参数给脚本）——docker-py 对已启动容器的 stdin attach 需要走底层
  socket API，稳定性没有子进程 `communicate()` 有把握，本轮先返回明确的失败
  说明而不是静默挂起，需要用到 stdin 的场景请使用 `SANDBOX_PROVIDER=local`。
"""
from __future__ import annotations

import asyncio
import sys

from loguru import logger

from src.agent_core.sandbox.env_policy import build_sandbox_env
from src.agent_core.sandbox.local_sandbox import LocalSandbox
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.common.constants import SandboxCommandStatus, WorkspaceDirectory

_CONTAINER_WORKSPACE_PATH = "/workspace"
_CONTAINER_PROJECT_ROOT_PATH = "/app"
_STDIN_UNSUPPORTED_MESSAGE = "DockerSandbox 暂不支持 stdin 管道输入，请改用 SANDBOX_PROVIDER=local"

# 容器有自己的文件系统布局，宿主机的这几个变量传进去反而会指错路径（例如
# Windows 风格的 PATH/HOME 会覆盖镜像自带的 Linux 默认值），需要在清洗后的
# 环境变量里再去掉一层。
_HOST_ONLY_ENV_NAMES = frozenset({"PATH", "HOME", "TMPDIR", "TMP", "TEMP", "VIRTUAL_ENV"})


class DockerSandbox(Sandbox):
    """基于 Docker 容器的沙箱实现，命令在一次性容器内执行，文件操作委托给 LocalSandbox。"""

    def __init__(
        self,
        workspace: ThreadWorkspace,
        docker_client,
        *,
        image: str,
        project_root: str,
        default_timeout_seconds: int,
        max_output_bytes: int,
        max_memory_mb: int,
    ) -> None:
        """初始化 Docker 沙箱。

        Args:
            workspace: 该沙箱绑定的会话隔离工作区。
            docker_client: 已完成握手的 `docker.DockerClient` 实例。
            image: 执行命令使用的镜像名。
            project_root: 项目根目录的宿主机绝对路径，用于技能脚本绝对路径翻译。
            default_timeout_seconds: 命令未显式指定 timeout 时使用的默认超时秒数。
            max_output_bytes: stdout 截断上限。
            max_memory_mb: 容器内存上限（cgroup 真正强制，比 Local 更强）。
        """
        self._workspace = workspace
        self._docker_client = docker_client
        self._image = image
        self._project_root = project_root.rstrip("/\\")
        self._default_timeout_seconds = default_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_memory_mb = max_memory_mb
        self._local = LocalSandbox(
            workspace,
            default_timeout_seconds=default_timeout_seconds,
            max_output_bytes=max_output_bytes,
            max_memory_mb=max_memory_mb,
        )

    @property
    def workspace(self) -> ThreadWorkspace:
        """返回该沙箱绑定的会话隔离工作区，供上层查询虚拟路径映射关系。"""
        return self._workspace

    async def read_file(
        self, virtual_path: str, directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE
    ) -> str:
        return await self._local.read_file(virtual_path, directory)

    async def write_file(
        self,
        virtual_path: str,
        content: str,
        *,
        overwrite: bool = True,
        directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE,
    ) -> None:
        await self._local.write_file(virtual_path, content, overwrite=overwrite, directory=directory)

    async def list_dir(
        self, virtual_path: str = "", directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE
    ) -> list[dict]:
        return await self._local.list_dir(virtual_path, directory)

    async def execute_command(
        self,
        command: list[str],
        *,
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not command:
            raise ValueError("command 不能为空")
        if stdin:
            logger.warning(f"[DockerSandbox] 暂不支持 stdin 管道输入，已拒绝 command={command}")
            return CommandResult(
                status=SandboxCommandStatus.FAILED,
                return_code=None,
                stderr=_STDIN_UNSUPPORTED_MESSAGE.encode("utf-8"),
            )

        resolved_timeout = timeout or self._default_timeout_seconds
        resolved_env = self._build_container_env(env)
        translated_command = self._translate_command(command)

        result = await asyncio.to_thread(
            self._run_container_blocking,
            translated_command,
            env=resolved_env,
            timeout=resolved_timeout,
        )

        if result.status == SandboxCommandStatus.TIMEOUT:
            logger.warning(f"[DockerSandbox] 命令执行超时 command={command} timeout={resolved_timeout}s")
        elif result.status == SandboxCommandStatus.OUTPUT_TRUNCATED:
            logger.warning(f"[DockerSandbox] 命令输出超过上限 command={command} limit={self._max_output_bytes}")
        elif result.status == SandboxCommandStatus.FAILED:
            logger.warning(f"[DockerSandbox] 命令执行失败 command={command} return_code={result.return_code}")
        return result

    def _translate_command(self, command: list[str]) -> list[str]:
        """把宿主机解释器路径/项目内绝对路径翻译成容器内的等价路径。"""
        translated: list[str] = []
        for arg in command:
            if arg == sys.executable:
                translated.append("python3")
                continue
            normalized = arg.replace("\\", "/")
            project_root_normalized = self._project_root.replace("\\", "/")
            if normalized.startswith(project_root_normalized):
                translated.append(_CONTAINER_PROJECT_ROOT_PATH + normalized[len(project_root_normalized):])
                continue
            translated.append(arg)
        return translated

    def _build_container_env(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        cleaned = build_sandbox_env(extra_env)
        return {name: value for name, value in cleaned.items() if name not in _HOST_ONLY_ENV_NAMES}

    def _run_container_blocking(
        self, command: list[str], *, env: dict[str, str], timeout: float
    ) -> CommandResult:
        """同步阻塞版本，跑在线程池里（docker-py 是同步 SDK）。"""
        container = self._docker_client.containers.run(
            self._image,
            command,
            working_dir=_CONTAINER_WORKSPACE_PATH,
            volumes={
                str(self._workspace.workspace_dir): {"bind": _CONTAINER_WORKSPACE_PATH, "mode": "rw"},
                self._project_root: {"bind": _CONTAINER_PROJECT_ROOT_PATH, "mode": "ro"},
            },
            environment=env,
            mem_limit=f"{self._max_memory_mb}m",
            detach=True,
        )
        try:
            try:
                exit_info = container.wait(timeout=timeout)
                return_code = exit_info.get("StatusCode")
            except Exception:
                container.kill()
                return CommandResult(status=SandboxCommandStatus.TIMEOUT, return_code=None)

            stdout = container.logs(stdout=True, stderr=False)
            stderr = container.logs(stdout=False, stderr=True)
        finally:
            try:
                container.remove(force=True)
            except Exception as exc:
                logger.warning(f"[DockerSandbox] 清理容器失败 container_id={container.id}: {exc}")

        truncated = len(stdout) > self._max_output_bytes
        if truncated:
            stdout = stdout[: self._max_output_bytes]

        if truncated:
            status = SandboxCommandStatus.OUTPUT_TRUNCATED
        elif return_code == 0:
            status = SandboxCommandStatus.SUCCESS
        else:
            status = SandboxCommandStatus.FAILED

        return CommandResult(
            status=status, return_code=return_code, stdout=stdout, stderr=stderr, truncated=truncated
        )
