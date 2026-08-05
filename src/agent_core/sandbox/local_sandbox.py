"""本地进程沙箱实现。

对应设计文档 5.3 节"一期只做 LocalSandboxProvider"：本质是把原项目
`src/core/skills/content_reader.py::_run_script_blocking` 的子进程执行能力
重新包一层 Sandbox 接口，行为基本不变（子进程 + `asyncio.to_thread` 调度，
POSIX 下保留 `RLIMIT_AS` 内存限制），但补了两件事：

1. 命令执行的 cwd 固定为该会话的 workspace/ 目录，相对路径天然落在隔离目录里。
2. 执行前的环境变量经 `env_policy.build_sandbox_env` 清洗，敏感变量不会
   透传给子进程。

Windows 下没有 `resource` 模块，内存限制这一层防护缺失，与原项目的已知
缺口保持一致（在文档里显式记录，而不是假装解决了）。
"""
from __future__ import annotations

import subprocess
import sys

from loguru import logger

from src.agent_core.sandbox.env_policy import build_sandbox_env
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.common.constants import SandboxCommandStatus, WorkspaceDirectory
from src.common.exceptions import PathTraversalError


def _limit_resources(max_memory_mb: int) -> None:
    """子进程 preexec_fn：仅 POSIX 可用，给子进程设置地址空间上限。

    Args:
        max_memory_mb: 允许的最大地址空间（MB）。
    """
    import resource  # POSIX-only stdlib 模块，仅在 sys.platform != "win32" 时被调用

    max_bytes = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))  # type: ignore[attr-defined]


def _run_command_blocking(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin_data: bytes,
    max_output_bytes: int,
    max_memory_mb: int,
    timeout: float,
) -> CommandResult:
    """同步阻塞版本，跑在线程池里（由 asyncio.to_thread 调度）。

    使用标准库 subprocess.Popen 而非 asyncio 原生子进程支持，原因与原项目
    一致：uvicorn 在 Windows + reload=True 场景下使用的 SelectorEventLoop
    不支持 asyncio 的子进程 API，改用线程池里的同步 subprocess 可以规避
    这个事件循环类型的限制，两边（HTTP 服务 + 沙箱子进程）都能正常工作。

    Args:
        command: 命令及参数列表。
        cwd: 子进程工作目录（该会话的 workspace 绝对路径）。
        env: 已完成清洗的环境变量字典。
        stdin_data: 写入子进程标准输入的字节内容。
        max_output_bytes: stdout 截断上限。
        max_memory_mb: POSIX 下的地址空间上限（Windows 无效）。
        timeout: 超时时间（秒）。

    Returns:
        命令执行结果。
    """
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = lambda: _limit_resources(max_memory_mb)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    try:
        stdout, stderr = process.communicate(input=stdin_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return CommandResult(status=SandboxCommandStatus.TIMEOUT, return_code=None)

    truncated = len(stdout) > max_output_bytes
    if truncated:
        stdout = stdout[:max_output_bytes]

    if truncated:
        status = SandboxCommandStatus.OUTPUT_TRUNCATED
    elif process.returncode == 0:
        status = SandboxCommandStatus.SUCCESS
    else:
        status = SandboxCommandStatus.FAILED

    return CommandResult(
        status=status,
        return_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        truncated=truncated,
    )


class LocalSandbox(Sandbox):
    """基于宿主机本地进程的沙箱实现。

    命令直接在宿主机上以子进程方式执行，隔离性仅依赖"cwd 固定在会话工作区
    + 环境变量清洗 + 资源限制"，不具备容器级别的隔离强度，适用于执行项目
    自身维护的可信脚本（如技能脚本），不适合执行不受信任的任意代码
    （详见设计文档 5.3 节"为什么沙箱一期不上容器隔离"）。
    """

    def __init__(
        self,
        workspace: ThreadWorkspace,
        *,
        default_timeout_seconds: int,
        max_output_bytes: int,
        max_memory_mb: int,
    ) -> None:
        """初始化本地沙箱。

        Args:
            workspace: 该沙箱绑定的会话隔离工作区。
            default_timeout_seconds: 命令未显式指定 timeout 时使用的默认超时秒数。
            max_output_bytes: stdout 截断上限。
            max_memory_mb: POSIX 下子进程的地址空间上限。
        """
        self._workspace = workspace
        self._default_timeout_seconds = default_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_memory_mb = max_memory_mb

    @property
    def workspace(self) -> ThreadWorkspace:
        """返回该沙箱绑定的会话隔离工作区，供上层查询虚拟路径映射关系。"""
        return self._workspace

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

        import asyncio

        resolved_timeout = timeout or self._default_timeout_seconds
        resolved_env = build_sandbox_env(env)

        result = await asyncio.to_thread(
            _run_command_blocking,
            command,
            cwd=str(self._workspace.workspace_dir),
            env=resolved_env,
            stdin_data=stdin or b"",
            max_output_bytes=self._max_output_bytes,
            max_memory_mb=self._max_memory_mb,
            timeout=resolved_timeout,
        )

        if result.status == SandboxCommandStatus.TIMEOUT:
            logger.warning(f"[LocalSandbox] 命令执行超时 command={command} timeout={resolved_timeout}s")
        elif result.status == SandboxCommandStatus.OUTPUT_TRUNCATED:
            logger.warning(f"[LocalSandbox] 命令输出超过上限 command={command} limit={self._max_output_bytes}")
        elif result.status == SandboxCommandStatus.FAILED:
            logger.warning(f"[LocalSandbox] 命令执行失败 command={command} return_code={result.return_code}")
        return result

    async def read_file(self, virtual_path: str) -> str:
        real_path = self._workspace.resolve(virtual_path, WorkspaceDirectory.WORKSPACE)
        if not real_path.is_file():
            raise FileNotFoundError(f"文件不存在: {virtual_path}")
        return real_path.read_text(encoding="utf-8")

    async def write_file(self, virtual_path: str, content: str, *, overwrite: bool = True) -> None:
        real_path = self._workspace.resolve(virtual_path, WorkspaceDirectory.WORKSPACE)
        if real_path.exists() and not overwrite:
            raise FileExistsError(f"文件已存在且 overwrite=False: {virtual_path}")
        real_path.parent.mkdir(parents=True, exist_ok=True)
        real_path.write_text(content, encoding="utf-8")

    async def list_dir(self, virtual_path: str = "") -> list[dict]:
        try:
            real_dir = self._workspace.resolve(virtual_path, WorkspaceDirectory.WORKSPACE)
        except PathTraversalError:
            raise
        if not real_dir.is_dir():
            raise ValueError(f"'{virtual_path}' 不是一个目录")

        entries = sorted(real_dir.iterdir(), key=lambda entry: (entry.is_file(), entry.name.lower()))
        base = self._workspace.workspace_dir
        return [
            {
                "name": entry.name,
                "path": str(entry.relative_to(base)).replace("\\", "/"),
                "type": "dir" if entry.is_dir() else "file",
            }
            for entry in entries
        ]
