"""沙箱抽象接口。

对应设计文档 5.3 节：技能脚本、未来的代码执行类工具统一经过这一层执行，
接口形状参考 DeerFlow 的 `Sandbox` 最小接口集（execute_command / read_file /
write_file / list_dir），本轮只提供 LocalSandbox 一种实现，接口先按可平滑
扩展到容器化实现的形状设计。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.common.constants import SandboxCommandStatus, WorkspaceDirectory


@dataclass(frozen=True)
class CommandResult:
    """一次沙箱命令执行的结果。

    Attributes:
        status: 执行结果状态（成功/失败/超时/输出截断）。
        return_code: 进程退出码，超时场景下为 None。
        stdout: 标准输出（可能已被截断，见 truncated 字段）。
        stderr: 标准错误输出。
        truncated: stdout 是否因超过 `SANDBOX_MAX_OUTPUT_BYTES` 被截断。
    """

    status: SandboxCommandStatus
    return_code: int | None
    stdout: bytes = field(default=b"")
    stderr: bytes = field(default=b"")
    truncated: bool = False

    @property
    def succeeded(self) -> bool:
        """命令是否正常执行完成（退出码为 0 且未超时未截断）。"""
        return self.status == SandboxCommandStatus.SUCCESS

    def stdout_text(self, encoding: str = "utf-8") -> str:
        """将 stdout 解码为文本，遇到非法字节时替换而不是抛异常。"""
        return self.stdout.decode(encoding, errors="replace")

    def stderr_text(self, encoding: str = "utf-8") -> str:
        """将 stderr 解码为文本，遇到非法字节时替换而不是抛异常。"""
        return self.stderr.decode(encoding, errors="replace")


class Sandbox(ABC):
    """沙箱执行环境的抽象接口。

    所有方法均以"虚拟路径"（相对某会话工作区的相对路径）寻址，不接受
    宿主机绝对路径，具体的路径映射由实现类负责（本轮的 LocalSandbox
    直接映射到宿主机目录，未来的容器化实现可以映射到容器内路径）。
    """

    @abstractmethod
    async def execute_command(
        self,
        command: list[str],
        *,
        stdin: bytes | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """在沙箱内执行一条命令。

        Args:
            command: 命令及其参数列表，例如 `["python3", "main.py"]`。
            stdin: 写入子进程标准输入的字节内容，可以为空。
            env: 额外注入的环境变量（会与沙箱默认的清洗后环境变量合并）。
            timeout: 超时时间（秒），为空时使用沙箱实现的默认值。

        Returns:
            命令执行结果。
        """

    @abstractmethod
    async def read_file(
        self, virtual_path: str, directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE
    ) -> str:
        """读取沙箱内某个虚拟路径对应文件的文本内容。

        Args:
            virtual_path: 相对 `directory` 子目录的虚拟路径。
            directory: 目标子目录，默认为 workspace（Agent 读写的中间文件）；
                读取 `outputs/` 下已落地的产物时传 `WorkspaceDirectory.OUTPUTS`。

        Returns:
            文件文本内容。

        Raises:
            PathTraversalError: virtual_path 越出了该子目录范围。
            FileNotFoundError: 文件不存在。
        """

    @abstractmethod
    async def write_file(
        self,
        virtual_path: str,
        content: str,
        *,
        overwrite: bool = True,
        directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE,
    ) -> None:
        """向沙箱内某个虚拟路径写入文本内容。

        Args:
            virtual_path: 相对 `directory` 子目录的虚拟路径。
            content: 要写入的文本内容。
            overwrite: 目标文件已存在时是否允许覆盖，默认允许。
            directory: 目标子目录，默认为 workspace；把最终产物落地到
                `outputs/` 供下载时传 `WorkspaceDirectory.OUTPUTS`。

        Raises:
            PathTraversalError: virtual_path 越出了该子目录范围。
            FileExistsError: overwrite=False 且目标文件已存在。
        """

    @abstractmethod
    async def list_dir(
        self, virtual_path: str = "", directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE
    ) -> list[dict]:
        """列出沙箱内某个虚拟目录下的直接子节点。

        Args:
            virtual_path: 相对 `directory` 子目录的虚拟路径，空字符串表示该
                子目录根目录。
            directory: 目标子目录，默认为 workspace。

        Returns:
            形如 `[{"name": ..., "path": ..., "type": "dir" | "file"}, ...]` 的列表。
        """
