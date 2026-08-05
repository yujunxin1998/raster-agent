"""单个会话的隔离工作区（值对象）。

对应设计文档 5.2 节"文件系统（Per-Thread 虚拟工作区）"：每个 conversation_id
拥有 workspace / uploads / outputs 三个子目录，供沙箱工具和技能脚本落地产物。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent_core.workspace.path_guard import PathGuard
from src.common.constants import WorkspaceDirectory


@dataclass(frozen=True)
class ThreadWorkspace:
    """一个会话对应的隔离目录集合。

    Attributes:
        conversation_id: 会话 ID，同时是隔离目录的最内层目录名。
        user_id: 归属用户 ID，用于目录路径的用户级隔离。
        root: 该会话隔离目录的根路径，形如
            ``{WORKSPACE_ROOT}/users/{user_id}/threads/{conversation_id}``。
    """

    conversation_id: str
    user_id: str
    root: Path

    @property
    def workspace_dir(self) -> Path:
        """工作区目录：Agent 读写的中间文件。"""
        return self.root / WorkspaceDirectory.WORKSPACE.value

    @property
    def uploads_dir(self) -> Path:
        """上传目录：用户上传的文件。"""
        return self.root / WorkspaceDirectory.UPLOADS.value

    @property
    def outputs_dir(self) -> Path:
        """产物目录：最终交付给用户的文件（图表、生成的文档等）。"""
        return self.root / WorkspaceDirectory.OUTPUTS.value

    def directory_for(self, directory: WorkspaceDirectory) -> Path:
        """按枚举取对应子目录路径，避免调用方用裸字符串拼路径。

        Args:
            directory: 三个固定子目录之一。

        Returns:
            对应子目录的绝对路径。
        """
        return self.root / directory.value

    def resolve(self, relative_path: str, directory: WorkspaceDirectory = WorkspaceDirectory.WORKSPACE) -> Path:
        """解析虚拟路径（相对某个子目录）为真实绝对路径，并做越界校验。

        Args:
            relative_path: 相对子目录的路径，例如 "reports/summary.md"。
            directory: 目标子目录，默认为 workspace。

        Returns:
            校验通过的绝对路径。

        Raises:
            PathTraversalError: relative_path 越出了该子目录范围。
        """
        guard = PathGuard(root=self.directory_for(directory))
        return guard.resolve(relative_path)

    def ensure_directories(self) -> None:
        """确保三个子目录物理存在（幂等，可重复调用）。"""
        for directory in WorkspaceDirectory:
            self.directory_for(directory).mkdir(parents=True, exist_ok=True)
