"""按会话创建/复用隔离工作区。

对应设计文档中借鉴 DeerFlow `ThreadDataMiddleware` 的部分：把"给每个 thread
建隔离目录"这件事收口到一个类里，未来接入中间件流水线时，中间件只需要调用
`ThreadWorkspaceManager.get_or_create()` 即可，不需要关心目录创建细节。
"""
from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.common.constants import WorkspaceDirectory

_DEFAULT_USER_ID = "default"
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_directory_id(value: str, field_name: str) -> str:
    """校验会参与目录拼接的外部标识，拒绝路径控制字符和超长名称。"""
    if not _SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} 只能包含字母、数字、下划线、连字符，长度为 1~64")
    return value


class ThreadWorkspaceManager:
    """会话隔离工作区的管理器。

    职责单一：给定 (user_id, conversation_id)，返回对应的 ThreadWorkspace，
    并保证其三个子目录已在磁盘上创建好。不负责目录内容的读写，读写操作由
    Sandbox / 具体工具完成（职责分离：本类只管"目录在哪、有没有创建"）。
    """

    def __init__(self, workspace_root: str) -> None:
        """初始化管理器。

        Args:
            workspace_root: 所有会话隔离目录的公共根路径，对应配置项
                `WORKSPACE_ROOT`。

        Raises:
            ValueError: workspace_root 为空字符串。
        """
        if not workspace_root:
            raise ValueError("workspace_root 不能为空")
        self._workspace_root = Path(workspace_root).resolve()

    def get_or_create(self, conversation_id: str, user_id: str | None = None) -> ThreadWorkspace:
        """获取（必要时创建）指定会话的隔离工作区。

        Args:
            conversation_id: 会话 ID，不能为空。
            user_id: 归属用户 ID，为空时回退为 "default"。

        Returns:
            对应会话的 ThreadWorkspace，三个子目录已确保存在。

        Raises:
            ValueError: conversation_id 为空。
        """
        safe_conversation_id = _validate_directory_id(conversation_id, "conversation_id")
        resolved_user_id = _validate_directory_id(user_id or _DEFAULT_USER_ID, "user_id")
        users_root = (self._workspace_root / "users").resolve()
        root = (users_root / resolved_user_id / "threads" / safe_conversation_id).resolve()
        if root == users_root or users_root not in root.parents:
            raise ValueError("会话工作区路径越出 WORKSPACE_ROOT")
        workspace = ThreadWorkspace(conversation_id=conversation_id, user_id=resolved_user_id, root=root)
        workspace.ensure_directories()
        return workspace

    def describe(self, workspace: ThreadWorkspace) -> dict[str, str]:
        """返回三个子目录的绝对路径字符串，便于日志/调试展示。

        Args:
            workspace: 目标工作区。

        Returns:
            以子目录名为 key 的绝对路径字典。
        """
        return {
            directory.value: str(workspace.directory_for(directory))
            for directory in WorkspaceDirectory
        }


_manager: ThreadWorkspaceManager | None = None


def init_thread_workspace_manager(workspace_root: str) -> None:
    """应用启动时调用一次，完成全局单例初始化。

    Args:
        workspace_root: 对应配置项 `WORKSPACE_ROOT`。
    """
    global _manager
    _manager = ThreadWorkspaceManager(workspace_root)
    logger.info(f"[ThreadWorkspaceManager] 初始化完成，root={workspace_root}")


def get_thread_workspace_manager() -> ThreadWorkspaceManager:
    """返回全局唯一的 ThreadWorkspaceManager 实例。

    Raises:
        RuntimeError: init_thread_workspace_manager() 尚未被调用。
    """
    if _manager is None:
        raise RuntimeError("ThreadWorkspaceManager 尚未初始化，请确认应用已完成启动")
    return _manager
