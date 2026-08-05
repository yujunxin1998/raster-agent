"""虚拟工作区模块：按会话隔离的文件系统（对应设计文档 5.2 节）。"""
from src.agent_core.workspace.path_guard import PathGuard
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.agent_core.workspace.thread_workspace_manager import (
    ThreadWorkspaceManager,
    get_thread_workspace_manager,
    init_thread_workspace_manager,
)

__all__ = [
    "PathGuard",
    "ThreadWorkspace",
    "ThreadWorkspaceManager",
    "get_thread_workspace_manager",
    "init_thread_workspace_manager",
]
