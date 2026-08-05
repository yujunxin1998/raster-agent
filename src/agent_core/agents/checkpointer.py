"""Lead Agent 会话持久化 checkpointer 的全局单例。

`AsyncPostgresSaver` 的连接生命周期绑定在 `main.py` lifespan 的 `async with` 块内
（对应原项目同款的连接管理方式），本模块只负责把已经 `setup()` 好的实例登记成
全局单例，供各 `build_lead_agent()` 调用点取用。
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver

_checkpointer: BaseCheckpointSaver | None = None


def init_checkpointer(checkpointer: BaseCheckpointSaver) -> None:
    """应用启动时调用一次，注册全局单例。

    Args:
        checkpointer: 已完成 `setup()` 的 checkpointer 实例。
    """
    global _checkpointer
    _checkpointer = checkpointer


def get_checkpointer() -> BaseCheckpointSaver:
    """返回全局唯一的 checkpointer 实例。

    Raises:
        RuntimeError: init_checkpointer() 尚未被调用。
    """
    if _checkpointer is None:
        raise RuntimeError("Checkpointer 尚未初始化，请确认应用已完成启动")
    return _checkpointer
