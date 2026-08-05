"""按会话建立/复用隔离工作区的中间件（设计文档 4.2 节 #2、5.2 节）。

对应 DeerFlow 的 `ThreadDataMiddleware`：给每个 thread 建立隔离目录，本工程
里这件事已经由 `ThreadWorkspaceManager.get_or_create()` 实现（见
`src/agent_core/workspace/thread_workspace_manager.py`），本中间件只是把
"在 Agent 运行开始前调用一次"这个动作收口成流水线的一环。

**不把 ThreadWorkspace 实例写入 state**：原因与 `sandbox_middleware.py` 相同
——`ThreadWorkspace` 不是可 msgpack 序列化的纯数据，写入 state 会在
checkpointer 落盘时报错。本中间件的实例本身随每次 `build_lead_agent()`
现造（每请求一份），存成实例私有属性即可安全传递，不需要经过会被持久化的
state。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager


class ThreadDataMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """在 Agent 运行开始前获取（必要时创建）该会话的隔离工作区。"""

    def __init__(self, workspace_manager: ThreadWorkspaceManager) -> None:
        """初始化中间件。

        Args:
            workspace_manager: 会话隔离工作区管理器，通常传入
                `get_thread_workspace_manager()` 返回的全局单例。
        """
        super().__init__()
        self._workspace_manager = workspace_manager
        self._workspace: ThreadWorkspace | None = None

    async def abefore_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """获取本次会话的隔离工作区，存为实例私有属性（不写入 state）。

        Args:
            state: 当前 Agent 状态（本方法不使用）。
            runtime: 运行时上下文，携带 `conversation_id`/`user_id`。

        Returns:
            始终返回 None；缺少 `conversation_id` 时记 warning 并跳过
            （不阻断 Agent 运行）。
        """
        context = runtime.context
        if context is None or not context.conversation_id:
            logger.warning("[ThreadDataMiddleware] 缺少 conversation_id，跳过工作区创建")
            return None

        self._workspace = self._workspace_manager.get_or_create(context.conversation_id, context.user_id)
        return None
