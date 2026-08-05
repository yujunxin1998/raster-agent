"""沙箱获取中间件（设计文档 4.2 节 #5、5.3 节）。

按需获取 Sandbox 实例，供沙箱工具使用。与 `skill_tool_factory.py` 里
"技能工具自行按 conversation_id 获取/释放沙箱"是两条独立路径，互不冲突：
本中间件面向未来"按预先获取好的沙箱执行"的沙箱工具，技能脚本继续走自己已经
验证过的获取方式，不受影响。

**不把 Sandbox 实例写入 state**：早期实现把 `abefore_agent` 获取的 Sandbox
存进 `state["sandbox"]`，结果被 `AsyncPostgresSaver` checkpointer 尝试 msgpack
序列化时直接报错（`Sandbox` 持有子进程/文件句柄等运行时资源，不是可序列化的
纯数据）。本中间件的实例本身就是每次 `build_lead_agent()`（每请求现造一个）
的产物，`abefore_agent`/`aafter_agent` 在同一次请求内顺序执行，把 Sandbox
存成实例私有属性即可安全地在两者之间传递，不需要经过会被持久化的 state。
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime
from loguru import logger

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.sandbox.sandbox import Sandbox
from src.agent_core.sandbox.sandbox_provider import SandboxProvider


class SandboxMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    """在 Agent 运行开始前获取沙箱、运行结束后释放。"""

    def __init__(self, sandbox_provider: SandboxProvider) -> None:
        """初始化中间件。

        Args:
            sandbox_provider: 沙箱提供者，通常传入 `get_sandbox_provider()` 单例。
        """
        super().__init__()
        self._sandbox_provider = sandbox_provider
        self._sandbox: Sandbox | None = None

    async def abefore_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """获取本次会话的沙箱实例，存为实例私有属性（不写入 state）。

        Args:
            state: 当前 Agent 状态（本方法不使用）。
            runtime: 运行时上下文，携带 `conversation_id`/`user_id`。

        Returns:
            始终返回 None；缺少 `conversation_id` 时记 warning 并跳过。
        """
        context = runtime.context
        if context is None or not context.conversation_id:
            logger.warning("[SandboxMiddleware] 缺少 conversation_id，跳过沙箱获取")
            return None

        self._sandbox = await self._sandbox_provider.acquire(context.conversation_id, context.user_id)
        return None

    async def aafter_agent(self, state: Any, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any] | None:
        """释放本次运行获取的沙箱实例。

        Args:
            state: 当前 Agent 状态（本方法不使用）。
            runtime: 运行时上下文（本方法不使用）。

        Returns:
            始终返回 None。
        """
        if self._sandbox is not None:
            await self._sandbox_provider.release(self._sandbox)
            self._sandbox = None
        return None
