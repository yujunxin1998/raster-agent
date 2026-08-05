"""中间件流水线共用的运行时上下文类型。

`langchain.agents.middleware.AgentMiddleware` 的各钩子通过 `runtime.context`
（`langgraph.runtime.Runtime[ContextT]`）访问一次请求级别的静态依赖，这与
现有工具（`skill_tool_factory.py`/`memory_tools.py`）built 在旧式
`RunnableConfig.configurable` 之上是两条并行的通道——`create_agent` 在新
框架下仍然会把 `context` 里的字段透传进 `config["configurable"]`，所以
现有工具代码不需要因为中间件骨架的引入而改动。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentRuntimeContext:
    """一次 Agent 运行的静态上下文（对应 DeerFlow 的 thread 级依赖）。

    Attributes:
        conversation_id: 会话 ID，同时是虚拟工作区/沙箱的隔离粒度。
        user_id: 归属用户 ID，可为空（内部调用/测试场景）。
        thinking: 是否处于深度思考模式，供 Guardrail 等策略引用。
        datasource_id: 本次请求绑定的数据源 ID，供
            `DatasourceRoutingMiddleware`/`delegate_to_database_agent` 读取。
    """

    conversation_id: str
    user_id: str | None = None
    thinking: bool = False
    datasource_id: str | None = None
