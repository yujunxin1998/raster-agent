"""中间件流水线共用的运行时上下文类型。

`langchain.agents.middleware.AgentMiddleware` 的各钩子通过 `runtime.context`
（`langgraph.runtime.Runtime[ContextT]`）访问一次请求级别的静态依赖，这与
现有工具（`skill_tool_factory.py`/`memory_tools.py`）built 在旧式
`RunnableConfig.configurable` 之上是两条并行的通道——`create_agent` 在新
框架下仍然会把 `context` 里的字段透传进 `config["configurable"]`，所以
现有工具代码不需要因为中间件骨架的引入而改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRuntimeContext:
    """一次 Agent 运行的静态上下文（对应 DeerFlow 的 thread 级依赖）。

    Attributes:
        conversation_id: 会话 ID，同时是虚拟工作区/沙箱的隔离粒度。
        user_id: 归属用户 ID，可为空（内部调用/测试场景）。
        thinking: 是否处于深度思考模式，供 Guardrail 等策略引用。
        datasource_id: 本次请求绑定的数据源 ID，供
            `DatasourceRoutingMiddleware`/`delegate_to_database_agent` 读取。
        registry_revision: `resolve_tools()`（工具注册中心设计文档 4.3 节）
            解析工具集时依据的 `RegistrySnapshot.revision`，为空代表本次
            运行未经过统一注册中心（如测试场景直接构造 Agent）。
            `ToolAuditMiddleware` 记录审计日志时带上这个字段，用于事后
            复现"这次对话当时模型实际能看到哪些工具"。
        memory_cache: 单次 Agent Run 内的长期记忆上下文缓存（设计文档 §8.2）。
            一次 Run 可能有多次模型调用/工具循环，`MemoryContextBuilder` 用它
            把 Profile/Facts 只查一次；不写入 LangGraph Checkpoint（dataclass
            字段不参与状态持久化），新的用户消息到来时这个 context 实例本身
            就会被重新构造，天然失效，不需要显式清空。
    """

    conversation_id: str
    user_id: str | None = None
    thinking: bool = False
    datasource_id: str | None = None
    registry_revision: int | None = None
    memory_cache: dict = field(default_factory=dict)
