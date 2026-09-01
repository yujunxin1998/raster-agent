"""中间件流水线共用的运行时上下文类型。

`langchain.agents.middleware.AgentMiddleware` 的各钩子、以及标注了
`runtime: ToolRuntime[AgentRuntimeContext]` 参数的工具函数，都通过
`runtime.context`（`langgraph.runtime.Runtime[ContextT]`）访问一次请求级别
的静态依赖。这是唯一的业务上下文通道——`create_agent`/`StateGraph` 并不会把
`context` dataclass 的字段展开透传进 `config["configurable"]`（查过
`langgraph.pregel` 源码：`context` 整体作为一个不透明对象存进
`config["configurable"]["__pregel_runtime"]`，不产生 `user_id` 这样的顶层
key）。`config["configurable"]` 只保留 LangGraph 框架本身需要的键，目前
唯一的例外是 `thread_id`——Checkpointer 硬性要求从这里读取，不能迁移进
`context_schema`。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentRuntimeContext:
    """一次 Agent 运行的静态上下文（对应 DeerFlow 的 thread 级依赖）。

    Attributes:
        conversation_id: 会话 ID，同时是虚拟工作区/沙箱的隔离粒度，也是
            Checkpointer 的 `thread_id`（两者取值相同，只是后者因为框架
            限制必须另外放进 `config["configurable"]["thread_id"]`）。
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
        explicit_skill_names: 用户/API 本次请求显式指定必须激活的技能名
            （`SkillRouter.route()` 的 `forced` 判定信号之一，见
            `docs/Skill与Tool完全解耦重构设计.md` 7.2 节）。目前调用方
            （`chat_pipeline.py`）尚未提供真实来源（前端还没有"强制指定
            技能"这个入口），恒为空元组，预留字段。
    """

    conversation_id: str
    user_id: str | None = None
    thinking: bool = False
    datasource_id: str | None = None
    registry_revision: int | None = None
    memory_cache: dict = field(default_factory=dict)
    explicit_skill_names: tuple[str, ...] = ()
    trace_id: str | None = None
    task_id: str | None = None
    parent_task_id: str | None = None
    root_task_id: str | None = None
    agent_name: str = "lead-agent"
    delegation_depth: int = 0
