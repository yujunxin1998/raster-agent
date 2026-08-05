"""中间件流水线共用的扩展 AgentState。

`AgentMiddleware.state_schema` 允许每个中间件声明自己需要的额外 state 字段，
`create_agent` 组装时会把所有中间件的 schema 合并。

**重要约束**：这里声明的字段最终都会被 checkpointer（`AsyncPostgresSaver`）
按 msgpack 序列化、持久化——只能放"纯数据"（字符串/数字/列表/字典这类），
不能放 `Sandbox`/`ThreadWorkspace` 这类持有资源引用的运行时对象，否则每一步
都会在写 checkpoint 时抛 `TypeError: Type is not msgpack serializable`（这曾是
一个真实踩过的坑：`ThreadDataMiddleware`/`SandboxMiddleware` 早期实现直接把
`ThreadWorkspace`/`Sandbox` 实例塞进 state，上线后第一次真实请求就在
`checkpointer.aput_writes` 时炸掉）。两者现在都改成把资源存在中间件实例的
私有属性上（见各自模块说明），不再经过 state/checkpoint 这条路径，因此本模块
只保留一个真正是"纯数据"的字段。
"""
from __future__ import annotations

from langchain.agents.middleware.types import AgentState
from typing_extensions import NotRequired


class PipelineState(AgentState):
    """在官方 `AgentState`（`messages` 等）基础上追加的骨架专用字段。

    Attributes:
        recent_tool_calls: 由 LoopDetectionMiddleware 维护的最近工具调用
            签名列表（`f"{tool_name}:{sorted(args.items())}"`，纯字符串，
            可安全序列化），用于检测连续重复调用。
    """

    recent_tool_calls: NotRequired[list[str]]
