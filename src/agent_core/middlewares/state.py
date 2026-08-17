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

from typing import Annotated, Optional

from langchain.agents.middleware.types import AgentState
from typing_extensions import NotRequired

# 与 LoopDetectionMiddleware 的"连续重复调用"判断窗口对应——保留太少条历史会让
# 判断失真，保留太多则让每步 checkpoint 体积/序列化成本上升，两者需要保持同一个值。
RECENT_TOOL_CALLS_WINDOW = 10


def _replace_plan(current: Optional[list[dict]], new: list[dict]) -> list[dict]:
    """`plan` 的合并函数（reducer）：整体替换语义，不是 `recent_tool_calls` 那种
    追加。

    `update_plan` 工具每次调用都传入模型认为的完整计划（不是增量），所以正常
    情况下直接取最新值就够。单独声明这个 reducer 只是防御性的——万一模型在
    同一步里把 `update_plan` 和其它工具一起并行调用，默认的 `LastValue` channel
    一步只能接受一次写入会直接抛 `InvalidUpdateError`（`recent_tool_calls`
    真实踩过的坑，见 `_append_recent_tool_calls`）；`current` 参数在这里其实用
    不上，只是为了保持跟 `_append_recent_tool_calls` 一致的 reducer 签名。

    Args:
        current: 上一步结束时的计划（未写过时为 None，本函数不使用）。
        new: 本步提交的完整新计划。

    Returns:
        `new` 本身——并发写入时以最后处理到的一次为准。
    """
    return new


def _merge_activated_skills(current: Optional[dict[str, dict]], new: dict[str, dict]) -> dict[str, dict]:
    """`activated_skills` 的合并函数（reducer）：按 skill_name 做字典合并（增量）。

    与 `_append_recent_tool_calls` 同样的并发写入问题（`SkillMiddleware.
    awrap_model_call` 一次可能激活多个技能，`awrap_tool_call` 拦截
    `load_skill` 补充激活时又是另一个可能并行的分支）——每个分支只提交自己
    这一次新增/更新的条目（`new` 是一个只含本次变化的小字典，不是全量重算），
    交给这个 reducer 合并进上一步的完整状态，多值合并变成结合律操作。

    Args:
        current: 上一步结束时已激活的技能状态（`skill_name -> ActivatedSkill`
            纯 dict，尚未写过时为 None）。
        new: 本步某一个分支新增/更新的条目（只包含这次变化的 key）。

    Returns:
        合并后的完整激活状态字典，同名 key 以 `new` 为准（重复激活/版本更新
        场景下覆盖旧记录）。
    """
    return {**(current or {}), **new}


def _append_recent_tool_calls(current: Optional[list[str]], new: list[str]) -> list[str]:
    """`recent_tool_calls` 的合并函数（reducer）。

    默认的 `LastValue` channel 一步内只能接受一次写入——Lead Agent 一次模型响应
    可能并行发起多个工具调用（各自独立经过 `LoopDetectionMiddleware.
    awrap_tool_call`），若每个分支都各自基于同一份起始 state 重算"追加后的完整
    列表"再整体写回，多个分支的写入会互相冲突，LangGraph 直接抛
    `InvalidUpdateError`（真实报错："At key 'recent_tool_calls': Can receive
    only one value per step"）。改成每个分支只提交自己这一条新签名（`new` 恒为
    单元素列表），交给这个 reducer 把本步所有并行分支的新签名依次追加到上一步
    的历史后面，再从右侧截断到窗口大小——多值合并变成结合律操作，天然支持并行
    写入。

    Args:
        current: 上一步结束时的历史（`recent_tool_calls` 尚未写过时为 None）。
        new: 本步某一个分支提交的新增签名（`LoopDetectionMiddleware` 每次只提交
            长度为 1 的列表）。

    Returns:
        合并并截断到 `RECENT_TOOL_CALLS_WINDOW` 条以内的历史列表。
    """
    return ((current or []) + new)[-RECENT_TOOL_CALLS_WINDOW:]


class PipelineState(AgentState):
    """在官方 `AgentState`（`messages` 等）基础上追加的骨架专用字段。

    Attributes:
        recent_tool_calls: 由 LoopDetectionMiddleware 维护的最近工具调用
            签名列表（`f"{tool_name}:{sorted(args.items())}"`，纯字符串，
            可安全序列化），用于检测连续重复调用。带 `_append_recent_tool_calls`
            reducer，支持同一步内多个并行工具调用各自写入而不冲突（见其文档）。
        plan: Lead Agent 自己维护的任务执行计划（`update_plan` 工具写入，
            `PlanContextMiddleware` 读出注入进每次模型调用的 system_prompt）。
            每个元素是 `{"content": str, "status": "pending"|"in_progress"|
            "completed"}` 纯 dict——跟 `recent_tool_calls` 一样要过 msgpack
            checkpoint 序列化，不能是 pydantic 对象。没建过计划时不存在这个 key
            （不是空列表），`PlanContextMiddleware`/`get()` 调用方都按"key 不存在
            或空列表"同等对待。
        activated_skills: `SkillMiddleware` 维护的"本次 Agent 运行已经激活过
            哪些技能"记录（重构文档 7.4 节），键是 `skill_name`，值是纯 dict
            `{"skill_version": str | None, "activation_source": "explicit"|
            "required"|"catalog"|"load_skill", "activated_at": str}`——同一个
            技能同一个版本已经激活过时不重复把正文注入 system_prompt/消息，
            只返回轻量的"已激活"提示。带 `_merge_activated_skills` reducer，
            语义与 `recent_tool_calls` 一致：支持同一步内多个并行分支各自写入
            不冲突。
    """

    recent_tool_calls: NotRequired[Annotated[list[str], _append_recent_tool_calls]]
    plan: NotRequired[Annotated[list[dict], _replace_plan]]
    activated_skills: NotRequired[Annotated[dict[str, dict], _merge_activated_skills]]
