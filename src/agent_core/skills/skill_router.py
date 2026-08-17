"""Skill 预路由：把"这个技能要不要在模型第一次调用前就激活"变成确定性规则判断。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 7.1、7.2 节："生产路径不得只
依赖主模型主动调用 load_skill"——单纯把 Skill Catalog 放进 System Prompt，
可能出现任务命中了某个技能、但模型认为自己能直接完成而跳过 load_skill 的
情况。`SkillMiddleware` 在每次模型调用前先跑一遍 `route()`，`forced` 结果
在**首次模型调用前**就把正文注入 system_prompt，不依赖模型自己判断"要不要
查目录、要不要调用 load_skill"。

**匹配方式的范围收窄（已与用户确认，仅第一阶段）**：文档 7.2 节把匹配方式
留为开放项（规则匹配 / Embedding 召回 / 轻量 LLM 重排）。这里只实现规则/
元数据匹配——不引入额外的 Embedding 或 LLM 调用：LLM 重排会在每轮对话首次
模型调用前插入一次额外 LLM 调用，与项目已经明确废弃的 Supervisor"每轮强制
路由决策调用"同性质（只是频率低得多）；Embedding 需要额外的向量化基础设施
且没有评测集可以校准阈值。文档本身也说"阈值不得硬编码为未经验证的经验
值……上线阈值必须由评测集校准"——在没有评测数据的第一阶段，规则匹配是唯一
不需要臆造置信度分数的选项。`scores`/`reasons` 字段保留、返回确定性值
（1.0/0.0），只是为了不阻塞未来接入 Embedding/LLM 时的接口兼容，本阶段不
做任何语义相似度计算。

`route()` 只负责"要不要激活"这个纯规则判断，不做权限校验——权限校验
（用户是否被允许加载某个具体技能）是 `SkillMiddleware` 在实际调用
`SkillActivationService.activate()` 前单独用 `GuardrailProvider` 做的一步，
两者正交（重构文档 11 节"Skill 激活权限"）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.agent_core.skills.skill_definition import SkillDefinition

_REASON_EXPLICIT = "explicit"
_REASON_REQUIRED = "required"
_REASON_EXPLICIT_ONLY_NOT_SPECIFIED = "explicit_only_not_specified"
_REASON_CATALOG_CANDIDATE = "catalog_candidate"

_SCORE_MATCHED = 1.0
_SCORE_NOT_MATCHED = 0.0


@dataclass(frozen=True)
class SkillRoutingDecision:
    """一次预路由的结构化结果（对应重构文档 7.2 节）。

    Attributes:
        forced: 用户/API 显式指定或 Agent 必需（`activation == "required"`），
            权限通过后必须加载的技能名集合。
        auto_activate: 路由器判断完成本任务需要该技能、达到自动激活阈值的
            技能名集合。规则路由阶段恒等于 `forced`（见模块文档说明）。
        catalog_candidates: 可能相关但未被强制/自动激活，只向主模型展示轻量
            目录、留给模型按需 `load_skill` 补充加载的技能名集合。
        rejected: 不相关、无权限、依赖缺失或激活策略不允许的技能名集合
            （规则路由阶段仅覆盖"`explicit_only` 但未被显式指定"这一种）。
        scores: 每个技能名对应的路由置信度，规则路由阶段恒为确定性值
            （命中 1.0 / 未命中 0.0），供未来 Embedding/LLM 路由复用同一接口。
        reasons: 每个技能名对应的路由原因，用于日志与审计。
    """

    forced: tuple[str, ...]
    auto_activate: tuple[str, ...]
    catalog_candidates: tuple[str, ...]
    rejected: tuple[str, ...]
    scores: Mapping[str, float]
    reasons: Mapping[str, str]


def route(
    skills: list[SkillDefinition],
    *,
    explicit_skill_names: frozenset[str] = frozenset(),
) -> SkillRoutingDecision:
    """对一组当前 Agent 可见的技能做一次纯规则预路由。

    Args:
        skills: 当前 Agent 可见的技能列表（调用方已经按 `allowed_categories`
            过滤过可见范围，这里不重复做分类过滤）。
        explicit_skill_names: 用户/API 本次请求显式指定的技能名集合。

    Returns:
        结构化的路由决策，`forced ∪ auto_activate ∪ catalog_candidates ∪
        rejected` 覆盖 `skills` 里的每一个技能名且互不重叠。
    """
    forced: list[str] = []
    catalog_candidates: list[str] = []
    rejected: list[str] = []
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    for skill in skills:
        name = skill.name
        if name in explicit_skill_names:
            forced.append(name)
            scores[name] = _SCORE_MATCHED
            reasons[name] = _REASON_EXPLICIT
        elif skill.activation == "required":
            forced.append(name)
            scores[name] = _SCORE_MATCHED
            reasons[name] = _REASON_REQUIRED
        elif skill.activation == "explicit_only":
            rejected.append(name)
            scores[name] = _SCORE_NOT_MATCHED
            reasons[name] = _REASON_EXPLICIT_ONLY_NOT_SPECIFIED
        else:
            catalog_candidates.append(name)
            scores[name] = _SCORE_NOT_MATCHED
            reasons[name] = _REASON_CATALOG_CANDIDATE

    forced_tuple = tuple(forced)
    return SkillRoutingDecision(
        forced=forced_tuple,
        # 规则路由阶段：auto_activate 恒等于 forced，见模块文档"匹配方式的范围收窄"。
        auto_activate=forced_tuple,
        catalog_candidates=tuple(catalog_candidates),
        rejected=tuple(rejected),
        scores=scores,
        reasons=reasons,
    )
