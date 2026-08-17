"""`validate_middleware_order()` 单元测试：中间件顺序约束的硬校验。"""
from __future__ import annotations

import pytest

from src.agent_core.agents.middleware_order import ORDER_CONSTRAINTS, validate_middleware_order
from src.agent_core.agents.plan_middleware import PlanContextMiddleware
from src.agent_core.agents.skill_middleware import SkillMiddleware
from src.agent_core.agents.streaming_model_middleware import StreamingModelMiddleware
from src.agent_core.middlewares.guardrail import GuardrailMiddleware
from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.tool_audit import ToolAuditMiddleware
from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware


def _bare(cls):
    """构造一个不经过 `__init__` 的空实例——`validate_middleware_order` 只按
    `type(m)` 判断，不访问实例属性，用真实构造函数（部分需要重依赖，如
    `SkillMiddleware` 要 `SkillManager`）纯属浪费。"""
    return cls.__new__(cls)


def _lead_agent_style_order() -> list:
    """镜像 `lead_agent.py` 里真实的中间件列表顺序（只取跟顺序约束相关的
    几个类型，其余 8 个通用中间件跟这几条约束无关，不影响校验结果）。"""
    return [
        _bare(GuardrailMiddleware),
        _bare(ToolAuditMiddleware),
        _bare(ToolErrorHandlingMiddleware),
        _bare(LoopDetectionMiddleware),
        _bare(SkillMiddleware),
        _bare(PlanContextMiddleware),
        _bare(StreamingModelMiddleware),
    ]


def test_lead_agent_order_satisfies_all_constraints() -> None:
    """当前 lead_agent.py 里的真实顺序应该完全满足 ORDER_CONSTRAINTS。"""
    validate_middleware_order(_lead_agent_style_order())


def test_missing_middleware_types_are_skipped_not_errored() -> None:
    """约束涉及的类型没有全部出现在列表里时（比如测试场景只装了部分中间件），
    不应该报错——只校验"两者都出现时"的相对顺序。"""
    validate_middleware_order([_bare(GuardrailMiddleware)])


@pytest.mark.parametrize("outer_cls,inner_cls", ORDER_CONSTRAINTS)
def test_violating_any_single_constraint_raises(outer_cls, inner_cls) -> None:
    """把 ORDER_CONSTRAINTS 里任意一条约束的相对顺序颠倒，都应该被拒绝。"""
    violating_order = [_bare(inner_cls), _bare(outer_cls)]

    with pytest.raises(RuntimeError, match="中间件顺序错误"):
        validate_middleware_order(violating_order)
