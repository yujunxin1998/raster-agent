"""权限控制抽象接口。

对应设计文档 5.1 节：把原先散落在 `SkillAccessGuard` 里的"用户是否关闭了
某个技能"这一层校验，升级为统一的、可插拔的权限控制层。`GuardrailProvider`
是这层的核心协议，`AllowlistGuardrailProvider` 是内置的默认实现。

拒绝时的行为约定（与原 `SkillAccessGuard`/`SkillToolFactory._invoke` 的
容错哲学保持一致）：调用方拿到 DENY 决策后应当返回一段说明文本给 Agent
继续推理，而不是抛异常中断整轮对话——权限拒绝是"这次不能做这件事"，
不是"系统出错了"。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.common.constants import GuardrailDecisionType

_DEFAULT_DENY_REASON = "未获授权执行"


@dataclass(frozen=True)
class GuardrailDecision:
    """一次权限校验的结果。

    Attributes:
        decision_type: ALLOW 或 DENY。
        reason: 拒绝原因说明；ALLOW 时通常为空字符串。
    """

    decision_type: GuardrailDecisionType
    reason: str = ""

    @property
    def is_allowed(self) -> bool:
        """是否放行。"""
        return self.decision_type == GuardrailDecisionType.ALLOW

    @classmethod
    def allow(cls) -> "GuardrailDecision":
        """构造一个放行决策。"""
        return cls(decision_type=GuardrailDecisionType.ALLOW)

    @classmethod
    def deny(cls, reason: str = _DEFAULT_DENY_REASON) -> "GuardrailDecision":
        """构造一个拒绝决策。

        Args:
            reason: 拒绝原因，会原样透传给 Agent 作为工具返回值的一部分。
        """
        return cls(decision_type=GuardrailDecisionType.DENY, reason=reason)


class GuardrailContext:
    """一次权限校验所需的上下文信息（值对象）。

    Attributes:
        conversation_id: 会话 ID，可为空（内部调用/测试场景）。
        thinking: 是否处于深度思考模式，供未来更细粒度的策略引用。
        extra: 其余按需扩展的上下文字段（如 datasource_id），避免每新增
            一个字段就要改一次 GuardrailProvider 的方法签名。
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        thinking: bool = False,
        extra: dict | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.thinking = thinking
        self.extra = extra or {}


class GuardrailProvider(ABC):
    """权限校验的抽象接口。

    设计为可插拔协议是为了以后接入更复杂的策略（按角色的 RBAC、按数据源的
    行级权限等）时不用改中间件本身，只需要换一个 Provider 实现——这一点
    直接对应设计文档里"内置 AllowlistProvider，也可接第三方策略引擎"的说法。
    """

    @abstractmethod
    async def check(
        self,
        *,
        user_id: str | None,
        tool_name: str,
        tool_args: dict,
        context: GuardrailContext,
    ) -> GuardrailDecision:
        """对一次工具调用做权限校验。

        Args:
            user_id: 发起调用的用户 ID，为空时代表内部调用/测试场景。
            tool_name: 待调用的工具名（可以是普通后端工具，也可以是技能工具）。
            tool_args: 工具调用参数，供需要参数级校验的策略实现使用（内置的
                AllowlistGuardrailProvider 不使用这个字段，只做工具名级校验）。
            context: 校验所需的上下文信息。

        Returns:
            GuardrailDecision，ALLOW 或 DENY（附带原因）。
        """
