"""内置的白名单式权限控制实现。

零外部依赖（不接第三方策略引擎），组合两张表完成校验：

    - `SkillSettingsStore`（`user_skill_settings` 表）：技能级开关，原样复用
      迁移前 `SkillAccessGuard` 的判断逻辑与语义。
    - `ToolPermissionStore`（`user_tool_permissions` 表，新增）：更广义的
      工具级权限，覆盖非技能类工具。

两张表任一命中"拒绝"即整体拒绝；都没有记录时默认放行——延续原项目
"没有 user_id（内部调用/测试场景）时默认放行，只有明确知道是哪个用户在
调用，才有'用户级开关'的意义"这一设计取舍。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.guardrail.guardrail_provider import (
    GuardrailContext,
    GuardrailDecision,
    GuardrailProvider,
)
from src.storage.skill_settings_store import SkillSettingsStore
from src.storage.tool_permission_store import ToolPermissionStore

_SKILL_DISABLED_REASON_TEMPLATE = "技能 [{tool_name}] 已被用户在设置中关闭，本次未执行。"
_TOOL_DENIED_REASON_TEMPLATE = "工具 [{tool_name}] 未获授权执行，本次未执行。"


class AllowlistGuardrailProvider(GuardrailProvider):
    """基于两张持久化表的白名单校验实现。"""

    def __init__(
        self,
        skill_settings_store: SkillSettingsStore,
        tool_permission_store: ToolPermissionStore,
    ) -> None:
        """初始化白名单权限校验器。

        Args:
            skill_settings_store: 技能级开关存储。
            tool_permission_store: 工具级权限存储。
        """
        self._skill_settings_store = skill_settings_store
        self._tool_permission_store = tool_permission_store

    async def check(
        self,
        *,
        user_id: str | None,
        tool_name: str,
        tool_args: dict,
        context: GuardrailContext,
    ) -> GuardrailDecision:
        if not tool_name:
            raise ValueError("tool_name 不能为空")

        if not user_id:
            # 没有 user_id 即无法归属到具体用户，用户级开关无从谈起，默认放行。
            return GuardrailDecision.allow()

        try:
            if await self._skill_settings_store.is_disabled(user_id, tool_name):
                return GuardrailDecision.deny(_SKILL_DISABLED_REASON_TEMPLATE.format(tool_name=tool_name))

            if await self._tool_permission_store.is_denied(user_id, tool_name):
                return GuardrailDecision.deny(_TOOL_DENIED_REASON_TEMPLATE.format(tool_name=tool_name))
        except Exception as exc:
            # 权限存储不可用属于基础设施异常，不应该让"查权限"这件事本身
            # 拖垮整轮对话——按原 SkillAccessGuard 的取舍，异常时默认放行，
            # 但必须记录 warning，否则权限失效会在无声无息中发生。
            logger.warning(
                f"[AllowlistGuardrailProvider] 权限校验异常，默认放行 "
                f"user_id={user_id} tool_name={tool_name} error={exc}"
            )
            return GuardrailDecision.allow()

        return GuardrailDecision.allow()


_provider: AllowlistGuardrailProvider | None = None


def init_guardrail_provider(
    skill_settings_store: SkillSettingsStore,
    tool_permission_store: ToolPermissionStore,
) -> AllowlistGuardrailProvider:
    """应用启动时调用一次，完成全局单例初始化。"""
    global _provider
    _provider = AllowlistGuardrailProvider(skill_settings_store, tool_permission_store)
    logger.info("[AllowlistGuardrailProvider] 初始化完成")
    return _provider


def get_guardrail_provider() -> AllowlistGuardrailProvider:
    """返回全局唯一的权限校验器实例。

    Raises:
        RuntimeError: init_guardrail_provider() 尚未被调用。
    """
    if _provider is None:
        raise RuntimeError("GuardrailProvider 尚未初始化，请确认应用已完成启动")
    return _provider
