"""技能机制对外的唯一入口。

原样迁移自 `src/core/skills/manager.py`，构造参数新增
guardrail_provider/sandbox_provider/skill_script_timeout_seconds，
用于装配 SkillToolFactory（详见设计文档 5.1、5.3 节）。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from src.agent_core.guardrail.guardrail_provider import GuardrailProvider
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.skills.skill_tool_factory import SkillToolFactory


class SkillManager:
    """Skill 机制对外的唯一入口。

    内部持有 SkillRegistry（轻量索引）+ SkillToolFactory（按需建工具）。
    典型用法::

        tools = [..., *skill_manager.get_tools("rag")]
    """

    def __init__(
        self,
        skill_dirs: list[Path],
        enabled: bool,
        guardrail_provider: GuardrailProvider,
        sandbox_provider: SandboxProvider,
        skill_script_timeout_seconds: int,
    ) -> None:
        """初始化技能管理器。

        Args:
            skill_dirs: 技能根目录列表。
            enabled: 功能总开关，False 时持有空注册表，get_tools() 恒返回空列表。
            guardrail_provider: 权限校验器，透传给 SkillToolFactory。
            sandbox_provider: 沙箱提供者，透传给 SkillToolFactory。
            skill_script_timeout_seconds: 脚本执行超时时间。
        """
        self.enabled = enabled
        self.registry: SkillRegistry = SkillLoader(skill_dirs).load() if enabled else SkillRegistry()
        self._factory = SkillToolFactory(
            guardrail_provider=guardrail_provider,
            sandbox_provider=sandbox_provider,
            skill_script_timeout_seconds=skill_script_timeout_seconds,
        )

    def get_tools(self, category: str) -> list[StructuredTool]:
        """返回某个分类下全部技能对应的 StructuredTool。

        每次调用都现造工具（不缓存正文）——渐进式披露落地在这里：
        SkillDefinition 本身一直很轻，重内容只在工具真正被执行时经
        SkillContentReader 现读。

        Args:
            category: 技能分类。

        Returns:
            该分类下全部技能包装成的 StructuredTool 列表。
        """
        if not self.enabled:
            return []
        return [self._factory.create(skill) for skill in self.registry.by_category(category)]
