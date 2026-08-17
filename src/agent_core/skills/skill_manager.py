"""技能机制对外的唯一入口。

`docs/Skill与Tool完全解耦重构设计.md` 第 15 节阶段 6 收尾：`SkillToolFactory`
已删除（不再有任何技能生成独立 `StructuredTool`），`SkillManager` 现在只是
`SkillRegistry` 的一层薄封装 + 热重载可原地替换的落脚点，不再持有
guardrail/sandbox 相关的工具装配依赖。
"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_registry import SkillRegistry


class SkillManager:
    """Skill 机制对外的唯一入口，持有当前生效的 SkillRegistry。"""

    def __init__(self, skill_dirs: list[Path], enabled: bool) -> None:
        """初始化技能管理器。

        Args:
            skill_dirs: 技能根目录列表。
            enabled: 功能总开关，False 时持有空注册表。
        """
        self.enabled = enabled
        self.registry: SkillRegistry = SkillLoader(skill_dirs).load() if enabled else SkillRegistry()
