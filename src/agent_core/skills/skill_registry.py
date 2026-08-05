"""技能注册表（内存索引）。

原样迁移自 `src/core/skills/registry.py`，仅补充类型注解层面的常量引用。
"""
from __future__ import annotations

from src.agent_core.skills.skill_definition import SkillDefinition
from src.common.exceptions import SkillNotFoundError


class SkillRegistry:
    """技能的内存索引，支持按 tool_name 精确查询、按 category 批量查询。

    典型用法::

        registry.get("search_knowledge_base")   # 按 tool_name 取单个技能
        registry.by_category("rag")             # 取该分类下全部技能
        registry.all / registry.names            # 获取全量信息
    """

    def __init__(self) -> None:
        self._store: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """注册一个技能，同名 tool_name 会被覆盖（后加载的生效）。

        Args:
            skill: 待注册的技能定义。
        """
        self._store[skill.tool_name] = skill

    def get(self, tool_name: str) -> SkillDefinition:
        """按 tool_name 精确查询。

        Args:
            tool_name: 技能对应的工具名。

        Returns:
            对应的 SkillDefinition。

        Raises:
            SkillNotFoundError: 未找到该 tool_name 对应的技能。
        """
        if tool_name not in self._store:
            raise SkillNotFoundError(f"技能 '{tool_name}' 未找到。已注册：{self.names}")
        return self._store[tool_name]

    def __getitem__(self, tool_name: str) -> SkillDefinition:  # noqa: D105
        return self.get(tool_name)

    def by_category(self, category: str) -> list[SkillDefinition]:
        """按分类批量查询。

        Args:
            category: 技能分类。

        Returns:
            该分类下的全部技能定义列表，不存在时返回空列表。
        """
        return [skill for skill in self._store.values() if skill.category == category]

    @property
    def all(self) -> list[SkillDefinition]:
        """全部已注册技能。"""
        return list(self._store.values())

    @property
    def names(self) -> list[str]:
        """全部已注册技能的 tool_name（升序排列）。"""
        return sorted(self._store.keys())

    def __len__(self) -> int:  # noqa: D105
        return len(self._store)

    def __repr__(self) -> str:  # noqa: D105
        return f"SkillRegistry({self.names})"
