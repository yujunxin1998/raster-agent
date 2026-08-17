"""技能注册表（内存索引）。

原样迁移自 `src/core/skills/registry.py`，仅补充类型注解层面的常量引用。
"""
from __future__ import annotations

from src.agent_core.skills.skill_definition import SkillDefinition
from src.common.exceptions import DuplicateSkillError, SkillNotFoundError


class SkillRegistry:
    """技能的内存索引，支持按 name 精确查询、按 category 批量查询。

    典型用法::

        registry.get("knowledge-base-answering")   # 按 name 取单个技能
        registry.by_category("rag")                # 取该分类下全部技能
        registry.all / registry.names               # 获取全量信息

    Attributes:
        revision: 单调递增的版本号，普通可变属性——本类语义是"整体替换引用"
            （`SkillHotReloader.reload_once()` 重载成功后整个换掉
            `SkillManager.registry` 指向的实例，不原地修改旧实例），不需要
            `RegistrySnapshot` 那套不可变 copy-on-write 机制。默认 0，
            由持有"当前状态"引用的调用方（`SkillHotReloader`）在每次成功
            重载后自增，供 `subagent_capability_resolver.py` 等需要追溯
            "当时用的是哪个版本"的场景引用。
    """

    def __init__(self) -> None:
        self._store: dict[str, SkillDefinition] = {}
        self.revision: int = 0

    def register(self, skill: SkillDefinition) -> None:
        """注册一个技能，同名 `name` 显式拒绝，不静默覆盖。

        原实现"后加载的生效"依赖 `SkillLoader` 目录扫描的 `sorted()` 顺序，
        本质是未定义行为伪装成确定性——`SkillLoader.load()` 单个技能解析
        失败已经是"记 error 日志 + 跳过"的容错策略，这里抛出的
        `DuplicateSkillError` 会被同一个 `except Exception` 捕获，效果是
        "两个同名技能都不注册、只留错误日志"，不会导致启动失败。

        Args:
            skill: 待注册的技能定义。

        Raises:
            DuplicateSkillError: 已存在同名 `name` 的技能。
        """
        if skill.name in self._store:
            raise DuplicateSkillError(f"技能 name 冲突: {skill.name}")
        self._store[skill.name] = skill

    def get(self, name: str) -> SkillDefinition:
        """按 name 精确查询。

        Args:
            name: 技能标识。

        Returns:
            对应的 SkillDefinition。

        Raises:
            SkillNotFoundError: 未找到该 name 对应的技能。
        """
        if name not in self._store:
            raise SkillNotFoundError(f"技能 '{name}' 未找到。已注册：{self.names}")
        return self._store[name]

    def __getitem__(self, name: str) -> SkillDefinition:  # noqa: D105
        return self.get(name)

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
