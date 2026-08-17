"""Skill 机制模块。

使用方式::

    from src.agent_core.skills import get_skill_manager

    get_skill_manager().registry.by_category("rag")   # 取某分类下全部技能
    get_skill_manager().registry.names                 # 已注册的全部技能名

新增技能只需在 skills/core/ 或 skills/public/ 下新建一个带 category 字段的
SKILL.md，无需改动任何代码——`SkillMiddleware` 每次模型调用都现算目录/预
路由，写完文件（或触发热重载）下一次调用就会生效。

`docs/Skill与Tool完全解耦重构设计.md` 第 15 节阶段 6 之后：`SkillManager`
不再依赖 GuardrailProvider/SandboxProvider（那是 Tool 执行层的关注点，见
`skill_middleware.py` 模块文档"三个独立平面"），初始化不再需要等这两者就绪，
但仍保留显式的 `init_skill_manager()`（而不是模块导入时就地构建），跟
`MemoryManager` 的初始化方式（`init_memory_manager`）保持同一种模式，方便
测试场景按需跳过。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_catalog import build_skill_catalog
from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_file_tree_reader import SkillFileTreeReader
from src.agent_core.skills.skill_load_tool import create_load_skill_tool
from src.agent_core.skills.skill_loader import SkillLoader, resolve_skill_dirs
from src.agent_core.skills.skill_manager import SkillManager
from src.agent_core.skills.skill_path_rewriter import SkillPathRewriter
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.constants import SkillCategory

_manager: SkillManager | None = None


def init_skill_manager(skills_dirs_setting: str, enabled: bool) -> SkillManager:
    """应用启动时调用一次，扫描技能目录并注册全局单例。

    Args:
        skills_dirs_setting: 逗号分隔的技能目录配置，对应 `SKILLS_DIRS`。
        enabled: 功能总开关，对应 `SKILLS_ENABLED`。

    Returns:
        创建好的 SkillManager 实例，同时也已注册为全局单例。
    """
    global _manager
    _manager = SkillManager(resolve_skill_dirs(skills_dirs_setting), enabled=enabled)

    category_counts = {
        category.value: len(_manager.registry.by_category(category.value)) for category in SkillCategory
    }
    logger.info(
        f"[SkillManager] 就绪 | enabled={_manager.enabled} "
        f"| 共 {len(_manager.registry)} 个技能 | categories={category_counts}"
    )
    return _manager


def get_skill_manager() -> SkillManager:
    """返回全局唯一的 SkillManager 实例。

    Raises:
        RuntimeError: init_skill_manager() 尚未被调用。
    """
    if _manager is None:
        raise RuntimeError("SkillManager 尚未初始化，请确认应用已完成启动")
    return _manager


__all__ = [
    "init_skill_manager",
    "get_skill_manager",
    "SkillManager",
    "SkillDefinition",
    "SkillRegistry",
    "SkillLoader",
    "SkillFileTreeReader",
    "SkillActivationService",
    "SkillPathRewriter",
    "create_load_skill_tool",
    "build_skill_catalog",
    "resolve_skill_dirs",
]
