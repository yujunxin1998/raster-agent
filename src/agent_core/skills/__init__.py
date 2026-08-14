"""Skill 机制模块。

使用方式::

    from src.agent_core.skills import get_skill_manager

    get_skill_manager().get_tools("rag")          # 取某分类下全部技能工具
    get_skill_manager().registry.names            # 已注册的全部 tool_name

新增技能只需在 skills/core/ 或 skills/public/ 下新建一个带 category 字段的
SKILL.md，无需改动任何代码——对应分类下次调用 get_tools() 时会自动包含它
（因为 SkillLoader 在应用启动阶段就已扫描完毕，写入 SkillRegistry）。

与原项目 `src/core/skills/__init__.py` 的差异：原实现在模块导入时就地构建
`skill_manager` 全局单例；本工程的 SkillManager 依赖 GuardrailProvider 和
SandboxProvider（两者都需要数据库连接池/工作区配置就绪后才能创建），因此
改为显式的 `init_skill_manager()`，在 `main.py` 的 lifespan 中，等
guardrail/sandbox 基础设施初始化完成后再调用，与 `MemoryManager` 的初始化
方式（`init_memory_manager`）保持同一种模式。
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agent_core.guardrail.guardrail_provider import GuardrailProvider
from src.agent_core.sandbox.sandbox_provider import SandboxProvider
from src.agent_core.skills.skill_activation_service import SkillActivationService
from src.agent_core.skills.skill_catalog import build_skill_catalog
from src.agent_core.skills.skill_definition import SkillDefinition, SkillKind
from src.agent_core.skills.skill_file_tree_reader import SkillFileTreeReader
from src.agent_core.skills.skill_load_tool import create_load_skill_tool
from src.agent_core.skills.skill_loader import SkillLoader, resolve_skill_dirs
from src.agent_core.skills.skill_manager import SkillManager
from src.agent_core.skills.skill_path_rewriter import SkillPathRewriter
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.skills.skill_tool_factory import SkillParameterSchemaBuilder, SkillToolFactory
from src.common.constants import SkillCategory

_manager: SkillManager | None = None


def init_skill_manager(
    skills_dirs_setting: str,
    enabled: bool,
    guardrail_provider: GuardrailProvider,
    sandbox_provider: SandboxProvider,
    skill_script_timeout_seconds: int,
) -> SkillManager:
    """应用启动时调用一次，扫描技能目录并注册全局单例。

    Args:
        skills_dirs_setting: 逗号分隔的技能目录配置，对应 `SKILLS_DIRS`。
        enabled: 功能总开关，对应 `SKILLS_ENABLED`。
        guardrail_provider: 权限校验器。
        sandbox_provider: 沙箱提供者。
        skill_script_timeout_seconds: 脚本执行超时时间。

    Returns:
        创建好的 SkillManager 实例，同时也已注册为全局单例。
    """
    global _manager
    _manager = SkillManager(
        resolve_skill_dirs(skills_dirs_setting),
        enabled=enabled,
        guardrail_provider=guardrail_provider,
        sandbox_provider=sandbox_provider,
        skill_script_timeout_seconds=skill_script_timeout_seconds,
    )

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
    "SkillKind",
    "SkillRegistry",
    "SkillLoader",
    "SkillToolFactory",
    "SkillParameterSchemaBuilder",
    "SkillFileTreeReader",
    "SkillActivationService",
    "SkillPathRewriter",
    "create_load_skill_tool",
    "build_skill_catalog",
    "resolve_skill_dirs",
]
