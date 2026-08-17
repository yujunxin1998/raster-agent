"""扫描技能目录、解析 frontmatter，构建 SkillRegistry。

原样迁移自 `src/core/skills/loader.py`：启动阶段只解析 frontmatter，
不读取 SKILL.md 正文——渐进式披露的"轻"那一半。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from loguru import logger

from src.agent_core.skills.skill_definition import (
    _DEFAULT_ACTIVATION_MODE,
    _VALID_ACTIVATION_MODES,
    SkillActivationMode,
    SkillDefinition,
)
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.constants import SkillCategory
from src.common.exceptions import SkillDefinitionInvalidError

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VALID_CATEGORIES = {category.value for category in SkillCategory}
_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
#: 已删除字段（重构文档第 6、15 节，`SkillDefinition` 不再有对应字段）。
#: 出现时只记 warning，不影响加载——给还没更新 frontmatter 的 SKILL.md 一个
#: 软过渡，不因为写了废弃字段就直接加载失败。
_DEPRECATED_FRONTMATTER_KEYS = ("tool_name", "parameters", "runtime_context_keys", "required_secrets")


class SkillLoader:
    """扫描 `skill_dirs` 下所有 `*/SKILL.md`，产出轻量的 SkillDefinition 列表。"""

    def __init__(self, skill_dirs: list[Path]) -> None:
        """初始化加载器。

        Args:
            skill_dirs: 待扫描的技能根目录列表，如
                `[Path("skills/core"), Path("skills/public")]`。
        """
        self._skill_dirs = skill_dirs

    def load(self) -> SkillRegistry:
        """执行一次全量扫描，返回构建好的 SkillRegistry。

        单个技能解析失败不会中断整体加载，只记 error 日志并跳过该技能，
        保证一个格式错误的 SKILL.md 不会导致其它技能全部加载失败。

        Returns:
            已注册全部合法技能的 SkillRegistry。
        """
        registry = SkillRegistry()

        for skill_dir_root in self._skill_dirs:
            if not skill_dir_root.exists():
                logger.warning(f"[SkillLoader] 目录不存在: {skill_dir_root}")
                continue

            for skill_md in sorted(skill_dir_root.glob("*/SKILL.md")):
                skill_dir = skill_md.parent
                try:
                    frontmatter = self._parse_frontmatter(skill_md)
                    skill = self._build_definition(skill_dir, frontmatter)
                    registry.register(skill)
                    logger.debug(f"[SkillLoader] 已加载: {skill.name} <- {skill_md}")
                except Exception as exc:
                    logger.error(f"[SkillLoader] 加载失败: {skill_md} - {exc}")

        logger.info(f"[SkillLoader] 注册完成，共 {len(registry)} 条: {registry.names}")
        return registry

    def _parse_frontmatter(self, skill_md: Path) -> dict:
        """解析 SKILL.md 头部的 YAML frontmatter。

        Args:
            skill_md: SKILL.md 文件路径。

        Returns:
            解析后的 frontmatter 字典。

        Raises:
            SkillDefinitionInvalidError: 缺少合法的 frontmatter 或不是 YAML 字典。
        """
        text = skill_md.read_text(encoding="utf-8")
        match = _FRONTMATTER_PATTERN.match(text)
        if not match:
            raise SkillDefinitionInvalidError(f"{skill_md} 缺少合法的 YAML frontmatter")

        frontmatter = yaml.safe_load(match.group(1))
        if not isinstance(frontmatter, dict):
            raise SkillDefinitionInvalidError(f"{skill_md} frontmatter 必须是 YAML 字典")
        return frontmatter

    def _build_definition(self, skill_dir: Path, frontmatter: dict) -> SkillDefinition:
        """把 frontmatter 字典组装为 SkillDefinition。

        Args:
            skill_dir: 技能所在目录。
            frontmatter: 已解析的 frontmatter 字典。

        Returns:
            组装好的 SkillDefinition。
        """
        for deprecated_key in _DEPRECATED_FRONTMATTER_KEYS:
            if deprecated_key in frontmatter:
                logger.warning(
                    f"[SkillLoader] {skill_dir.name} 的 frontmatter 包含已废弃字段 "
                    f"{deprecated_key!r}，已忽略（SkillDefinition 不再有对应字段）"
                )

        name = str(frontmatter.get("name") or skill_dir.name)
        description = str(frontmatter.get("description") or "").strip()

        raw_category = str(frontmatter.get("category") or SkillCategory.default().value)
        if raw_category not in _VALID_CATEGORIES:
            logger.warning(
                f"[SkillLoader] {skill_dir.name} 的 category={raw_category!r} 不合法，"
                f"回退为 {SkillCategory.default().value!r}"
            )
            raw_category = SkillCategory.default().value
        category = SkillCategory(raw_category)

        activation = self._parse_activation(skill_dir, frontmatter.get("activation"))
        version = frontmatter.get("version")
        version = str(version) if version is not None else None
        tags = tuple(str(tag) for tag in (frontmatter.get("tags") or []))
        allowed_agents = tuple(str(agent) for agent in (frontmatter.get("allowed_agents") or []))
        required_tools = tuple(str(tool) for tool in (frontmatter.get("required_tools") or []))

        return SkillDefinition(
            name=name,
            description=description,
            category=category,
            skill_dir=skill_dir,
            activation=activation,
            version=version,
            tags=tags,
            allowed_agents=allowed_agents,
            required_tools=required_tools,
        )

    def _parse_activation(self, skill_dir: Path, raw_activation: object) -> SkillActivationMode:
        """解析 frontmatter 里可选的 `activation.mode`（重构文档 7.3 节）。

        Args:
            skill_dir: 技能所在目录，仅用于日志定位。
            raw_activation: frontmatter 里 `activation` 键的原始值，预期形如
                `{"mode": "automatic"}`；缺失、非字典或 `mode` 不合法时都
                回退为默认值，只记 warning，不中断加载。

        Returns:
            合法的 `SkillActivationMode`。
        """
        if raw_activation is None:
            return _DEFAULT_ACTIVATION_MODE
        if not isinstance(raw_activation, dict):
            logger.warning(f"[SkillLoader] {skill_dir.name} 的 activation 不是字典，已忽略: {raw_activation!r}")
            return _DEFAULT_ACTIVATION_MODE

        mode = str(raw_activation.get("mode") or "").strip()
        if not mode:
            return _DEFAULT_ACTIVATION_MODE
        if mode not in _VALID_ACTIVATION_MODES:
            logger.warning(
                f"[SkillLoader] {skill_dir.name} 的 activation.mode={mode!r} 不合法，"
                f"回退为 {_DEFAULT_ACTIVATION_MODE!r}"
            )
            return _DEFAULT_ACTIVATION_MODE
        return mode  # type: ignore[return-value]


def resolve_skill_dirs(skills_dirs_setting: str) -> list[Path]:
    """把形如 "skills/core,skills/public" 的配置字符串解析成项目根目录下的绝对路径列表。

    Args:
        skills_dirs_setting: 逗号分隔的相对路径字符串，对应配置项 `SKILLS_DIRS`。

    Returns:
        绝对路径列表。
    """
    return [
        _PROJECT_ROOT / part.strip()
        for part in skills_dirs_setting.split(",")
        if part.strip()
    ]
