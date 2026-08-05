"""扫描技能目录、解析 frontmatter，构建 SkillRegistry。

原样迁移自 `src/core/skills/loader.py`：启动阶段只解析 frontmatter，
不读取 SKILL.md 正文——渐进式披露的"轻"那一半。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from loguru import logger

from src.agent_core.skills.skill_definition import RequiredSecret, SkillDefinition
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.constants import SkillCategory
from src.common.exceptions import SkillDefinitionInvalidError

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VALID_CATEGORIES = {category.value for category in SkillCategory}
_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


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
                    logger.debug(f"[SkillLoader] 已加载: {skill.tool_name} <- {skill_md}")
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
        name = str(frontmatter.get("name") or skill_dir.name)
        tool_name = str(frontmatter.get("tool_name") or name)
        description = str(frontmatter.get("description") or "").strip()

        raw_category = str(frontmatter.get("category") or SkillCategory.default().value)
        if raw_category not in _VALID_CATEGORIES:
            logger.warning(
                f"[SkillLoader] {skill_dir.name} 的 category={raw_category!r} 不合法，"
                f"回退为 {SkillCategory.default().value!r}"
            )
            raw_category = SkillCategory.default().value
        category = SkillCategory(raw_category)

        parameters = frontmatter.get("parameters") or []
        runtime_context_keys = frontmatter.get("runtime_context_keys") or []
        required_secrets = self._parse_required_secrets(skill_dir, frontmatter.get("required_secrets") or [])

        return SkillDefinition(
            name=name,
            tool_name=tool_name,
            description=description,
            category=category,
            skill_dir=skill_dir,
            parameters=parameters,
            runtime_context_keys=runtime_context_keys,
            required_secrets=required_secrets,
        )

    def _parse_required_secrets(self, skill_dir: Path, raw_entries: list) -> list[RequiredSecret]:
        """解析 frontmatter 里可选的 `required_secrets` 列表。

        单个条目格式不合法（不是字典或缺少 name）不会中断整体加载，只记
        warning 并跳过该条目——与本类其余解析逻辑一致的容错风格。

        Args:
            skill_dir: 技能所在目录，仅用于日志定位。
            raw_entries: frontmatter 里 `required_secrets` 的原始值。

        Returns:
            解析后的 RequiredSecret 列表。
        """
        secrets: list[RequiredSecret] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                logger.warning(f"[SkillLoader] {skill_dir.name} 的 required_secrets 条目不是字典，已跳过: {entry!r}")
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                logger.warning(f"[SkillLoader] {skill_dir.name} 的 required_secrets 条目缺少 name，已跳过: {entry!r}")
                continue
            secrets.append(RequiredSecret(name=name, optional=bool(entry.get("optional", True))))
        return secrets


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
