"""扫描提示词模板目录，构建 PromptRegistry。

原样迁移自 `src/core/prompts/loader.py::load_all_prompts()`，改造为
`PromptLoader` 类（原实现是模块级函数），风格与 `SkillLoader` 保持一致。
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agent_core.prompts.prompt_registry import PromptRegistry
from src.agent_core.prompts.prompt_template import PromptTemplate

_DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PromptLoader:
    """扫描模板目录下全部 `*.md` 文件，产出 PromptRegistry。"""

    def __init__(self, templates_dir: Path | None = None) -> None:
        """初始化加载器。

        Args:
            templates_dir: 待扫描的模板目录，默认使用本模块内置的
                `templates/` 目录（迁移自原项目的全部提示词）。
        """
        self._templates_dir = templates_dir or _DEFAULT_TEMPLATES_DIR

    def load(self) -> PromptRegistry:
        """执行一次全量扫描，返回构建好的 PromptRegistry。

        约定：MD 文件名（去除扩展名，转大写）作为 PromptTemplate.name，
        文件全文（strip 后）作为 PromptTemplate.content。单个文件解析失败
        不会中断整体加载，只记 error 日志并跳过。

        Returns:
            已注册全部合法模板的 PromptRegistry；目录不存在或为空时返回
            空注册表并记 warning 日志。
        """
        registry = PromptRegistry()

        if not self._templates_dir.exists():
            logger.warning(f"[PromptLoader] 模板目录不存在: {self._templates_dir}")
            return registry

        md_files = sorted(self._templates_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"[PromptLoader] {self._templates_dir} 下未找到任何 .md 文件")
            return registry

        for md_file in md_files:
            name = md_file.stem.upper()
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                source = str(md_file.relative_to(_PROJECT_ROOT))
                registry.register(PromptTemplate(name=name, content=content, source=source))
                logger.debug(f"[PromptLoader] 已加载: {name} <- {md_file.name}")
            except Exception as exc:
                logger.error(f"[PromptLoader] 加载失败: {md_file.name} - {exc}")

        logger.info(f"[PromptLoader] 注册完成，共 {len(registry)} 条: {registry.names}")
        return registry
