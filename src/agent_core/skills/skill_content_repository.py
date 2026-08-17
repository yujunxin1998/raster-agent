"""渐进式披露中"重"的那一半：只有技能被真正激活时才会用到这个类。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 6、9 节：Skill 框架不再自动
执行脚本（旧的"脚本协议"已随 `query_database`/`search_knowledge_base` 迁移
成独立 Tool 一起废弃）——新语义见 `skill_resource_tool.py`：模型改为通过
`read_skill_resource` 按需读取脚本源码或参考资料，自行决定是否调用
`run_python`/`run_command` 执行，脚本只是"资源"，不是框架自动发现和执行的
固定入口。本类现在只做两件事：读正文、读参考资料索引。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.skills.skill_definition import SkillDefinition

_FRONTMATTER_BOUNDARY = "---"


class SkillContentRepository:
    """负责读取 SKILL.md 正文、参考资料。拼装成激活时的最终正文是调用方
    （`SkillActivationService.activate()`）的职责——它还需要额外做路径改写，
    本类只管"读"，不管"怎么拼"。
    """

    def __init__(self, skill: SkillDefinition) -> None:
        """初始化内容读取器。

        Args:
            skill: 目标技能的元数据。
        """
        self._skill = skill

    def read_instructions(self) -> str:
        """读取 SKILL.md，返回 frontmatter 之后的正文部分。

        Returns:
            SKILL.md 正文文本（已去除首尾空白）。
        """
        text = self._skill.skill_md_path.read_text(encoding="utf-8")
        parts = text.split(_FRONTMATTER_BOUNDARY, 2)
        # parts: ["", frontmatter, body] —— 取最后一段作为正文
        if len(parts) >= 3:
            return parts[2].strip()
        return text.strip()

    def read_references(self) -> list[str]:
        """遍历 references/ 目录，逐个文件读取内容。

        Returns:
            每个参考资料文件格式化为 "### 文件名\\n内容" 的字符串列表，
            references/ 目录不存在时返回空列表。
        """
        references: list[str] = []
        references_dir = self._skill.references_dir
        if not references_dir.is_dir():
            return references

        for reference_file in sorted(references_dir.iterdir()):
            if not reference_file.is_file():
                continue
            try:
                content = reference_file.read_text(encoding="utf-8")
                references.append(f"### {reference_file.name}\n{content}")
            except OSError as exc:
                logger.warning(
                    f"[SkillContentRepository] 无法读取参考资料文件 "
                    f"skill={self._skill.name} file={reference_file} error={exc}"
                )
        return references
