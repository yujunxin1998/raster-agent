"""把一个 `WORKFLOW` 形态技能名解析为可以直接喂给模型的指令文本。

对应 `docs/Skill注入与Load-Skill重构设计.md` 第四节：只处理 `SkillKind.WORKFLOW`
技能——`TOOL` 形态技能（`search_knowledge_base`/`query_database`）继续走
`SkillToolFactory`，不经过这里，`activate()` 会主动拒绝按 `TOOL` 技能名调用。
"""
from __future__ import annotations

from src.agent_core.skills.skill_content_reader import SkillContentReader
from src.agent_core.skills.skill_definition import SkillKind
from src.agent_core.skills.skill_path_rewriter import SkillPathRewriter
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.exceptions import SkillDefinitionInvalidError, SkillNotFoundError

_REFERENCES_SECTION_TITLE = "## 参考资料"


class SkillActivationService:
    """按名称激活一个 WORKFLOW 技能，返回已完成路径替换的正文文本。"""

    def __init__(self, registry: SkillRegistry, path_rewriter: SkillPathRewriter | None = None) -> None:
        """初始化激活服务。

        Args:
            registry: 技能注册表，通常传入 `get_skill_manager().registry`。
            path_rewriter: 路径改写器，默认新建一个（无状态，可以随意共享/新建）。
        """
        self._registry = registry
        self._path_rewriter = path_rewriter or SkillPathRewriter()

    async def activate(self, skill_name: str, allowed_categories: frozenset[str]) -> str:
        """激活一个技能，返回可直接注入模型上下文的正文文本。

        Args:
            skill_name: 待加载的技能 `tool_name`。
            allowed_categories: 调用方（Lead Agent / 某个 subagent）允许加载的
                技能分类集合，用于确认这次调用没有越出调用方自己的技能范围。

        Returns:
            指令正文 + 参考资料（若有）+ 路径替换后的文本。

        Raises:
            SkillNotFoundError: 技能不存在，或存在但不在 `allowed_categories`
                范围内（两者用同一措辞，不额外泄露"存在但你无权用"这一区分）。
            SkillDefinitionInvalidError: 该技能是 `TOOL` 形态，不应该经
                `load_skill` 加载——应该直接调用同名工具。
        """
        skill = self._registry.get(skill_name)

        if skill.category not in allowed_categories:
            raise SkillNotFoundError(f"技能 '{skill_name}' 未找到。已注册：{self._registry.names}")

        if skill.kind is not SkillKind.WORKFLOW:
            raise SkillDefinitionInvalidError(
                f"技能 '{skill_name}' 是参数化工具技能，请直接调用同名工具而不是 load_skill"
            )

        reader = SkillContentReader(skill)
        body = reader.read_instructions()
        references = reader.read_references()
        if references:
            body += f"\n\n{_REFERENCES_SECTION_TITLE}\n" + "\n\n".join(references)

        return self._path_rewriter.rewrite(skill, body)
