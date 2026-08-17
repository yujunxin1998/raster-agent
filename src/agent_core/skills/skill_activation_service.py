"""把一个技能名解析为可以直接喂给模型的指令文本。

**迁移状态**（对应 `docs/Skill与Tool完全解耦重构设计.md` 第 15 节阶段 1）：
原来只处理 `SkillKind.WORKFLOW` 技能、显式拒绝 `TOOL` 形态技能
（`search_knowledge_base`/`query_database`）的限制已经取消——`SkillKind` 是
待删除的过时区分（见 `skill_definition.py` 模块文档"迁移状态"），激活逻辑
现在对任意技能一视同仁，都是"读正文 + 参考资料 + 路径替换"。这两个技能在
未完成 Tool 化迁移（重构文档第 15 节阶段 3）之前仍然*同时*可以被
`load_skill` 读到指令文本、也可以被直接当工具调用——不冲突，只是过渡期内
的一个技能有两个入口。
"""
from __future__ import annotations

from src.agent_core.skills.skill_content_repository import SkillContentRepository
from src.agent_core.skills.skill_path_rewriter import SkillPathRewriter
from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.exceptions import SkillNotFoundError

_REFERENCES_SECTION_TITLE = "## 参考资料"


class SkillActivationService:
    """按名称激活一个技能，返回已完成路径替换的正文文本。"""

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
        """
        skill = self._registry.get(skill_name)

        if skill.category not in allowed_categories:
            raise SkillNotFoundError(f"技能 '{skill_name}' 未找到。已注册：{self._registry.names}")

        reader = SkillContentRepository(skill)
        body = reader.read_instructions()
        references = reader.read_references()
        if references:
            body += f"\n\n{_REFERENCES_SECTION_TITLE}\n" + "\n\n".join(references)

        return self._path_rewriter.rewrite(skill, body)
