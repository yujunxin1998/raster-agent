"""技能元数据值对象。

原样迁移自 diit-agent-server 的 `src/core/skills/definition.py`：单个技能的
轻量元数据（仅来自 SKILL.md frontmatter），刻意不含正文/参考资料内容——
渐进式披露要求启动阶段只保留"目录信息"，重内容留给 `SkillContentRepository`
在工具被调用时才去读取。

`docs/Skill与Tool完全解耦重构设计.md` 第 6、15 节收敛后的最终形态：不再有
`SkillKind`/`parameters`/`runtime_context_keys`/`required_secrets`/
`script_path`/`has_script()` 这套"Skill 也能是参数化 Tool"的旧概念——
`query_database`/`search_knowledge_base` 已经迁移成独立业务 Tool
（`database_query_tool.py`/`knowledge_search_tool.py`），Skill 现在统一是
"按需加载的指令与资源包"，用 `activation` 字段表达激活策略，`required_tools`
纯声明式地指向它指导模型使用的 Tool（不代表拥有那些 Tool 的执行权限）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.common.constants import SkillCategory

#: Skill 的激活策略（对应重构文档 7.3 节）。automatic：可被 SkillRouter 自动
#: 激活，也可被模型通过 load_skill 补充激活；explicit_only：只有用户/API 或
#: 上层 Agent 显式指定时才能激活；required：绑定到指定 Agent，在该 Agent 中
#: 始终强制激活。
SkillActivationMode = Literal["automatic", "explicit_only", "required"]
_VALID_ACTIVATION_MODES: frozenset[str] = frozenset({"automatic", "explicit_only", "required"})
_DEFAULT_ACTIVATION_MODE: SkillActivationMode = "automatic"


@dataclass(frozen=True)
class SkillDefinition:
    """单个技能的元数据。

    Attributes:
        name: 技能标识，也是 `SkillRegistry` 的索引键，如
            `knowledge-base-answering`。
        description: 技能描述，决定 LLM 何时选用该技能，出现在
            `<skill_catalog>` 目录里。
        category: 技能分类，决定挂载到哪个 Agent。
        skill_dir: 技能所在目录，如 skills/core/knowledge-base-answering。
        activation: 激活策略（`automatic`/`explicit_only`/`required`），
            `SkillRouter` 据此决定预路由结果，见模块顶部 `SkillActivationMode`。
        version: 技能版本号，缺省 None；`SkillMiddleware` 用它 + `name`
            判断"同一版本是否已激活过"（重构文档 7.4 节），暂无版本管理时
            可以一直缺省。
        tags: 供未来路由（Embedding/LLM 重排）使用的标签集合，当前规则路由
            不消费这个字段，只做透传保留。
        allowed_agents: 限定这个技能只对哪些 Agent 可见，为空表示不做限制
            （沿用调用方原有的 `allowed_categories` 范围校验）。
        required_tools: 该技能指导模型使用的 Tool 名称列表，纯声明性质，
            不代表拥有或自动获得这些 Tool 的执行权限（权限仍由
            `GuardrailMiddleware`/`ToolResolver` 独立把关）。
    """

    name: str
    description: str
    category: SkillCategory
    skill_dir: Path
    activation: SkillActivationMode = _DEFAULT_ACTIVATION_MODE
    version: str | None = None
    tags: tuple[str, ...] = ()
    allowed_agents: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()

    @property
    def tool_name(self) -> str:
        """**已废弃**只读别名，等同于 `name`。

        仅为尚未完成迁移的外部调用方（如 `src/api/router/skill_router.py`
        的历史响应字段）保留读兼容，新代码一律直接用 `name`（重构文档
        第 6 节"如前端 API 暂时依赖 tool_name，可提供带弃用告警的只读
        别名"）。
        """
        return self.name

    @property
    def skill_md_path(self) -> Path:
        """SKILL.md 文件的绝对路径。"""
        return self.skill_dir / "SKILL.md"

    @property
    def references_dir(self) -> Path:
        """参考资料目录路径（不保证存在）。"""
        return self.skill_dir / "references"

    @property
    def source(self) -> str:
        """技能来源：core（内置）/ public（社区技能），取自 skill_dir 的上一级目录名。"""
        return self.skill_dir.parent.name

    def __repr__(self) -> str:  # noqa: D105
        return f"SkillDefinition(name={self.name!r}, category={self.category!r})"
