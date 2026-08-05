"""技能元数据值对象。

原样迁移自 diit-agent-server 的 `src/core/skills/definition.py`：单个技能的
轻量元数据（仅来自 SKILL.md frontmatter），刻意不含正文/参考资料内容——
渐进式披露要求启动阶段只保留"目录信息"，重内容留给 SkillContentReader
在工具被调用时才去读取。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.constants import SkillCategory

_SCRIPT_ENTRY_CANDIDATES = ("main.py", "main.js")


@dataclass(frozen=True)
class RequiredSecret:
    """技能脚本声明需要的一个密钥（对应设计文档 5.4 节）。

    Attributes:
        name: 密钥名，与调用方请求里 `secrets` 字典的 key 对应，同时也是
            注入到脚本子进程时使用的环境变量名。
        optional: 调用方未提供该密钥时是否允许技能继续执行（不阻断，只是
            拿不到这个密钥）。当前"三重交集"注入模型下这个字段仅作声明性
            记录，不影响是否注入的判断逻辑。
    """

    name: str
    optional: bool = True


@dataclass(frozen=True)
class SkillDefinition:
    """单个技能的元数据。

    Attributes:
        name: 技能标识，如 search_knowledge_base。
        tool_name: 暴露给 LangChain 的工具名（缺省等于 name）。
        description: 工具描述，决定 LLM 何时选用该技能。
        category: 技能分类，决定挂载到哪个 Agent。
        skill_dir: 技能所在目录，如 skills/core/search-knowledge-base。
        parameters: 参数定义列表，元素形如
            `{"name": ..., "type": ..., "required": ..., "description": ...}`。
        runtime_context_keys: 需要从调用方运行时上下文注入的键名列表，
            如 workflow_id、user_id。
        required_secrets: 技能脚本声明需要的密钥列表，由调用方按请求提供
            具体值，绝不进 prompt/日志/checkpoint（见 5.4 节）。
    """

    name: str
    tool_name: str
    description: str
    category: SkillCategory
    skill_dir: Path
    parameters: list[dict] = field(default_factory=list)
    runtime_context_keys: list[str] = field(default_factory=list)
    required_secrets: list[RequiredSecret] = field(default_factory=list)

    @property
    def skill_md_path(self) -> Path:
        """SKILL.md 文件的绝对路径。"""
        return self.skill_dir / "SKILL.md"

    @property
    def scripts_dir(self) -> Path:
        """脚本目录路径（不保证存在）。"""
        return self.skill_dir / "scripts"

    @property
    def references_dir(self) -> Path:
        """参考资料目录路径（不保证存在）。"""
        return self.skill_dir / "references"

    @property
    def script_path(self) -> Path | None:
        """返回 scripts/ 下可执行的入口脚本路径（优先 main.py，其次 main.js）。

        Returns:
            找到的入口脚本绝对路径；不存在任何入口脚本时返回 None。
        """
        for candidate in _SCRIPT_ENTRY_CANDIDATES:
            path = self.scripts_dir / candidate
            if path.is_file():
                return path
        return None

    def has_script(self) -> bool:
        """该技能是否为脚本型技能（有可执行入口）。"""
        return self.script_path is not None

    @property
    def source(self) -> str:
        """技能来源：core（内置）/ public（社区技能），取自 skill_dir 的上一级目录名。"""
        return self.skill_dir.parent.name

    def __repr__(self) -> str:  # noqa: D105
        return f"SkillDefinition(tool_name={self.tool_name!r}, category={self.category!r})"
