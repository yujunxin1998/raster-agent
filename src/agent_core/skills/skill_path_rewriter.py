"""把 SKILL.md 正文里 Anthropic 公开技能库的路径约定，替换成本项目沙箱的真实约定。

背景（对应 `docs/Skill注入与Load-Skill重构设计.md` 第六节）：`skills/public/` 下多个
SKILL.md 正文原样照搬了 Anthropic 公开 Agent Skills 库的路径写法
（`/mnt/skills/<source>/<name>/...`、`/mnt/user-data/{uploads,outputs,workspace}/...`），
但本项目的 `Sandbox`（`LocalSandbox.execute_command`）从未挂载过 `/mnt`——命令执行的
cwd 固定在该会话的 `ThreadWorkspace.workspace_dir`，这些路径在本项目里从未真正可达。
这里只做字符串替换，不解析 Markdown/代码块。
"""
from __future__ import annotations

from src.agent_core.skills.skill_definition import SkillDefinition

_UPLOADS_TOKEN = "/mnt/user-data/uploads"
_OUTPUTS_TOKEN = "/mnt/user-data/outputs"
_WORKSPACE_TOKEN = "/mnt/user-data/workspace"


class SkillPathRewriter:
    """把技能正文里的路径 token 替换为本项目沙箱下的真实路径。"""

    def rewrite(self, skill: SkillDefinition, text: str) -> str:
        """对一个技能的已装配正文做路径替换。

        Args:
            skill: 目标技能定义，用其 `source`/`skill_dir` 精确匹配"这个技能自己的"
                脚本目录前缀，不做全局正则替换，避免误改到正文里恰好提到的
                别的技能名字符串。
            text: 待替换的正文文本（通常是 `SkillContentReader.read_instructions()` +
                `read_references()` 拼接后的结果）。

        Returns:
            替换后的正文文本。
        """
        skill_root_token = f"/mnt/skills/{skill.source}/{skill.skill_dir.name}"
        text = text.replace(skill_root_token, str(skill.skill_dir))
        # 会话上传/产物目录：沙箱工具执行 cwd 固定在 workspace/，
        # uploads/outputs 是 ThreadWorkspace 下的同级目录，用相对路径可达。
        text = text.replace(_UPLOADS_TOKEN, "../uploads")
        text = text.replace(_OUTPUTS_TOKEN, "../outputs")
        text = text.replace(_WORKSPACE_TOKEN, ".")
        return text
