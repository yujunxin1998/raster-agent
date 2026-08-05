"""技能管理 REST API 的请求/响应模型。

字段与原项目 `src/schema/skills.py` 保持完全一致，确保
`diit-agent-web/src/api/skillsApi.js` 无需任何改动即可对接。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SkillParameterInfo(BaseModel):
    """技能参数的展示信息。"""

    name: str = Field(..., description="参数名")
    type: str = Field("string", description="参数类型：string/integer/number/boolean/array/object")
    required: bool = Field(False, description="是否必填")
    description: str = Field("", description="参数说明")


class SkillInfo(BaseModel):
    """技能的基础展示信息（列表接口使用）。"""

    tool_name: str = Field(..., description="技能唯一标识，对应 LangChain 工具名")
    name: str = Field(..., description="技能名称（来自 SKILL.md frontmatter 的 name 字段）")
    description: str = Field(..., description="技能描述，说明何时会被调用")
    category: str = Field(..., description="分类：general / rag / web_search / database / tool")
    parameters: list[SkillParameterInfo] = Field(default_factory=list, description="参数列表")
    has_script: bool = Field(..., description="是否带可执行脚本（false 表示纯指导型技能）")
    enabled: bool = Field(..., description="该用户是否已启用此技能")


class SkillDetail(SkillInfo):
    """技能详情（详情接口使用，含正文）。"""

    instructions: str = Field("", description="SKILL.md 正文（技能指令），仅在查看详情时才读取")
    source: str = Field(..., description="来源：core（内置）/ public（社区技能）")


class SkillToggleRequest(BaseModel):
    """切换技能启用状态的请求体。"""

    user_id: str = Field(..., description="用户ID")
    enabled: bool = Field(..., description="true=启用，false=禁用")


class SkillToggleResponse(BaseModel):
    """切换技能启用状态的响应体。"""

    tool_name: str = Field(..., description="技能唯一标识")
    enabled: bool = Field(..., description="切换后的启用状态")


class SkillTreeNode(BaseModel):
    """技能目录树中的一个节点（文件或目录）。"""

    name: str = Field(..., description="文件/目录名")
    path: str = Field(..., description="相对技能根目录的路径，用于下一次请求子节点或文件内容")
    type: str = Field(..., description="dir 或 file")


class SkillFileContent(BaseModel):
    """技能目录下某个文件的原文内容。"""

    path: str = Field(..., description="相对技能根目录的路径")
    content: str = Field(..., description="文件原文（二进制文件会降级为提示文本）")
    truncated: bool = Field(False, description="是否因文件过大被截断（超过 200KB）")
