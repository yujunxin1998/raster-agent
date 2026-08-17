"""`read_skill_resource` 工具：按需读取技能目录下的具体资源文件。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 7.6 节、第 8 节"三级渐进式
披露"的 Level 3：`load_skill` 只返回指令正文 + 参考资料索引，参考资料的
具体内容、脚本源码、模板文件都要等模型明确需要时再单独用这个工具按路径读取，
不在激活时一次性全部拼进上下文。

路径安全：拒绝绝对路径和 `..` 越界，用 `Path.resolve()` 之后校验结果路径
仍落在技能目录内——这一步天然也拦住了符号链接逃逸（`resolve()` 会先解引用
符号链接，再判断解引用后的真实路径是否越界，不需要额外的 `is_symlink()`
遍历）。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agent_core.skills.skill_registry import SkillRegistry
from src.common.exceptions import SkillNotFoundError

READ_SKILL_RESOURCE_TOOL_NAME = "read_skill_resource"
_TOOL_DESCRIPTION = (
    "按路径读取一个已加载技能的具体资源文件（参考资料、脚本、模板），"
    "path 取自 load_skill 返回的资源索引。"
)

#: 文本资源大小上限，超过则拒绝读取（避免一次性把大文件塞进模型上下文）。
_MAX_RESOURCE_BYTES = 200_000
#: 直接判定为二进制、只返回元数据不读内容的扩展名。
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".bin", ".pyc",
    ".db", ".sqlite", ".sqlite3",
})


class ReadSkillResourceInput(BaseModel):
    skill_name: str = Field(description="资源所属的技能名")
    resource_path: str = Field(description="技能目录内的相对路径，如 references/spec.md")


def _read_resource(skill_dir: Path, resource_path: str) -> str:
    """在 `skill_dir` 范围内安全读取一个相对路径，返回内容或拒绝/错误说明文本。

    Args:
        skill_dir: 技能所在目录（沙箱边界）。
        resource_path: 调用方传入的相对路径，未经信任。

    Returns:
        文件内容，或者一段说明"为什么没有读到内容"的文本——不抛异常，
        保证不中断 Agent 推理链，与本项目其余工具的失败处理风格一致。
    """
    raw = Path(resource_path)
    if raw.is_absolute() or ".." in raw.parts:
        return f"拒绝：resource_path 不允许绝对路径或包含 '..'（收到 {resource_path!r}）"

    skill_root = skill_dir.resolve()
    candidate = (skill_dir / raw).resolve()
    if candidate != skill_root and skill_root not in candidate.parents:
        return f"拒绝：resource_path 越出技能目录（收到 {resource_path!r}）"

    if not candidate.is_file():
        return f"资源不存在: {resource_path}"

    size = candidate.stat().st_size
    if candidate.suffix.lower() in _BINARY_EXTENSIONS:
        return f"[二进制文件，仅返回元数据] path={resource_path} size={size}B"
    if size > _MAX_RESOURCE_BYTES:
        return f"[文件过大 ({size}B > {_MAX_RESOURCE_BYTES}B 上限)，未读取内容] path={resource_path}"

    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[无法按 UTF-8 解码，可能是二进制文件] path={resource_path} size={size}B"


def create_read_skill_resource_tool(
    registry: SkillRegistry,
    allowed_categories: frozenset[str],
) -> StructuredTool:
    """构造一个绑定了具体技能可见范围的 `read_skill_resource` 工具实例。

    Args:
        registry: 技能注册表，通常传入 `get_skill_manager().registry`。
        allowed_categories: 调用方允许访问的技能分类集合，与 `load_skill`
            用同一个范围校验，越权访问一律回落成"未找到"。

    Returns:
        绑定好范围的 `StructuredTool` 实例。
    """

    async def _invoke(skill_name: str, resource_path: str) -> str:
        try:
            skill = registry.get(skill_name)
        except SkillNotFoundError as exc:
            return str(exc)
        if skill.category not in allowed_categories:
            return f"技能 '{skill_name}' 未找到。已注册：{registry.names}"

        return _read_resource(skill.skill_dir, resource_path)

    return StructuredTool(
        name=READ_SKILL_RESOURCE_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema=ReadSkillResourceInput,
        coroutine=_invoke,
    )
