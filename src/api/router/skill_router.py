"""技能管理 REST API。

路径、参数、响应结构与原项目 `src/api/router/skills.py` 完全一致，确保
`diit-agent-web` 的 `skillsApi.js` 无需任何改动即可对接本工程。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.agent_core.skills import (
    SkillDefinition,
    SkillFileTreeReader,
    get_skill_manager,
)
from src.common.exceptions import PathTraversalError, SkillNotFoundError
from src.common.response import ApiResponse, success
from src.schema.skill_schema import (
    SkillDetail,
    SkillFileContent,
    SkillInfo,
    SkillToggleRequest,
    SkillToggleResponse,
    SkillTreeNode,
)
from src.storage.skill_settings_store import get_skill_settings_store

router = APIRouter(prefix="/skills")


def _get_skill_or_404(tool_name: str) -> SkillDefinition:
    """按 tool_name 查询技能，未找到时转换为 HTTP 404。"""
    try:
        return get_skill_manager().registry.get(tool_name)
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"技能 '{tool_name}' 不存在") from exc


def _to_skill_info(skill: SkillDefinition, enabled: bool) -> SkillInfo:
    """把 SkillDefinition 转换为对外展示的 SkillInfo。"""
    return SkillInfo(
        tool_name=skill.tool_name,
        name=skill.name,
        description=skill.description,
        category=skill.category.value,
        parameters=skill.parameters,
        has_script=skill.has_script(),
        enabled=enabled,
    )


@router.get("/", response_model=ApiResponse[list[SkillInfo]], summary="列出全部技能（含该用户的启用状态）")
async def list_skills(user_id: str) -> ApiResponse:
    """列出全部已注册技能，并标注该用户对每个技能的启用状态。"""
    disabled = await get_skill_settings_store().get_disabled_tools(user_id)
    skills = [
        _to_skill_info(skill, enabled=skill.tool_name not in disabled)
        for skill in get_skill_manager().registry.all
    ]
    return success(skills)


@router.get("/{tool_name}", response_model=ApiResponse[SkillDetail], summary="获取单个技能详情（含 SKILL.md 正文）")
async def get_skill_detail(tool_name: str, user_id: str) -> ApiResponse:
    """获取单个技能详情，正文只在这一步才读取（延续渐进式披露原则）。"""
    skill = _get_skill_or_404(tool_name)
    disabled = await get_skill_settings_store().is_disabled(user_id, tool_name)

    instructions = _read_instructions(skill) if skill.skill_md_path.exists() else ""
    detail = SkillDetail(
        **_to_skill_info(skill, enabled=not disabled).model_dump(),
        instructions=instructions,
        source=skill.source,
    )
    return success(detail)


def _read_instructions(skill: SkillDefinition) -> str:
    """读取技能 SKILL.md 正文，抽成独立函数便于测试时打桩。"""
    from src.agent_core.skills.skill_content_reader import SkillContentReader

    return SkillContentReader(skill).read_instructions()


@router.get(
    "/{tool_name}/tree",
    response_model=ApiResponse[list[SkillTreeNode]],
    summary="列出技能目录下某一层的文件/子目录（懒加载，不递归）",
)
async def get_skill_tree(tool_name: str, path: str = "") -> ApiResponse:
    """列出技能目录树中某一层的子节点，供前端懒加载展开。"""
    skill = _get_skill_or_404(tool_name)
    try:
        children = SkillFileTreeReader(skill).list_children(path)
    except (ValueError, PathTraversalError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return success([SkillTreeNode(**child) for child in children])


@router.get("/{tool_name}/file", response_model=ApiResponse[SkillFileContent], summary="读取技能目录下某个文件的原文")
async def get_skill_file(tool_name: str, path: str) -> ApiResponse:
    """读取技能目录下某个文件的原文内容。"""
    skill = _get_skill_or_404(tool_name)
    try:
        result = SkillFileTreeReader(skill).read_file(path)
    except (ValueError, PathTraversalError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return success(SkillFileContent(path=path, **result))


@router.put("/{tool_name}/toggle", response_model=ApiResponse[SkillToggleResponse], summary="启用/禁用某个技能（用户级）")
async def toggle_skill(tool_name: str, body: SkillToggleRequest) -> ApiResponse:
    """切换某个技能对指定用户的启用状态。"""
    # 提前校验技能存在，避免为不存在的 tool_name 写入开关记录
    _get_skill_or_404(tool_name)

    await get_skill_settings_store().set_enabled(body.user_id, tool_name, body.enabled)
    return success(SkillToggleResponse(tool_name=tool_name, enabled=body.enabled))
