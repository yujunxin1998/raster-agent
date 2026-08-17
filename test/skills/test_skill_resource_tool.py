"""`read_skill_resource` 工具的路径安全与读取行为单元测试。"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_resource_tool import create_read_skill_resource_tool


def _registry_with_skill(tmp_path: Path, *, name: str = "demo") -> tuple[Path, object]:
    public_root = tmp_path / "public"
    public_root.mkdir()
    skill_dir = public_root / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test\ncategory: general\n---\n\n正文\n",
        encoding="utf-8",
    )
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    (references_dir / "spec.md").write_text("参考内容", encoding="utf-8")
    (skill_dir / "large.txt").write_text("x" * 300_000, encoding="utf-8")
    (skill_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    registry = SkillLoader([public_root]).load()
    return skill_dir, registry


async def test_reads_file_within_skill_directory(tmp_path: Path) -> None:
    skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path="references/spec.md")

    assert result == "参考内容"


async def test_rejects_absolute_path(tmp_path: Path) -> None:
    skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path=str(skill_dir / "references" / "spec.md"))

    assert "拒绝" in result


async def test_rejects_parent_directory_traversal(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path="../../etc/passwd")

    assert "拒绝" in result


async def test_missing_resource_returns_not_found_text(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path="references/missing.md")

    assert "不存在" in result


async def test_oversized_file_is_rejected_without_reading_content(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path="large.txt")

    assert "过大" in result
    assert "x" * 100 not in result


async def test_binary_file_returns_metadata_only(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="demo", resource_path="image.png")

    assert "二进制文件" in result
    assert "size=" in result


async def test_out_of_category_scope_is_treated_as_not_found(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"web_search"}))

    result = await tool.coroutine(skill_name="demo", resource_path="references/spec.md")

    assert "未找到" in result


async def test_unknown_skill_name_returns_not_found(tmp_path: Path) -> None:
    _skill_dir, registry = _registry_with_skill(tmp_path)
    tool = create_read_skill_resource_tool(registry, allowed_categories=frozenset({"general"}))

    result = await tool.coroutine(skill_name="does-not-exist", resource_path="references/spec.md")

    assert "未找到" in result
