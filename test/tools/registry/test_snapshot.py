"""RegistrySnapshot.replace_source 单元测试：冲突检测 + 整体替换语义。"""
from __future__ import annotations

import pytest

from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import ToolDefinition


def _definition(*, canonical_name: str, model_name: str, source_id: str) -> ToolDefinition:
    return ToolDefinition(
        canonical_name=canonical_name,
        model_name=model_name,
        description="",
        source_type="builtin" if source_id == "builtin" else "skill",
        source_id=source_id,
        scope="application",
        permissions_key=model_name,
        build_tool=(lambda: object()),
    )


def test_replace_source_adds_new_definitions() -> None:
    snapshot = RegistrySnapshot.empty()
    definition = _definition(canonical_name="builtin.foo", model_name="foo", source_id="builtin")

    new_snapshot = snapshot.replace_source("builtin", [definition])

    assert new_snapshot.revision == snapshot.revision + 1
    assert new_snapshot.by_canonical_name["builtin.foo"] is definition
    assert new_snapshot.by_model_name["foo"] is definition
    assert new_snapshot.by_source["builtin"] == ("builtin.foo",)


def test_replace_source_removes_stale_definitions_when_source_shrinks() -> None:
    snapshot = RegistrySnapshot.empty()
    first = _definition(canonical_name="skill.a", model_name="a", source_id="skill")
    second = _definition(canonical_name="skill.b", model_name="b", source_id="skill")
    snapshot = snapshot.replace_source("skill", [first, second])

    # 第二次发布只带 "a"：模拟一个 Skill 被删除，新快照里 "b" 必须完全消失，
    # 不是残留（设计文档 5.2 节 replace_source 语义）。
    shrunk = snapshot.replace_source("skill", [first])

    assert "skill.b" not in shrunk.by_canonical_name
    assert "b" not in shrunk.by_model_name
    assert shrunk.by_source["skill"] == ("skill.a",)


def test_replace_source_with_empty_list_evicts_entire_source() -> None:
    snapshot = RegistrySnapshot.empty()
    definition = _definition(canonical_name="mcp.github.create_issue", model_name="create_issue", source_id="mcp:github")
    snapshot = snapshot.replace_source("mcp:github", [definition])

    evicted = snapshot.replace_source("mcp:github", [])

    assert "mcp:github" not in evicted.by_source
    assert "create_issue" not in evicted.by_model_name


def test_replace_source_rejects_cross_source_model_name_conflict() -> None:
    """跨来源同名冲突必须整体拒绝发布，不能静默覆盖（设计文档 4.3/5.3 节）。"""
    snapshot = RegistrySnapshot.empty()
    builtin_tool = _definition(canonical_name="builtin.search", model_name="search", source_id="builtin")
    snapshot = snapshot.replace_source("builtin", [builtin_tool])

    conflicting_skill_tool = _definition(canonical_name="skill.rag.search", model_name="search", source_id="skill")

    with pytest.raises(ValueError, match="冲突"):
        snapshot.replace_source("skill", [conflicting_skill_tool])

    # 拒绝发生在构造新快照期间，原快照必须保持不变（本身是 frozen，这里
    # 断言语义没有被绕过篡改）。
    assert snapshot.by_model_name["search"] is builtin_tool


def test_replace_source_rejects_duplicate_canonical_name_within_same_source() -> None:
    snapshot = RegistrySnapshot.empty()
    dup_a = _definition(canonical_name="skill.rag.dup", model_name="dup_a", source_id="skill")
    dup_b = _definition(canonical_name="skill.rag.dup", model_name="dup_b", source_id="skill")

    with pytest.raises(ValueError, match="重复"):
        snapshot.replace_source("skill", [dup_a, dup_b])


def test_replace_source_same_source_can_update_its_own_model_name() -> None:
    """同一来源自己更新（比如 Skill 热重载后 tool_name 不变但内容变了）不应该
    被误判为"跨来源冲突"。"""
    snapshot = RegistrySnapshot.empty()
    old = _definition(canonical_name="skill.rag.search", model_name="search", source_id="skill")
    snapshot = snapshot.replace_source("skill", [old])

    updated = _definition(canonical_name="skill.rag.search", model_name="search", source_id="skill")
    new_snapshot = snapshot.replace_source("skill", [updated])

    assert new_snapshot.by_model_name["search"] is updated
