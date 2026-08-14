"""ToolRegistry 单元/并发测试（设计文档第八节）。"""
from __future__ import annotations

import asyncio

import pytest

from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.agent_core.tools.registry.tool_registry import ToolRegistry


def _definition(*, model_name: str, source_id: str) -> ToolDefinition:
    return ToolDefinition(
        canonical_name=f"{source_id}.{model_name}",
        model_name=model_name,
        description="",
        source_type="builtin" if source_id == "builtin" else "skill",
        source_id=source_id,
        scope="application",
        permissions_key=model_name,
        build_tool=(lambda: object()),
    )


async def test_publish_updates_current_snapshot() -> None:
    registry = ToolRegistry()
    definition = _definition(model_name="foo", source_id="builtin")

    snapshot = await registry.publish("builtin", [definition])

    assert registry.current_snapshot() is snapshot
    assert snapshot.revision == 1


async def test_publish_failure_keeps_old_snapshot_unchanged() -> None:
    """`publish()` 校验失败时 `current_snapshot()` 保持不变（设计文档第八节）。"""
    registry = ToolRegistry()
    await registry.publish("builtin", [_definition(model_name="search", source_id="builtin")])
    before = registry.current_snapshot()

    conflicting = _definition(model_name="search", source_id="skill")
    with pytest.raises(ValueError):
        await registry.publish("skill", [conflicting])

    assert registry.current_snapshot() is before
    assert registry.current_snapshot().revision == before.revision


async def test_concurrent_publish_is_serialized_and_revision_monotonic() -> None:
    """两次并发 `publish()` 串行执行，最终 revision 正确递增、不丢更新。"""
    registry = ToolRegistry()

    async def publish_one(i: int) -> None:
        await registry.publish(f"source-{i}", [_definition(model_name=f"tool-{i}", source_id=f"source-{i}")])

    await asyncio.gather(*(publish_one(i) for i in range(10)))

    snapshot = registry.current_snapshot()
    assert snapshot.revision == 10
    assert len(snapshot.by_canonical_name) == 10
    for i in range(10):
        assert f"tool-{i}" in snapshot.by_model_name


async def test_concurrent_reads_during_publish_never_see_partial_state() -> None:
    """大量并发 `current_snapshot()` 读取期间发生 `publish()`，
    每次读取要么拿到完整旧快照要么拿到完整新快照，不会有半写状态。"""
    registry = ToolRegistry()
    await registry.publish("builtin", [_definition(model_name="a", source_id="builtin")])

    observed_sizes: set[int] = set()

    async def reader() -> None:
        for _ in range(200):
            observed_sizes.add(len(registry.current_snapshot().by_canonical_name))
            await asyncio.sleep(0)

    async def writer() -> None:
        await registry.publish("skill", [_definition(model_name="b", source_id="skill")])

    await asyncio.gather(reader(), reader(), writer())

    # 只应该观察到"发布前"(1个)或"发布后"(2个)两种完整尺寸，不应该出现
    # 中间态（比如 by_canonical_name 只写了一半）。
    assert observed_sizes <= {1, 2}


async def test_rollback_restores_previous_revision() -> None:
    registry = ToolRegistry()
    await registry.publish("builtin", [_definition(model_name="a", source_id="builtin")])
    target_revision = registry.current_snapshot().revision
    await registry.publish("skill", [_definition(model_name="b", source_id="skill")])
    assert registry.current_snapshot().revision != target_revision

    restored = await registry.rollback(target_revision)

    assert restored.revision == target_revision
    assert registry.current_snapshot().revision == target_revision
    assert "b" not in registry.current_snapshot().by_model_name


async def test_rollback_to_unknown_revision_raises() -> None:
    registry = ToolRegistry()
    await registry.publish("builtin", [_definition(model_name="a", source_id="builtin")])

    with pytest.raises(ValueError, match="不在可回滚窗口内"):
        await registry.rollback(999)
