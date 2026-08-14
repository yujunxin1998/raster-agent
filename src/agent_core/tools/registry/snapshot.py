"""RegistrySnapshot：进程内不可变的工具登记快照（设计文档 4.1 节）。

核心是 copy-on-write：`replace_source` 从不修改已发布的快照，只基于它
构建一份新的、revision 递增的快照；`ToolRegistry.publish()` 校验通过后
才把引用整体切换过去。读方（`current_snapshot()`）永远拿到一份完整、
不会中途被修改的快照，不需要为"读"加锁。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from src.agent_core.tools.registry.tool_definition import ToolDefinition


@dataclass(frozen=True)
class RegistrySnapshot:
    """某一时刻已发布的全部 application 级工具登记表。

    Attributes:
        revision: 单调递增的版本号，每次 `replace_source` 成功都 +1。
        created_at: 本快照的生成时间（UTC）。
        by_canonical_name: 全部条目，按 `canonical_name` 索引（唯一）。
        by_model_name: 全部条目，按 `model_name` 索引——理论上不应该出现
            跨来源冲突：`replace_source` 会在冲突发生时直接拒绝合并（见
            下方实现），冲突留给 `resolve_tools` 里 request 级定义按优先级
            处理，不允许两个 application 级来源静默覆盖彼此。
        by_source: `source_id -> canonical_name 元组`，用于下一次
            `replace_source` 时先摘除该来源现有的全部旧条目，而不是只增量
            追加（一个 Skill 被删除后，新快照里不能残留旧定义）。
    """

    revision: int
    created_at: datetime
    by_canonical_name: Mapping[str, ToolDefinition] = field(default_factory=dict)
    by_model_name: Mapping[str, ToolDefinition] = field(default_factory=dict)
    by_source: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "RegistrySnapshot":
        """构造一个空快照（revision=0），`ToolRegistry` 初始化时的起点。"""
        return cls(revision=0, created_at=datetime.now(timezone.utc))

    def replace_source(self, source_id: str, definitions: list[ToolDefinition]) -> "RegistrySnapshot":
        """用 `definitions` 整体替换某个来源现有的全部条目，返回一份新快照。

        本方法不修改 `self`（frozen dataclass），失败时直接抛异常，调用方
        （`ToolRegistry.publish()`）保证异常发生时 `self` 不会被当作新快照
        发布——这就是"静态校验失败保留旧状态"的落地位置。

        Args:
            source_id: 待替换的来源标识，如 "skill"、"builtin"、"mcp:github"。
            definitions: 该来源本次发现的全部定义；可以是空列表，代表把该
                来源从快照里整体摘除（MCP Server 断线时的做法，见设计文档
                4.4 节）。

        Returns:
            新的 `RegistrySnapshot`，`revision` 在 `self.revision` 基础上 +1。

        Raises:
            ValueError: `definitions` 里某条的 `source_id` 与参数不一致、
                同一来源内部 `canonical_name` 重复，或本次要写入的
                `model_name` 已被另一个仍然存活的来源占用（跨来源命名
                冲突，必须整体拒绝发布，不允许静默覆盖）。
        """
        seen_canonical: set[str] = set()
        for definition in definitions:
            if definition.source_id != source_id:
                raise ValueError(
                    f"definitions 里出现 source_id={definition.source_id!r}，"
                    f"与目标 source_id={source_id!r} 不一致"
                )
            if definition.canonical_name in seen_canonical:
                raise ValueError(f"来源 {source_id!r} 内部 canonical_name 重复: {definition.canonical_name!r}")
            seen_canonical.add(definition.canonical_name)

        by_canonical = dict(self.by_canonical_name)
        by_model = dict(self.by_model_name)

        # 先摘除该来源现有的全部旧条目（不是增量追加），保证"来源发布了
        # 更少的工具"这种情况能正确体现为快照里对应条目消失。
        for old_canonical_name in self.by_source.get(source_id, ()):
            old = by_canonical.pop(old_canonical_name, None)
            if old is not None and by_model.get(old.model_name) is old:
                del by_model[old.model_name]

        for definition in definitions:
            existing = by_model.get(definition.model_name)
            if existing is not None and existing.source_id != source_id:
                raise ValueError(
                    f"model_name={definition.model_name!r} 与已发布来源 "
                    f"{existing.source_id!r}（canonical_name={existing.canonical_name!r}）冲突，"
                    f"拒绝发布来源 {source_id!r} 的这批定义"
                )
            by_canonical[definition.canonical_name] = definition
            by_model[definition.model_name] = definition

        by_source = dict(self.by_source)
        if definitions:
            by_source[source_id] = tuple(d.canonical_name for d in definitions)
        else:
            by_source.pop(source_id, None)

        return RegistrySnapshot(
            revision=self.revision + 1,
            created_at=datetime.now(timezone.utc),
            by_canonical_name=by_canonical,
            by_model_name=by_model,
            by_source=by_source,
        )
