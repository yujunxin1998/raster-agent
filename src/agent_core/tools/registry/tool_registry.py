"""ToolRegistry：应用级工具登记中心（设计文档 4.1 节）。

沿用项目里 `GuardrailProvider`/`SandboxProvider`/`SkillManager` 一以贯之的
`init_xxx()`/`get_xxx()` 全局单例接入方式：`main.py` 启动时 `init_tool_registry()`
一次，其余各处 `get_tool_registry()` 取用；未初始化时 `raise RuntimeError`。
"""
from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger

from src.agent_core.tools.registry.snapshot import RegistrySnapshot
from src.agent_core.tools.registry.tool_definition import ToolDefinition

_MAX_ROLLBACK_HISTORY = 5


class ToolRegistry:
    """进程内单例：持有当前 `RegistrySnapshot`，支持按来源原子替换与回滚。

    不做多实例同步（设计文档"非目标"一节）——当前是单进程部署，
    进程内不可变快照已经拿到"读不加锁、原子切换"的全部收益；真的要横向
    扩容时再把内部存储换成"本地缓存 + Redis/Postgres 广播"。
    """

    def __init__(self) -> None:
        self._snapshot = RegistrySnapshot.empty()
        self._write_lock = asyncio.Lock()
        # 内存态即可，单进程不需要落库（设计文档 5.5 节）；保留最近几次
        # 发布前的快照，供 `rollback()` 使用。
        self._history: deque[RegistrySnapshot] = deque(maxlen=_MAX_ROLLBACK_HISTORY)

    def current_snapshot(self) -> RegistrySnapshot:
        """返回当前快照。读不加锁——快照发布后不可变，替换是整体引用切换。"""
        return self._snapshot

    async def publish(self, source_id: str, definitions: list[ToolDefinition]) -> RegistrySnapshot:
        """原子发布某个来源的最新定义列表。

        `_write_lock` 保证并发 `publish()` 串行执行；校验（`replace_source`
        内部做的 canonical_name/model_name 冲突检测）失败时直接抛异常，
        `self._snapshot` 保持不变，不影响在途请求和其它来源。

        Args:
            source_id: 来源标识，如 "builtin"、"skill"、"mcp:github"。
            definitions: 该来源本次发现的全部定义，可为空列表（整体摘除
                该来源，如 MCP Server 断线时）。

        Returns:
            发布成功后的新快照。

        Raises:
            ValueError: 详见 `RegistrySnapshot.replace_source`。
        """
        async with self._write_lock:
            candidate = self._snapshot.replace_source(source_id, definitions)
            self._history.append(self._snapshot)
            self._snapshot = candidate
            logger.info(
                f"[ToolRegistry] 发布 source_id={source_id!r} "
                f"tools={len(definitions)} revision={candidate.revision}"
            )
            return candidate

    async def rollback(self, to_revision: int) -> RegistrySnapshot:
        """回滚到最近历史窗口内的某个 revision（设计文档 5.5 节）。

        Args:
            to_revision: 目标 revision，必须在最近 `_MAX_ROLLBACK_HISTORY`
                次发布之内。

        Returns:
            回滚后的快照（即历史里 revision=to_revision 的那份）。

        Raises:
            ValueError: to_revision 不在可回滚窗口内。
        """
        async with self._write_lock:
            if self._snapshot.revision == to_revision:
                return self._snapshot
            for historical in reversed(self._history):
                if historical.revision == to_revision:
                    self._history.append(self._snapshot)
                    self._snapshot = historical
                    logger.warning(f"[ToolRegistry] 回滚至 revision={to_revision}")
                    return historical
            raise ValueError(f"revision={to_revision} 不在可回滚窗口内（最近 {_MAX_ROLLBACK_HISTORY} 次发布）")


_registry: ToolRegistry | None = None


def init_tool_registry() -> ToolRegistry:
    """应用启动时调用一次，注册全局单例。"""
    global _registry
    _registry = ToolRegistry()
    logger.info("[ToolRegistry] 初始化完成")
    return _registry


def get_tool_registry() -> ToolRegistry:
    """返回全局唯一的 ToolRegistry 实例。

    Raises:
        RuntimeError: init_tool_registry() 尚未被调用。
    """
    if _registry is None:
        raise RuntimeError("ToolRegistry 尚未初始化，请确认应用已完成启动")
    return _registry
