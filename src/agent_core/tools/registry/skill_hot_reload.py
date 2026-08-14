"""Skill 热重载：`watchfiles` 监听 SKILL.md 变化 -> 防抖 -> 重新扫描 -> 原子发布。

对应设计文档《工具注册中心与热重载设计.md》第五节：改 `SKILL.md` 不再需要
重启进程。`watchfiles` 已经在依赖树里（`uvicorn[standard]` 用它做
`--reload`），这里只是把它从"uvicorn 专用"提升为业务代码也直接 import。

复用 `SkillLoader`——它已经是"全量扫描 -> 单个失败跳过 -> 返回新
`SkillRegistry`"的实现，热重载不改它一行代码，只是在防抖后的每次触发都
重新调用一次。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from watchfiles import awatch

from src.agent_core.guardrail.guardrail_provider import GuardrailProvider
from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_manager import SkillManager
from src.agent_core.tools.registry.providers import SkillToolProvider
from src.agent_core.tools.registry.tool_registry import ToolRegistry

_DEFAULT_DEBOUNCE_MS = 500
_SKILL_MD_NAME = "SKILL.md"


class SkillHotReloader:
    """监听 `skill_dirs` 下全部 `SKILL.md` 的变化，防抖后重新扫描并原子发布。"""

    def __init__(
        self,
        skill_dirs: list[Path],
        skill_manager: SkillManager,
        tool_registry: ToolRegistry,
        guardrail_provider: GuardrailProvider,
        debounce_ms: int = _DEFAULT_DEBOUNCE_MS,
    ) -> None:
        """初始化热重载器（不启动 watcher，`start()` 才真正开始监听）。

        Args:
            skill_dirs: 待监听的技能根目录列表，与 `SKILLS_DIRS` 配置一致。
            skill_manager: 全局 `SkillManager` 单例，重新加载成功后原地
                替换其 `registry` 属性。
            tool_registry: 全局 `ToolRegistry` 单例，重新加载后发布到这里。
            guardrail_provider: 权限校验器，透传给 `SkillToolProvider`
                构造 `load_skill` 兜底实现。
            debounce_ms: 防抖窗口（毫秒），应对编辑器保存文件时常见的
                create+modify+rename 连续事件（设计文档 5.1 节，默认落在
                文档给出的 300~800ms 区间内）。
        """
        self._skill_dirs = [d for d in skill_dirs if d.exists()]
        self._skill_manager = skill_manager
        self._tool_registry = tool_registry
        self._guardrail_provider = guardrail_provider
        self._debounce_ms = debounce_ms
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """启动后台 watcher 任务。全部技能目录都不存在时是空操作。"""
        if not self._skill_dirs:
            logger.warning("[SkillHotReloader] 待监听的技能目录均不存在，热重载未启动")
            return
        self._task = asyncio.create_task(self._watch_loop(), name="skill-hot-reload")
        logger.info(f"[SkillHotReloader] 已启动，监听目录: {[str(d) for d in self._skill_dirs]}")

    async def stop(self) -> None:
        """取消 watcher 任务（应用关闭时调用）。未启动时是空操作。"""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _watch_loop(self) -> None:
        async for changes in awatch(*self._skill_dirs, debounce=self._debounce_ms):
            if any(Path(path).name == _SKILL_MD_NAME for _change_type, path in changes):
                await self.reload_once()

    async def reload_once(self) -> None:
        """执行一次"扫描 -> 候选发布"，供 watcher 触发，也可用于手动/测试调用。

        失败（扫描异常或发布被拒绝）时保留 `skill_manager`/`ToolRegistry`
        的旧状态不变，只记 error——不能因为一个 Skill 文件保存到一半就把
        线上正常的工具集摘掉（设计文档 5.4 节"发布与失败处理"）。
        """
        try:
            new_registry = SkillLoader(self._skill_dirs).load()
        except Exception as exc:
            logger.error(f"[SkillHotReloader] 扫描技能目录失败，保留旧状态: {exc}")
            return

        # 先用候选 registry 试算出 ToolDefinition、试发布，成功后才真正
        # 切换 skill_manager.registry——顺序不能反，否则 publish() 因命名
        # 冲突被拒绝时，SkillManager 已经换到了未被接受的新状态，两者不一致。
        candidate_definitions = SkillToolProvider(self._skill_manager, self._guardrail_provider).discover(
            registry=new_registry
        )
        try:
            snapshot = await self._tool_registry.publish(source_id="skill", definitions=candidate_definitions)
        except ValueError as exc:
            logger.error(f"[SkillHotReloader] 发布被拒绝（命名冲突），保留旧状态: {exc}")
            return

        self._skill_manager.registry = new_registry
        logger.info(
            f"[SkillHotReloader] 重新加载完成，共 {len(new_registry)} 个技能，"
            f"registry_revision={snapshot.revision}"
        )


_reloader: SkillHotReloader | None = None


def init_skill_hot_reloader(
    skill_dirs: list[Path],
    skill_manager: SkillManager,
    tool_registry: ToolRegistry,
    guardrail_provider: GuardrailProvider,
    debounce_ms: int = _DEFAULT_DEBOUNCE_MS,
) -> SkillHotReloader:
    """应用启动时调用一次，构造并启动全局单例。"""
    global _reloader
    _reloader = SkillHotReloader(skill_dirs, skill_manager, tool_registry, guardrail_provider, debounce_ms)
    _reloader.start()
    return _reloader


async def shutdown_skill_hot_reloader() -> None:
    """应用关闭时调用一次，停止 watcher。未初始化时是空操作。"""
    if _reloader is not None:
        await _reloader.stop()


def get_skill_hot_reloader() -> SkillHotReloader:
    """返回全局唯一的 SkillHotReloader 实例。

    Raises:
        RuntimeError: init_skill_hot_reloader() 尚未被调用。
    """
    if _reloader is None:
        raise RuntimeError("SkillHotReloader 尚未初始化，请确认应用已完成启动")
    return _reloader
