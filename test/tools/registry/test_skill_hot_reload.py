"""SkillHotReloader.reload_once() 单元/故障测试（设计文档第五节 + 第八节）。

复用真实 `SkillLoader`（不 mock 它——它本身已经有"单个 Skill 解析失败只
跳过、不中断整体加载"的容错测试覆盖，见 test/skills/），只在"扫描阶段
本身抛异常"这一种 `SkillLoader` 设计上不会触发的极端场景下才用 monkeypatch。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_core.guardrail.guardrail_provider import GuardrailDecision
from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.tools.registry import skill_hot_reload as skill_hot_reload_module
from src.agent_core.tools.registry.skill_hot_reload import SkillHotReloader
from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.agent_core.tools.registry.tool_registry import ToolRegistry


class _FakeSkillManager:
    """`SkillHotReloader`/`SkillToolProvider` 只用得到这三个属性，不需要真实
    `SkillManager`（省去装配 GuardrailProvider/SandboxProvider 的成本）。
    """

    def __init__(self, registry: SkillRegistry, enabled: bool = True) -> None:
        self.registry = registry
        self.enabled = enabled
        self.tool_factory = object()  # discover() 只存闭包，不会真的调用 .create()


class _FakeGuardrailProvider:
    async def check(self, **kwargs):
        return GuardrailDecision.allow()


def _write_skill_md(skill_dir: Path, *, tool_name: str = "demo_skill", category: str = "general") -> None:
    """写一个 `TOOL` 形态技能（带 `parameters`）——本文件测的是发布/回滚/
    原子性这些注册中心机制，不是 Skill 形态分流（那部分见
    `test/tools/registry/test_providers.py`），用 `TOOL` 形态让 `discover()`
    继续产出与技能同名的独立 `ToolDefinition`，下面的断言不用因为 Skill 注入
    重构改动跟着变。
    """
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {tool_name}\ntool_name: {tool_name}\ndescription: 测试技能\ncategory: {category}\n"
        "parameters:\n  - name: query\n    type: string\n    required: true\n---\n\n正文\n",
        encoding="utf-8",
    )


async def test_reload_once_publishes_newly_added_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill")

    skill_manager = _FakeSkillManager(SkillRegistry())
    tool_registry = ToolRegistry()
    reloader = SkillHotReloader([skills_root], skill_manager, tool_registry, _FakeGuardrailProvider())

    await reloader.reload_once()

    assert "demo_skill" in skill_manager.registry.names
    assert "demo_skill" in tool_registry.current_snapshot().by_model_name
    assert tool_registry.current_snapshot().by_model_name["demo_skill"].source_type == "skill"


async def test_reload_once_removes_deleted_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo-skill"
    _write_skill_md(skill_dir)

    skill_manager = _FakeSkillManager(SkillRegistry())
    tool_registry = ToolRegistry()
    reloader = SkillHotReloader([skills_root], skill_manager, tool_registry, _FakeGuardrailProvider())
    await reloader.reload_once()
    assert "demo_skill" in tool_registry.current_snapshot().by_model_name

    # 模拟 Skill 目录被删除，再触发一次重新加载。
    import shutil

    shutil.rmtree(skill_dir)
    await reloader.reload_once()

    assert "demo_skill" not in skill_manager.registry.names
    assert "demo_skill" not in tool_registry.current_snapshot().by_model_name


async def test_reload_once_keeps_old_state_when_scan_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """扫描阶段异常时保留旧状态（`SkillLoader.load()` 设计上不会因为单个
    文件解析失败而抛出，这里用 monkeypatch 模拟"整体扫描失败"这类更极端的
    基础设施异常，验证 `reload_once()` 的兜底路径）。"""
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill")

    skill_manager = _FakeSkillManager(SkillRegistry())
    tool_registry = ToolRegistry()
    reloader = SkillHotReloader([skills_root], skill_manager, tool_registry, _FakeGuardrailProvider())
    await reloader.reload_once()
    old_registry = skill_manager.registry
    old_revision = tool_registry.current_snapshot().revision

    class _RaisingLoader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self):
            raise OSError("模拟磁盘故障")

    monkeypatch.setattr(skill_hot_reload_module, "SkillLoader", _RaisingLoader)

    await reloader.reload_once()

    assert skill_manager.registry is old_registry
    assert tool_registry.current_snapshot().revision == old_revision


async def test_reload_once_keeps_old_state_when_publish_rejected(tmp_path: Path) -> None:
    """新扫描出的 Skill 与已发布的其它来源命名冲突时，`publish()` 被拒绝，
    `skill_manager.registry` 也必须保持旧状态——不能出现"注册中心拒绝了，
    但 SkillManager 已经换到新状态"的不一致（本轮实现时发现并修复的顺序
    问题，见 `SkillToolProvider.discover()` 的 `registry` 参数文档）。"""
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill", tool_name="conflicting_tool")

    tool_registry = ToolRegistry()
    conflicting_builtin = ToolDefinition(
        canonical_name="builtin.conflicting_tool",
        model_name="conflicting_tool",
        description="",
        source_type="builtin",
        source_id="builtin",
        scope="application",
        permissions_key="conflicting_tool",
        build_tool=(lambda: object()),
    )
    await tool_registry.publish("builtin", [conflicting_builtin])

    old_registry = SkillRegistry()
    skill_manager = _FakeSkillManager(old_registry)
    reloader = SkillHotReloader([skills_root], skill_manager, tool_registry, _FakeGuardrailProvider())

    await reloader.reload_once()

    assert skill_manager.registry is old_registry
    assert "conflicting_tool" not in skill_manager.registry.names
    assert tool_registry.current_snapshot().by_model_name["conflicting_tool"].source_type == "builtin"


async def test_reload_once_is_noop_when_no_skill_dirs_exist(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does-not-exist"
    skill_manager = _FakeSkillManager(SkillRegistry())
    tool_registry = ToolRegistry()
    reloader = SkillHotReloader([missing_dir], skill_manager, tool_registry, _FakeGuardrailProvider())

    reloader.start()  # 全部目录都不存在，应为空操作，不应该抛异常

    assert reloader._task is None
