"""SkillHotReloader.reload_once() 单元/故障测试。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 13 节简化后的语义：Skill 变化
只原子替换 `SkillManager.registry`，不再经过 `ToolRegistry.publish()` 两阶段
协调，也不推高 `ToolRegistry` revision（Skill 不再投影成独立
`ToolDefinition`）。

复用真实 `SkillLoader`（不 mock 它——它本身已经有"单个 Skill 解析失败只
跳过、不中断整体加载"的容错测试覆盖，见 test/skills/），只在"扫描阶段
本身抛异常"这一种 `SkillLoader` 设计上不会触发的极端场景下才用 monkeypatch。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.agent_core.skills.skill_registry import SkillRegistry
from src.agent_core.tools.registry import skill_hot_reload as skill_hot_reload_module
from src.agent_core.tools.registry.skill_hot_reload import SkillHotReloader


class _FakeSkillManager:
    """`SkillHotReloader` 只用得到 `registry` 属性，不需要真实 `SkillManager`
    （省去装配 GuardrailProvider/SandboxProvider 的成本）。
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry


def _write_skill_md(skill_dir: Path, *, name: str = "demo-skill", category: str = "general") -> None:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: 测试技能\ncategory: {category}\n---\n\n正文\n",
        encoding="utf-8",
    )


async def test_reload_once_picks_up_newly_added_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill")

    skill_manager = _FakeSkillManager(SkillRegistry())
    reloader = SkillHotReloader([skills_root], skill_manager)

    await reloader.reload_once()

    assert "demo-skill" in skill_manager.registry.names


async def test_reload_once_increments_revision(tmp_path: Path) -> None:
    """每次成功重载后 revision 单调 +1，供
    `subagent_capability_resolver.py` 等场景追溯"这次派发依据的是哪个版本
    的技能注册表"。"""
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill")

    skill_manager = _FakeSkillManager(SkillRegistry())
    reloader = SkillHotReloader([skills_root], skill_manager)
    assert skill_manager.registry.revision == 0

    await reloader.reload_once()
    assert skill_manager.registry.revision == 1

    await reloader.reload_once()
    assert skill_manager.registry.revision == 2


async def test_reload_once_removes_deleted_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "demo-skill"
    _write_skill_md(skill_dir)

    skill_manager = _FakeSkillManager(SkillRegistry())
    reloader = SkillHotReloader([skills_root], skill_manager)
    await reloader.reload_once()
    assert "demo-skill" in skill_manager.registry.names

    shutil.rmtree(skill_dir)
    await reloader.reload_once()

    assert "demo-skill" not in skill_manager.registry.names


async def test_reload_once_keeps_old_state_when_scan_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """扫描阶段异常时保留旧状态（`SkillLoader.load()` 设计上不会因为单个
    文件解析失败而抛出，这里用 monkeypatch 模拟"整体扫描失败"这类更极端的
    基础设施异常，验证 `reload_once()` 的兜底路径）。"""
    skills_root = tmp_path / "skills"
    _write_skill_md(skills_root / "demo-skill")

    skill_manager = _FakeSkillManager(SkillRegistry())
    reloader = SkillHotReloader([skills_root], skill_manager)
    await reloader.reload_once()
    old_registry = skill_manager.registry

    class _RaisingLoader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def load(self):
            raise OSError("模拟磁盘故障")

    monkeypatch.setattr(skill_hot_reload_module, "SkillLoader", _RaisingLoader)

    await reloader.reload_once()

    assert skill_manager.registry is old_registry


async def test_reload_once_is_noop_when_no_skill_dirs_exist(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does-not-exist"
    skill_manager = _FakeSkillManager(SkillRegistry())
    reloader = SkillHotReloader([missing_dir], skill_manager)

    reloader.start()  # 全部目录都不存在，应为空操作，不应该抛异常

    assert reloader._task is None
