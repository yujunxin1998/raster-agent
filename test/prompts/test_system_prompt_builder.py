"""`SystemPromptBuilder` 单元测试：条件拼装六大模块。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_core.prompts.system_prompt_builder import SystemPromptBuilder


@pytest.fixture
def fake_system_dir(tmp_path: Path) -> Path:
    agent_dir = tmp_path / "fake_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "role.md").write_text("你是测试助手。", encoding="utf-8")
    (agent_dir / "thinking_style.md").write_text("常规节奏。", encoding="utf-8")
    (agent_dir / "clarification_system.md").write_text("先澄清目标。", encoding="utf-8")
    (agent_dir / "skill_system.md").write_text("可以用技能 A。", encoding="utf-8")
    (agent_dir / "subagent_system.md").write_text("可以委派任务。", encoding="utf-8")
    (agent_dir / "response_style.md").write_text("回答要简洁。", encoding="utf-8")
    return tmp_path


def test_all_switches_enabled_includes_all_modules_in_order(fake_system_dir: Path) -> None:
    builder = SystemPromptBuilder(system_dir=fake_system_dir)
    result = builder.build("fake_agent")

    expected_order = [
        "<role>", "<thinking_style>", "<clarification_system>",
        "<skill_system>", "<subagent_system>", "<response_style>",
    ]
    positions = [result.index(tag) for tag in expected_order]
    assert positions == sorted(positions)
    for tag in expected_order:
        assert tag in result
        assert tag.replace("<", "</") in result


def test_subagent_disabled_omits_module_and_tag_entirely(fake_system_dir: Path) -> None:
    builder = SystemPromptBuilder(system_dir=fake_system_dir)
    result = builder.build("fake_agent", subagent_enabled=False)

    assert "<subagent_system>" not in result
    assert "可以委派任务" not in result
    # 其它模块不受影响
    assert "<role>" in result
    assert "<skill_system>" in result


def test_skill_disabled_omits_module_and_tag_entirely(fake_system_dir: Path) -> None:
    builder = SystemPromptBuilder(system_dir=fake_system_dir)
    result = builder.build("fake_agent", skill_enabled=False)

    assert "<skill_system>" not in result
    assert "可以用技能 A" not in result


def test_clarification_and_response_style_disabled(fake_system_dir: Path) -> None:
    builder = SystemPromptBuilder(system_dir=fake_system_dir)
    result = builder.build("fake_agent", clarification_enabled=False, response_style_enabled=False)

    assert "<clarification_system>" not in result
    assert "<response_style>" not in result
    assert "<role>" in result
    assert "<thinking_style>" in result
    assert "<skill_system>" in result
    assert "<subagent_system>" in result


def test_thinking_disabled_omits_thinking_style_module(fake_system_dir: Path) -> None:
    builder = SystemPromptBuilder(system_dir=fake_system_dir)
    result = builder.build("fake_agent", thinking_enabled=False)

    assert "<thinking_style>" not in result
    assert "常规节奏" not in result


def test_missing_module_file_is_skipped_without_raising(tmp_path: Path) -> None:
    agent_dir = tmp_path / "partial_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "role.md").write_text("你是部分配置的助手。", encoding="utf-8")
    # 其余 5 个模块文件都不存在

    builder = SystemPromptBuilder(system_dir=tmp_path)
    result = builder.build("partial_agent")

    assert "<role>" in result
    for tag in ("thinking_style", "clarification_system", "skill_system",
                "subagent_system", "response_style"):
        assert f"<{tag}>" not in result


def test_lead_agent_real_modules_build_successfully() -> None:
    """回归测试：真实的 `system/lead_agent/` 六个模块文件都存在且能正常拼装。"""
    from src.agent_core.prompts.system_prompt_builder import system_prompt_builder

    result = system_prompt_builder.build(
        "lead_agent", thinking_enabled=True, subagent_enabled=True, skill_enabled=True,
    )

    for tag in ("role", "thinking_style", "clarification_system",
                "skill_system", "subagent_system", "response_style"):
        assert f"<{tag}>" in result
        assert f"</{tag}>" in result
