"""`required_secrets` 三重交集机制的单元测试（设计文档 5.4 节）。

覆盖两层：
    1. `SkillLoader` 能正确解析 frontmatter 里的 `required_secrets` 列表，
       并跳过格式不合法的条目。
    2. `SkillToolFactory._resolve_secret_env` 的三重交集判断——技能声明 ×
       调用方本次请求提供，两者都满足才注入，任何一环缺失都不报错，只是
       拿不到这个密钥。
"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_definition import RequiredSecret, SkillDefinition
from src.agent_core.skills.skill_loader import SkillLoader
from src.agent_core.skills.skill_tool_factory import SkillToolFactory
from src.common.constants import SkillCategory

_SKILL_MD_TEMPLATE = """---
name: demo_skill
tool_name: demo_skill
description: 用于测试的技能
category: general
required_secrets:
  - name: ERP_API_TOKEN
    optional: false
  - name: OPTIONAL_TOKEN
  - not_a_dict_but_a_string
  - {}
---

## 技能指令

测试用正文。
"""


def _write_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_SKILL_MD_TEMPLATE, encoding="utf-8")
    return tmp_path


def test_skill_loader_parses_required_secrets(tmp_path: Path) -> None:
    """SkillLoader 应解析出合法条目，跳过缺少 name / 非字典的条目。"""
    root = _write_skill(tmp_path)
    registry = SkillLoader([root]).load()

    skill = registry.get("demo_skill")
    assert skill.required_secrets == [
        RequiredSecret(name="ERP_API_TOKEN", optional=False),
        RequiredSecret(name="OPTIONAL_TOKEN", optional=True),
    ]


def _build_skill(required_secrets: list[RequiredSecret]) -> SkillDefinition:
    return SkillDefinition(
        name="demo_skill",
        tool_name="demo_skill",
        description="",
        category=SkillCategory.GENERAL,
        skill_dir=Path("skills/public/demo-skill"),
        required_secrets=required_secrets,
    )


def _factory() -> SkillToolFactory:
    return SkillToolFactory(
        guardrail_provider=None,  # 本测试不触达 guardrail
        sandbox_provider=None,  # 本测试不触达 sandbox
        skill_script_timeout_seconds=60,
    )


def test_resolve_secret_env_three_way_intersection() -> None:
    """只有"frontmatter 声明 × 调用方提供"都满足的密钥才会被注入。"""
    skill = _build_skill([RequiredSecret(name="A"), RequiredSecret(name="B")])
    secrets = {"A": "value-a", "C": "value-c"}  # B 未提供，C 未声明

    result = _factory()._resolve_secret_env(skill, secrets)

    assert result == {"A": "value-a"}


def test_resolve_secret_env_no_required_secrets_declared() -> None:
    """技能没有声明 required_secrets 时，即使调用方提供了 secrets 也不注入任何内容。"""
    skill = _build_skill([])
    secrets = {"A": "value-a"}

    result = _factory()._resolve_secret_env(skill, secrets)

    assert result == {}


def test_resolve_secret_env_caller_did_not_provide_secrets() -> None:
    """调用方本次请求没有传 secrets 字段时，不报错，返回空字典。"""
    skill = _build_skill([RequiredSecret(name="A")])

    result = _factory()._resolve_secret_env(skill, secrets={})

    assert result == {}
