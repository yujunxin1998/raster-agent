"""`skill_router.route()` 规则路由单元测试。"""
from __future__ import annotations

from pathlib import Path

from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.skills.skill_router import route
from src.common.constants import SkillCategory


def _skill(name: str, activation: str = "automatic") -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description="",
        category=SkillCategory.GENERAL,
        skill_dir=Path(f"skills/public/{name}"),
        activation=activation,  # type: ignore[arg-type]
    )


def test_required_skill_goes_to_forced_and_auto_activate() -> None:
    decision = route([_skill("must-run", activation="required")])

    assert decision.forced == ("must-run",)
    assert decision.auto_activate == ("must-run",)
    assert decision.catalog_candidates == ()
    assert decision.rejected == ()
    assert decision.reasons["must-run"] == "required"


def test_explicitly_requested_skill_goes_to_forced_even_if_automatic() -> None:
    decision = route([_skill("data-analysis")], explicit_skill_names=frozenset({"data-analysis"}))

    assert decision.forced == ("data-analysis",)
    assert decision.reasons["data-analysis"] == "explicit"


def test_automatic_skill_without_explicit_request_is_catalog_candidate_only() -> None:
    decision = route([_skill("data-analysis")])

    assert decision.forced == ()
    assert decision.auto_activate == ()
    assert decision.catalog_candidates == ("data-analysis",)
    assert decision.rejected == ()


def test_explicit_only_skill_not_specified_is_rejected_not_catalog() -> None:
    decision = route([_skill("admin-only", activation="explicit_only")])

    assert decision.rejected == ("admin-only",)
    assert decision.catalog_candidates == ()
    assert decision.forced == ()


def test_explicit_only_skill_specified_is_forced() -> None:
    decision = route(
        [_skill("admin-only", activation="explicit_only")],
        explicit_skill_names=frozenset({"admin-only"}),
    )

    assert decision.forced == ("admin-only",)
    assert decision.rejected == ()


def test_mixed_batch_partitions_every_skill_exactly_once() -> None:
    skills = [
        _skill("required-one", activation="required"),
        _skill("explicit-one"),
        _skill("automatic-one"),
        _skill("explicit-only-one", activation="explicit_only"),
    ]

    decision = route(skills, explicit_skill_names=frozenset({"explicit-one"}))

    assert set(decision.forced) == {"required-one", "explicit-one"}
    assert decision.catalog_candidates == ("automatic-one",)
    assert decision.rejected == ("explicit-only-one",)
    all_names = {s.name for s in skills}
    covered = set(decision.forced) | set(decision.catalog_candidates) | set(decision.rejected)
    assert covered == all_names


def test_scores_are_deterministic_placeholder_values() -> None:
    decision = route(
        [_skill("required-one", activation="required"), _skill("automatic-one")],
    )

    assert decision.scores["required-one"] == 1.0
    assert decision.scores["automatic-one"] == 0.0
