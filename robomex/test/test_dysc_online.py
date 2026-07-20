from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from robomex.dysc.online import DySCOnlineConfig, OnlineEvolutionManager
from robomex.dysc.patch import SocietyPatch, SocietyPatchError, apply_society_patch
from robomex.dysc.society import SocietySpec
from robomex.skills import Skill, SkillLibrary


class QueuePolicy:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[list[dict]] = []

    def complete(self, prompt: list[dict]) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


@dataclass(frozen=True)
class FakeSubGoal:
    goal: str
    postcondition: str


@dataclass(frozen=True)
class FakeSubGoalResult:
    subgoal: FakeSubGoal
    success: bool = False
    note: str = ""


def _skill(skill_id: str, category: str = "motion") -> Skill:
    return Skill.from_markdown(
        "---\n"
        f"name: {skill_id}\n"
        f"category: {category}\n"
        f"description: {skill_id}\n"
        "---\n\n"
        "## Purpose\n\nTest skill.",
        skill_id=skill_id,
    )


def _library(tmp_path: Path) -> SkillLibrary:
    library = SkillLibrary(tmp_path / "library")
    library.admit(_skill("controlled_release", "motion"), source="test")
    library.admit(_skill("offset_aware_place_with_hover_verify", "motif"), source="test")
    contract = tmp_path / "library" / "motif" / "offset_aware_place_with_hover_verify" / "contract.yaml"
    contract.write_text(
        "skill_id: offset_aware_place_with_hover_verify\ntags: [motif, placement]\n",
        encoding="utf-8",
    )
    return library


def _society() -> SocietySpec:
    return SocietySpec.from_mapping({
        "name": "seed",
        "roles": {
            "act_executor": {
                "objective": "execute",
                "skill_access_policy": "executor_policy",
                "execution_boundary": "motion_allowed",
            }
        },
        "skill_access_policies": {
            "executor_policy": {
                "allow_tags": ["motif"],
                "prefer": ["controlled_release"],
            }
        },
        "topology": [{"from": "act_executor", "to": "curator", "when": "subgoal_end"}],
    })


def test_society_patch_applies_llm_skill_preference_without_rules() -> None:
    patch = SocietyPatch.from_mapping({
        "rationale": "Need hover verification before release.",
        "mutations": [
            {
                "op": "add_skill_preference",
                "policy": "executor_policy",
                "skill": "offset_aware_place_with_hover_verify",
            }
        ],
    })

    updated = apply_society_patch(_society(), patch)

    assert "offset_aware_place_with_hover_verify" in updated.skill_access_policies["executor_policy"].prefer


def test_society_patch_rejects_unknown_policy() -> None:
    patch = SocietyPatch.from_mapping({
        "mutations": [{"op": "add_skill_preference", "policy": "missing", "skill": "x"}],
    })

    with pytest.raises(SocietyPatchError):
        apply_society_patch(_society(), patch)


def test_online_evolution_writes_next_society_version(tmp_path: Path) -> None:
    society_path = tmp_path / "society.yaml"
    society_path.write_text(_society().to_yaml(), encoding="utf-8")
    policy = QueuePolicy([
        """
task_outcome:
  self_claimed_success: true
  env_success: false
candidate_lessons:
  - placement needs hover verification
tags: [predicate_mismatch]
""",
        """
rationale: Add a motif preference for offset-aware placement.
mutations:
  - op: add_skill_preference
    policy: executor_policy
    skill: offset_aware_place_with_hover_verify
""",
    ])
    manager = OnlineEvolutionManager(
        config=DySCOnlineConfig(society_path=str(society_path), society_dir=str(tmp_path / "versions")),
        library=_library(tmp_path),
        policy=policy,
        artifacts_dir=tmp_path / "run",
    )

    manager.evolve_after_subgoal(
        task="place bowl on plate",
        subgoal_dir=None,
        subgoal_result=FakeSubGoalResult(FakeSubGoal("place bowl", "bowl on plate")),
    )

    assert manager.store.latest_version() == 1
    assert "offset_aware_place_with_hover_verify" in manager.current.skill_access_policies["executor_policy"].prefer
    assert (tmp_path / "versions" / "society_patch_001.yaml").exists()
    assert len(policy.prompts) == 2


def test_online_manager_motion_role_view_uses_current_society(tmp_path: Path) -> None:
    society_path = tmp_path / "society.yaml"
    society_path.write_text(_society().to_yaml(), encoding="utf-8")
    manager = OnlineEvolutionManager(
        config=DySCOnlineConfig(society_path=str(society_path), society_dir=str(tmp_path / "versions")),
        library=_library(tmp_path),
        policy=QueuePolicy([]),
        artifacts_dir=tmp_path / "run",
    )

    view = manager.library_view_for_motion_role()

    assert [record.skill_id for record in view.all()] == [
        "controlled_release",
        "offset_aware_place_with_hover_verify",
    ]
