from __future__ import annotations

from pathlib import Path

import pytest

from robomex.dysc.contracts import load_skill_contracts
from robomex.dysc.society import load_society_spec
from robomex.dysc.views import SkillLibraryView
from robomex.skills import (
    Skill,
    SkillCategory,
    SkillLibrary,
    load_builtin_skills,
)


def _admit_skill(library: SkillLibrary, skill_id: str, category: str, description: str) -> None:
    skill = Skill.from_markdown(
        "---\n"
        f"name: {skill_id}\n"
        f"category: {category}\n"
        f"description: {description}\n"
        "---\n\n"
        "## Purpose\n\nTest skill.",
        skill_id=skill_id,
    )
    library.admit(skill, source="test")


def test_society_spec_loads_role_policies_without_fixed_registration(tmp_path: Path) -> None:
    spec_path = tmp_path / "society.yaml"
    spec_path.write_text(
        """
name: seed_society
roles:
  perception_scout:
    objective: ground objects
    skill_access_policy: scout_policy
    consumes: [observation]
    emits: [object_grounding]
skill_access_policies:
  scout_policy:
    allow_tags: [perception]
    prefer: [relational_grounding]
    forbid_tags: [motion_execution]
topology:
  - from: perception_scout
    to: affordance_geometer
    when: object_grounding.available
""",
        encoding="utf-8",
    )

    society = load_society_spec(spec_path)

    assert society.name == "seed_society"
    assert society.roles["perception_scout"].skill_access_policy == "scout_policy"
    assert society.policy_for_role("perception_scout").allow_tags == ("perception",)
    assert society.topology[0]["from"] == "perception_scout"


def test_skill_library_view_filters_by_policy_and_contract_tags(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    _admit_skill(library, "relational_grounding", "perception", "ground by relation")
    _admit_skill(library, "controlled_release", "motion", "release object")

    (tmp_path / "library" / "perception" / "relational_grounding" / "contract.yaml").write_text(
        "skill_id: relational_grounding\ntags: [perception, grounding]\n",
        encoding="utf-8",
    )
    (tmp_path / "library" / "motion" / "controlled_release" / "contract.yaml").write_text(
        "skill_id: controlled_release\ntags: [motion_execution]\nchanges_world: true\n",
        encoding="utf-8",
    )
    contracts = load_skill_contracts(tmp_path / "library")

    society_path = tmp_path / "society.yaml"
    society_path.write_text(
        """
name: test
roles:
  scout:
    skill_access_policy: scout_policy
skill_access_policies:
  scout_policy:
    allow_tags: [perception]
    forbid_tags: [motion_execution]
""",
        encoding="utf-8",
    )
    society = load_society_spec(society_path)
    view = SkillLibraryView(library, society.policy_for_role("scout"), contracts)

    assert [record.skill_id for record in view.all()] == ["relational_grounding"]
    assert view.get("relational_grounding").skill.category == SkillCategory.PERCEPTION
    with pytest.raises(KeyError):
        view.get("controlled_release")


def test_skill_library_admit_copies_contract_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source_skill"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: Demo\ncategory: perception\ndescription: demo\n---\n\nBody.",
        encoding="utf-8",
    )
    (source / "contract.yaml").write_text(
        "skill_id: source_skill\ntags: [perception]\noutputs: [object_grounding]\n",
        encoding="utf-8",
    )

    library = SkillLibrary(tmp_path / "library")
    library.admit(Skill.from_dir(source), source="test")
    contracts = load_skill_contracts(tmp_path / "library")

    assert contracts["source_skill"].tags == ("perception",)
    assert contracts["source_skill"].outputs == ("object_grounding",)


def test_task_skills_are_guidance_and_leaf_contracts_own_typed_ports(tmp_path: Path) -> None:
    library = SkillLibrary(tmp_path / "library")
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")
    contracts = load_skill_contracts(library.root)

    pick = contracts["pick_object"]
    ground = contracts["segment_object"]
    action = contracts["grasp_object"]
    verifier = contracts["verify_grasp_and_lift_via_robot_state"]

    assert "recipe" not in pick.raw
    assert not pick.role
    assert [port.schema for port in ground.output_ports] == [
        "robomex.mask.v1",
        "robomex.points3d.v1",
    ]
    assert action.changes_world
    assert action.role == "action_executor"
    assert not verifier.changes_world
    assert verifier.role == "verifier"
    assert contracts["grasp_open_bowl"].output_ports[0].schema == "robomex.affordance.v1"
