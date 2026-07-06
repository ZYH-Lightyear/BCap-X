from __future__ import annotations

from pathlib import Path

import pytest

from robomex.dysc.contracts import load_skill_contracts
from robomex.dysc.society import load_society_spec
from robomex.dysc.views import SkillLibraryView
from robomex.skills import Skill, SkillCategory, SkillLibrary


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
