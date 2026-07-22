from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from robomex.dysc.contracts import load_contract_for_skill
from robomex.scripts.check_skills import main
from robomex.skills import Skill, SkillLibrary
from robomex.skills.lint import collect_library_errors

BUILTIN_ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def _load_sidecar(relative_path: str, module_name: str):
    path = BUILTIN_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builtin_skill_library_passes_m3_lint() -> None:
    assert collect_library_errors(BUILTIN_ROOT) == []
    assert len(list(BUILTIN_ROOT.glob("*/*/SKILL.md"))) == 21
    assert not list((BUILTIN_ROOT / "motif").rglob("SKILL.md"))


def test_check_skills_cli_passes_builtin_library() -> None:
    assert main([str(BUILTIN_ROOT)]) == 0


def test_v2_bowl_skills_are_contract_first_admissible_packages(tmp_path) -> None:
    packages = (
        ("monitor", "author_attachment_monitor"),
        ("affordance", "estimate_support_alignment"),
        ("state", "propose_attachment_transition"),
        ("state", "propose_relation_transition"),
        ("motion", "author_sealed_phase_motion"),
    )
    library = SkillLibrary(tmp_path / "admitted")
    for category, skill_id in packages:
        source = BUILTIN_ROOT / category / skill_id
        text = (source / "SKILL.md").read_text(encoding="utf-8")
        assert "## Reference Code" in text
        assert "result_var" in text and "NODE_RESULT" in text
        contract = load_contract_for_skill(source)
        assert contract is not None
        assert contract.skill_id == skill_id
        assert contract.input_ports or skill_id == "author_attachment_monitor"
        assert contract.output_ports
        assert contract.functions
        record = library.admit(Skill.from_dir(source))
        assert record.skill_id == skill_id
        assert (record.skill.root / "contract.yaml").is_file()
        assert (record.skill.root / "scripts").is_dir()


def test_build_grasp_trajectory_keeps_safety_waypoints_above_grasp() -> None:
    module = _load_sidecar(
        "motion/plan_bounded_motion/scripts/grasp_trajectory.py",
        "grasp_trajectory",
    )
    plan = module.build_grasp_trajectory(
        {
            "position": [0.4, -0.1, 0.05],
            "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
        }
    )
    waypoints = {item["name"]: item for item in plan["waypoints"]}
    grasp_z = waypoints["grasp"]["position_xyz"][2]
    assert waypoints["approach"]["position_xyz"][2] > grasp_z
    assert waypoints["lift"]["position_xyz"][2] > grasp_z


def test_build_grasp_trajectory_rejects_downward_explicit_approach() -> None:
    module = _load_sidecar(
        "motion/plan_bounded_motion/scripts/grasp_trajectory.py",
        "grasp_trajectory_invalid",
    )
    with pytest.raises(ValueError, match="must be above"):
        module.build_grasp_trajectory(
            {
                "position": [0.4, -0.1, 0.05],
                "approach_position": [0.4, -0.1, -0.1],
            }
        )
