from __future__ import annotations

from pathlib import Path
import re

from robomex.skills import Skill


ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def _read(rel: str) -> str:
    return (ROOT / rel / "SKILL.md").read_text(encoding="utf-8")


def test_find_placement_declares_object_center_affordance_contract() -> None:
    text = _read("affordance/find_placement")

    assert "EVIDENCE[\"placement_affordance\"]" in text
    assert "desired_object_center" in text
    assert "compute_drop_affordance" in text
    assert "open_container" in text
    assert "support_surface" in text
    assert "bowl on a plate" in text.lower()
    assert "object center" in text
    assert "TCP drop" in text or "TCP drop pose" in text
    assert "Prohibited Shortcuts" in text
    assert "rim + 0.10" in text or "rim + 0.10 m" in text
    assert "place_pos` can still be used" not in text
    # target_points is the declared input port; referencing INPUTS["target_points"]
    # is correct, but it must never appear as a sidecar kwarg (the param is `points`).
    assert "target_points=" not in text
    assert "bowl_points" not in text


def test_segment_object_declares_canonical_grounding_contract() -> None:
    text = _read("perception/segment_object")

    assert "EVIDENCE[\"object_grounding\"]" in text
    assert "EVIDENCE[\"grounding.mask\"]" in text
    assert "EVIDENCE[\"grounding.points\"]" in text
    assert "referring expression" in text
    assert "Use dedicated grounding APIs" in text
    assert "Do not spend turns printing `obs.keys()`" in text
    assert "query_vlm` only for categorical" in text
    assert "target_points" not in text
    assert "bowl_points" not in text


def test_skill_frontmatter_id_fields_do_not_define_routing_id() -> None:
    skill = Skill.from_markdown(
        "---\n"
        "id: stale_id\n"
        "skill_id: stale_skill_id\n"
        "name: Display Name\n"
        "category: perception\n"
        "description: demo\n"
        "---\n\n"
        "Body.",
        skill_id="directory_id",
    )

    assert skill.skill_id == "directory_id"
    assert skill.meta["id"] == "stale_id"
    assert skill.meta["skill_id"] == "stale_skill_id"


def test_task_skills_guide_dynamic_specialist_graphs() -> None:
    pick = _read("task/pick_object")
    place = _read("task/place_object")

    assert "generate a graph that fits the live scene" in pick
    assert "Add geometry analysis only when" in pick
    assert "Only its hard passed report" in pick
    assert "MotionPlanner owns trajectory feasibility" in place
    assert "validated edges generated" in place
    assert "build_place_trajectory" in place
    assert "desired_object_center" in place
    assert "settle" in place.lower()


def test_release_at_is_action_execution_not_replanning() -> None:
    text = _read("motion/release_at")

    assert "supplied trajectory artifact" in text
    assert "Placement" in text and "MotionPlanner" in text
    assert "Do not ground" in text
    assert "do not combine planning and execution" in text
    assert "stops the sequence immediately" in text
    assert "settle_after_open" in text
    assert "Do not open the gripper and immediately retreat" in text


def test_plan_bounded_motion_requires_place_template() -> None:
    text = _read("motion/plan_bounded_motion")

    assert "build_place_trajectory" in text
    assert "transport_hover" in text
    assert "release_descend" in text
    assert "open_settle" in text
    assert "desired_object_center" in text
    assert "Do not treat `desired_object_center` as TCP" in text


def test_open_bowl_declares_held_object_frame_contract() -> None:
    text = _read("affordance/grasp_open_bowl")

    assert "EVIDENCE[\"held_object_frame\"]" in text
    assert "object_center_at_grasp" in text
    assert "object_center_offset_from_grasp" in text
    assert "align the object center" in text
    assert "EVIDENCE[\"object_grounding\"]" in text
    assert "EVIDENCE[\"grasp_affordance\"]" in text
    assert "`pos`" in text
    assert "`quat`" in text
    assert "`ik_ok`" in text
    assert "`grasp_pos`" in text


def test_all_builtin_skills_use_workflow_memory_template() -> None:
    required = (
        "## Purpose",
        "## When to use",
        "## When NOT to use",
        "## Workflow",
        "## Candidate Generation",
        "## Local Checks",
        "## Failure Modes",
        "## Clean Reusable Rules",
        "## Weak Priors",
        "## Prohibited Shortcuts",
        "## Artifacts to Save",
        "## Multimodal Evidence Contract",
    )
    for skill_md in sorted(ROOT.glob("*/*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        for section in required:
            assert section in text, (skill_md, section)


def test_skills_with_sidecar_scripts_document_reference_code() -> None:
    """A sidecar without an exact signature forces the agent to probe it at
    runtime (dir()/inspect), burning turn budget; the reference block is the
    single evolving place where discovered call patterns are written back."""

    for skill_md in sorted(ROOT.glob("*/*/SKILL.md")):
        scripts_dir = skill_md.parent / "scripts"
        if not scripts_dir.is_dir():
            continue
        text = skill_md.read_text(encoding="utf-8")
        assert "## Reference Code" in text, skill_md
        section = text.split("## Reference Code", 1)[1]
        assert "```python" in section, skill_md


def test_skill_guidance_does_not_reintroduce_fixed_subagent_roles_or_camera_indexing() -> None:
    forbidden = (
        "verifier-style SubAgent",
        "affordance SubAgent",
        "`affordance` SubAgent",
        "grounding SubAgent",
        "Grounding Object SubAgent",
        "Affordance SubAgent",
    )
    for skill_md in ROOT.glob("*/*/**/*.md"):
        text = skill_md.read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, (skill_md, phrase)
        assert 'obs["agentview"]' not in text, skill_md
        assert "obs['agentview']" not in text, skill_md


def test_builtin_skill_packages_use_only_claude_style_sidecar_layout() -> None:
    allowed_entries = {
        "SKILL.md",
        "assets",
        "references",
        "scripts",
        "prompts",
        "contract.yaml",
    }
    for skill_dir in sorted(path.parent for path in ROOT.glob("*/*/SKILL.md")):
        entries = {p.name for p in skill_dir.iterdir()}
        assert "reference" not in entries, skill_dir
        assert entries <= allowed_entries, (skill_dir, entries - allowed_entries)


def test_skill_sidecar_paths_mentioned_in_guidance_exist() -> None:
    sidecar_ref = re.compile(r"\b(?P<dir>scripts|references)/(?P<name>[A-Za-z0-9_.-]+)")
    for skill_md in sorted(ROOT.glob("*/*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        for match in sidecar_ref.finditer(text):
            path = skill_md.parent / match.group("dir") / match.group("name")
            assert path.exists(), f"{skill_md} references missing sidecar {path}"
