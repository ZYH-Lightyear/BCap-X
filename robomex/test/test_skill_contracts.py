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
    assert "open_container" in text
    assert "support_surface" in text
    assert "bowl on a plate" in text.lower()
    assert "object's center" in text
    assert "Prohibited Shortcuts" in text
    assert "place_pos` can still be used" not in text
    assert "target_points" not in text
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


def test_task_skills_keep_grounding_and_affordance_in_act() -> None:
    pick = _read("task/pick_object")
    place = _read("task/place_object")

    assert "Act generates grounding and grasp candidates itself" in pick
    assert "Use the Verifier SubAgent only for state checks or failure diagnosis" in pick
    assert "Act computes placement affordance itself" in place
    assert "Verifier outputs are verdicts only" in place


def test_release_at_declares_offset_compensated_release_contract() -> None:
    text = _read("motion/release_at")

    assert "EVIDENCE[\"placement_affordance\"]" in text
    assert "EVIDENCE[\"held_object_frame\"]" in text
    assert "object_center_offset_from_grasp" in text
    assert "tcp_release_pos = desired_object_center - offset" in text
    assert "Do not align the" in text


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
        "## Workflow",
        "## Candidate Generation",
        "## Local Checks",
        "## Failure Modes",
        "## Clean Reusable Rules",
        "## Weak Priors",
        "## Prohibited Shortcuts",
        "## Artifacts to Save",
    )
    for skill_md in sorted(ROOT.glob("*/*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        for section in required:
            assert section in text, (skill_md, section)
        assert "```python" not in text, skill_md


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
    allowed_entries = {"SKILL.md", "assets", "references", "scripts"}
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
