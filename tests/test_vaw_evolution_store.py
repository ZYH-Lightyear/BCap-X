from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vaw.evolution import (
    EvolutionLedger,
    EvolutionSpec,
    GateDecision,
    GatePair,
    GateReport,
    GenerationStore,
    MutationOperation,
    SkillMutation,
    skill_tree_digest,
)


def _write_skill(root: Path, skill_id: str, body: str) -> None:
    directory = root / skill_id
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(body, encoding="utf-8")


def _spec() -> EvolutionSpec:
    return EvolutionSpec(
        experiment_id="unit-evolution",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(0, 1),
        gate_tasks=(2,),
        reserve_tasks=(3,),
        heldout_suites=("libero_object_task", "libero_spatial_task", "libero_goal_task"),
        rollout_config={"model": "fake", "seeds": [1, 2]},
        frozen_components={"prompt": "abc", "canvas": "def"},
    )


def _mutation() -> SkillMutation:
    return SkillMutation(
        mutation_id="mutation-001",
        operation=MutationOperation.ADD,
        target_skill_ids=(),
        candidate_skill_id="new-skill",
        evidence_ids=("window-001",),
        rationale="成功与失败窗口显示需要补充通用接触判断。",
    )


def test_spec_round_trip_and_nested_config_are_immutable() -> None:
    spec = _spec()
    restored = EvolutionSpec.from_documents(spec.experiment_document(), spec.split_document())

    assert restored == spec
    assert restored.rollout_config["seeds"] == (1, 2)
    with pytest.raises(TypeError):
        restored.rollout_config["model"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="互不重叠"):
        EvolutionSpec(
            experiment_id="bad",
            base_generation="g000",
            evolve_suite="libero_90",
            evolve_tasks=(0,),
            gate_tasks=(0,),
            reserve_tasks=(),
            heldout_suites=(),
        )


def test_initialize_creates_frozen_generation_and_documents(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _write_skill(skills, "skill-a", "---\nname: A\ndescription: A\n---\n\n# A\n")
    root = tmp_path / "experiment"

    store = GenerationStore.initialize(
        root,
        _spec(),
        skills,
        created_at="2026-09-01T00:00:00+00:00",
    )

    assert store.active_generation() == "g000"
    assert store.read_spec() == _spec()
    manifest = store.read_manifest("g000")
    assert manifest.parent_generation is None
    assert manifest.skill_digest == skill_tree_digest(skills)
    assert (root / "generations/g000/skills/skill-a/SKILL.md").is_file()
    assert [event["event"] for event in store.ledger.read()] == [
        "experiment_initialized",
        "generation_activated",
    ]

    experiment_before = (root / "experiment.json").read_bytes()
    with pytest.raises(FileExistsError):
        GenerationStore.initialize(root, _spec(), skills)
    assert (root / "experiment.json").read_bytes() == experiment_before


def test_digest_depends_on_relative_path_and_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "skill-a", "same")
    _write_skill(second, "skill-a", "same")
    assert skill_tree_digest(first) == skill_tree_digest(second)

    (second / "skill-a/SKILL.md").write_text("changed", encoding="utf-8")
    assert skill_tree_digest(first) != skill_tree_digest(second)


def test_generation_is_atomic_and_duplicate_does_not_overwrite(tmp_path: Path) -> None:
    base_skills = tmp_path / "base-skills"
    candidate_skills = tmp_path / "candidate-skills"
    _write_skill(base_skills, "skill-a", "base")
    _write_skill(candidate_skills, "skill-a", "candidate")
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), base_skills)

    manifest = store.create_generation(
        "g001",
        parent_generation="g000",
        skill_root=candidate_skills,
        mutation=_mutation(),
        created_at="2026-09-01T01:00:00+00:00",
    )
    manifest_bytes = (store.generation_path("g001") / "manifest.json").read_bytes()
    assert manifest.parent_generation == "g000"
    assert store.active_generation() == "g000"

    with pytest.raises(FileExistsError):
        store.create_generation(
            "g001",
            parent_generation="g000",
            skill_root=base_skills,
            mutation=_mutation(),
        )
    assert (store.generation_path("g001") / "manifest.json").read_bytes() == manifest_bytes
    assert not any(path.name.startswith(".g001.") for path in store.generations_dir.iterdir())
    with pytest.raises(ValueError, match="单段目录名"):
        store.read_manifest("../outside")


def test_rollback_only_changes_pointer_and_appends_event(tmp_path: Path) -> None:
    base_skills = tmp_path / "base-skills"
    candidate_skills = tmp_path / "candidate-skills"
    _write_skill(base_skills, "skill-a", "base")
    _write_skill(candidate_skills, "skill-a", "candidate")
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), base_skills)
    store.create_generation(
        "g001",
        parent_generation="g000",
        skill_root=candidate_skills,
        mutation=_mutation(),
    )
    store.rollback("g001")

    generation_hashes = {
        generation: hashlib.sha256(
            (store.generation_path(generation) / "manifest.json").read_bytes()
        ).hexdigest()
        for generation in ("g000", "g001")
    }
    store.rollback("g000")

    assert store.active_generation() == "g000"
    assert generation_hashes == {
        generation: hashlib.sha256(
            (store.generation_path(generation) / "manifest.json").read_bytes()
        ).hexdigest()
        for generation in ("g000", "g001")
    }
    assert store.ledger.read()[-1]["event"] == "generation_rollback"
    assert store.ledger.read()[-1]["payload"]["previous_generation"] == "g001"


def test_ledger_is_append_only_and_gate_report_keeps_pair_facts(tmp_path: Path) -> None:
    ledger = EvolutionLedger(tmp_path / "ledger.jsonl")
    ledger.append("first", generation_id="g000", timestamp="t1")
    before = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    ledger.append("second", generation_id="g001", timestamp="t2")
    after = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert [record["event"] for record in ledger.read()] == ["first", "second"]

    report = GateReport(
        mutation_id="mutation-001",
        baseline_generation="g000",
        decision=GateDecision.PASSED,
        pairs=(
            GatePair("libero_90", 1, 1, False, True, True),
            GatePair("libero_90", 2, 1, True, False, False),
            GatePair("libero_90", 3, 1, False, True, True, infrastructure_error=True),
        ),
        checks={"privacy": True},
    )
    restored = GateReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored.recoveries == 1
    assert restored.regressions == 1
    assert restored.consulted_recoveries == 1
    assert restored.to_dict()["summary"] == report.to_dict()["summary"]
