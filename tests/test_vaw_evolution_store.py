from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vaw.evolution import (
    CandidateEvidence,
    CandidatePackage,
    EpisodeOutcome,
    EvolutionLedger,
    EvolutionSpec,
    GateDecision,
    GatePair,
    GatePolicy,
    GateReport,
    GenerationStore,
    MutationOperation,
    SkillMutation,
    skill_tree_digest,
)


def _write_skill(root: Path, skill_id: str, body: str) -> Path:
    directory = root / skill_id
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(body, encoding="utf-8")
    return directory


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
        gate_policy=GatePolicy(
            primary_seeds=(1, 2),
            confirmation_seeds=(3,),
            min_recoveries=2,
            min_attributed_recoveries=1,
        ),
    )


def _mutation(
    operation: MutationOperation = MutationOperation.ADD,
    *,
    skill_id: str = "new-skill",
) -> SkillMutation:
    return SkillMutation(
        mutation_id=f"mutation-{operation.value}",
        operation=operation,
        skill_id=skill_id,
        evidence_ids=("e1",),
        rationale="真实视觉证据支持这一项可归因的技能变更。",
    )


def _audit_roots(tmp_path: Path, name: str) -> tuple[Path, Path]:
    evolver = tmp_path / f"evolver-{name}"
    review = tmp_path / f"review-{name}"
    evolver.mkdir()
    review.mkdir()
    for filename in ("request.json", "response.json"):
        (evolver / filename).write_text("{}\n", encoding="utf-8")
        (review / filename).write_text("{}\n", encoding="utf-8")
    (review / "report.json").write_text(
        '{"decision": "accept", "reason": "evidence-backed"}\n',
        encoding="utf-8",
    )
    return evolver, review


def _candidate(
    tmp_path: Path,
    mutation: SkillMutation,
    *,
    body: str | None = None,
    name: str | None = None,
    with_reference: bool = False,
) -> CandidatePackage:
    m3_output = tmp_path / f"m3-{name or mutation.operation.value}"
    m3_output.mkdir()
    raster = m3_output / "context.png"
    raster.write_bytes(b"policy-visible-raster")
    evidence = CandidateEvidence(
        m3_output=str(m3_output),
        finding_id="finding-1",
        raster="context.png",
        sha256=hashlib.sha256(raster.read_bytes()).hexdigest(),
    )
    skill_root = None
    if mutation.operation is not MutationOperation.RETIRE:
        skill_root = tmp_path / f"skill-{name or mutation.operation.value}"
        skill_root.mkdir()
        (skill_root / "SKILL.md").write_text(body or "candidate", encoding="utf-8")
        if with_reference:
            references = skill_root / "references"
            references.mkdir()
            (references / "index.json").write_text('{"references": []}\n', encoding="utf-8")
            (references / "r001.png").write_bytes(b"reference-raster")
    candidate_name = name or mutation.operation.value
    evolver, review = _audit_roots(tmp_path, candidate_name)
    return CandidatePackage.create(
        tmp_path / f"candidate-{candidate_name}",
        mutation=mutation,
        evidence={"e1": evidence},
        skill_root=skill_root,
        evolver_root=evolver,
        review_root=review,
    )


def test_spec_v2_round_trip_freezes_gate_policy_and_rejects_v1() -> None:
    spec = _spec()
    restored = EvolutionSpec.from_documents(spec.experiment_document(), spec.split_document())

    assert restored == spec
    assert restored.schema == "vaw-evolution-v2"
    assert restored.rollout_config["seeds"] == (1, 2)
    assert restored.gate_policy.confirmation_seeds == (3,)
    assert restored.gate_policy.max_turn_ratio == 1.5
    assert restored.gate_policy.max_token_ratio == 1.5
    with pytest.raises(TypeError):
        restored.rollout_config["model"] = "changed"  # type: ignore[index]

    old_experiment = dict(spec.experiment_document(), schema="vaw-evolution-v1")
    with pytest.raises(ValueError, match="不支持的 schema"):
        EvolutionSpec.from_documents(old_experiment, spec.split_document())
    with pytest.raises(ValueError, match="互不重叠"):
        GatePolicy(primary_seeds=(1,), confirmation_seeds=(1,))


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
    assert manifest.schema == "vaw-generation-v2"
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


def test_candidate_package_is_atomic_and_evidence_backed(tmp_path: Path) -> None:
    mutation = _mutation()
    candidate = _candidate(tmp_path, mutation, with_reference=True)

    assert candidate.mutation == mutation
    assert candidate.skill_root == candidate.root / "skill"
    assert len(candidate.digest) == 64
    assert (candidate.root / "review/report.json").is_file()
    assert (candidate.root / "skill/references/r001.png").read_bytes() == b"reference-raster"
    assert set(candidate.evidence) == {"e1"}
    assert set(json.loads((candidate.root / "mutation.json").read_text())) == {
        "mutation_id",
        "operation",
        "skill_id",
        "evidence_ids",
        "rationale",
    }
    with pytest.raises(FileExistsError):
        CandidatePackage.create(
            candidate.root,
            mutation=mutation,
            evidence=candidate.evidence,
            skill_root=candidate.skill_root,
            evolver_root=candidate.root / "evolver",
            review_root=candidate.root / "review",
        )

    candidate.evidence["e1"].raster_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="摘要不匹配"):
        CandidatePackage.open(candidate.root)


def test_candidate_seal_requires_accepted_review_and_detects_tampering(tmp_path: Path) -> None:
    mutation = _mutation()
    accepted = _candidate(tmp_path, mutation, name="sealed")
    (accepted.root / "skill/SKILL.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="内容摘要不匹配"):
        CandidatePackage.open(accepted.root)

    skill = tmp_path / "rejected-skill"
    _write_skill(skill, "nested", "rejected")
    m3 = tmp_path / "rejected-m3"
    m3.mkdir()
    raster = m3 / "context.png"
    raster.write_bytes(b"evidence")
    evidence = CandidateEvidence(
        str(m3),
        "finding-rejected",
        "context.png",
        hashlib.sha256(raster.read_bytes()).hexdigest(),
    )
    evolver, review = _audit_roots(tmp_path, "rejected")
    (review / "report.json").write_text('{"decision": "reject"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Reviewer accept"):
        CandidatePackage.create(
            tmp_path / "candidate-rejected",
            mutation=mutation,
            evidence={"e1": evidence},
            skill_root=skill / "nested",
            evolver_root=evolver,
            review_root=review,
        )
    assert not (tmp_path / "candidate-rejected").exists()


def test_candidate_package_enforces_operation_shape(tmp_path: Path) -> None:
    retire = _candidate(tmp_path, _mutation(MutationOperation.RETIRE, skill_id="skill-a"))
    assert retire.skill_root is None
    evolver, review = _audit_roots(tmp_path, "invalid-shape")

    with pytest.raises(ValueError, match="不能携带 skill"):
        CandidatePackage.create(
            tmp_path / "bad-retire",
            mutation=_mutation(MutationOperation.RETIRE, skill_id="skill-a"),
            evidence=retire.evidence,
            skill_root=tmp_path,
            evolver_root=evolver,
            review_root=review,
        )
    with pytest.raises(ValueError, match="精确对应"):
        CandidatePackage.create(
            tmp_path / "bad-evidence",
            mutation=_mutation(),
            evidence={},
            skill_root=tmp_path,
            evolver_root=evolver,
            review_root=review,
        )


def test_add_revise_retire_materialize_one_change_without_activation(tmp_path: Path) -> None:
    base_skills = tmp_path / "base-skills"
    _write_skill(base_skills, "skill-a", "base-a")
    _write_skill(base_skills, "skill-b", "base-b")
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), base_skills)
    parent_digest = store.read_manifest("g000").skill_digest

    add = _candidate(
        tmp_path,
        _mutation(),
        body="added",
        name="add",
        with_reference=True,
    )
    add_manifest = store.create_generation(
        "g001",
        parent_generation="g000",
        candidate=add,
        created_at="2026-09-01T01:00:00+00:00",
    )
    assert (store.generation_path("g001") / "skills/new-skill/SKILL.md").read_text() == "added"
    assert (
        store.generation_path("g001") / "skills/new-skill/references/r001.png"
    ).read_bytes() == b"reference-raster"
    assert store.active_generation() == "g000"
    assert store.read_manifest("g000").skill_digest == parent_digest
    assert add_manifest.mutation == add.mutation

    revise = _candidate(
        tmp_path,
        _mutation(MutationOperation.REVISE, skill_id="skill-a"),
        body="revised-a",
        name="revise",
    )
    store.create_generation("g002", parent_generation="g001", candidate=revise)
    assert (
        store.generation_path("g002") / "skills/skill-a/SKILL.md"
    ).read_text() == "revised-a"
    assert (store.generation_path("g002") / "skills/skill-b/SKILL.md").read_text() == "base-b"

    retire = _candidate(
        tmp_path,
        _mutation(MutationOperation.RETIRE, skill_id="skill-b"),
        name="retire",
    )
    store.create_generation("g003", parent_generation="g002", candidate=retire)
    assert not (store.generation_path("g003") / "skills/skill-b").exists()
    assert (store.generation_path("g002") / "skills/skill-b/SKILL.md").read_text() == "base-b"


def test_invalid_or_duplicate_generation_leaves_no_partial_snapshot(tmp_path: Path) -> None:
    base_skills = tmp_path / "base-skills"
    _write_skill(base_skills, "skill-a", "base")
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), base_skills)
    add = _candidate(tmp_path, _mutation(), name="valid")

    store.create_generation("g001", parent_generation="g000", candidate=add)
    manifest_bytes = (store.generation_path("g001") / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        store.create_generation("g001", parent_generation="g000", candidate=add)
    assert (store.generation_path("g001") / "manifest.json").read_bytes() == manifest_bytes

    duplicate = _candidate(
        tmp_path,
        _mutation(skill_id="skill-a"),
        name="duplicate",
    )
    with pytest.raises(ValueError, match="ADD 目标技能已存在"):
        store.create_generation("g002", parent_generation="g000", candidate=duplicate)
    assert not store.generation_path("g002").exists()
    assert not any(path.name.startswith(".g002.") for path in store.generations_dir.iterdir())
    with pytest.raises(ValueError, match="单段目录名"):
        store.read_manifest("../outside")


def test_rollback_only_changes_pointer_and_appends_event(tmp_path: Path) -> None:
    base_skills = tmp_path / "base-skills"
    _write_skill(base_skills, "skill-a", "base")
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), base_skills)
    candidate = _candidate(tmp_path, _mutation(), name="rollback")
    store.create_generation("g001", parent_generation="g000", candidate=candidate)
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


def test_ledger_and_gate_report_keep_facts_without_summary_duplication(tmp_path: Path) -> None:
    ledger = EvolutionLedger(tmp_path / "ledger.jsonl")
    ledger.append("first", generation_id="g000", timestamp="t1")
    before = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    ledger.append("second", generation_id="g001", timestamp="t2")
    assert (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").startswith(before)

    report = GateReport(
        mutation_id="mutation-add",
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index="runs/g000/day_index.json",
        candidate_day_index="runs/g001/day_index.json",
        decision=GateDecision.INCONCLUSIVE,
        reason="smoke 只有一个 recovery，尚未达到正式门槛。",
        pairs=(
            GatePair(
                task_id=1,
                seed=1,
                phase="primary",
                baseline=EpisodeOutcome.FAILURE,
                candidate=EpisodeOutcome.SUCCESS,
                baseline_consulted=False,
                candidate_consulted=True,
            ),
            GatePair(
                task_id=2,
                seed=1,
                phase="primary",
                baseline=EpisodeOutcome.SUCCESS,
                candidate=EpisodeOutcome.FAILURE,
                baseline_consulted=False,
                candidate_consulted=False,
            ),
            GatePair(
                task_id=3,
                seed=1,
                phase="primary",
                baseline=EpisodeOutcome.INFRASTRUCTURE,
                candidate=EpisodeOutcome.FAILURE,
                baseline_consulted=False,
                candidate_consulted=False,
            ),
        ),
    )
    document = report.to_dict()
    restored = GateReport.from_dict(json.loads(json.dumps(document)))

    assert restored == report
    assert restored.recoveries == 1
    assert restored.regressions == 1
    assert "summary" not in document
    assert "checks" not in document
    assert "notes" not in document
