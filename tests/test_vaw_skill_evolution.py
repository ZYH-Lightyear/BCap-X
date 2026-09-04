from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tests.test_vaw_context_packet import _workspace
from tests.test_vaw_evolution_m3 import _trace
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.context_runtime.packet import ContextCompiler
from vaw.evolution.audit import audit_generation
from vaw.evolution.domain import (
    EvolutionSpec,
    GateDecision,
    GatePolicy,
    MutationOperation,
    RemainingFailure,
    SkillEffect,
    SkillEffectReport,
    SkillEffectReview,
    SkillMutation,
)
from vaw.evolution.gate import evaluate_gate, write_gate_report
from vaw.evolution.effect_reviewer import SkillEffectReviewer
from vaw.evolution.skill_agent import (
    EvidencePool,
    EvolverDecision,
    MutationReviewer,
    ReferenceChoice,
    run_skill_evolver,
)
from vaw.evolution.store import GenerationStore
from vaw.mmskill import MMSkillLibrary


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append((messages, tools))
        return self.responses.pop(0)


def _skill(root: Path, skill_id: str = "base-skill") -> None:
    target = root / skill_id
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: 当需要基础指导时使用。\n---\n\n# {skill_id}\n\n保持视觉闭环。\n",
        encoding="utf-8",
    )


def _spec() -> EvolutionSpec:
    return EvolutionSpec(
        experiment_id="skill-evolution",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(2,),
        gate_tasks=(0, 1),
        reserve_tasks=(3,),
        heldout_suites=("libero_object_task",),
        rollout_config={"seeds": [1, 2, 3]},
        gate_policy=GatePolicy(),
    )


def _m3(root: Path) -> Path:
    segment = root / "investigation" / "segments" / "segment_01.png"
    segment.parent.mkdir(parents=True)
    Image.new("RGB", (320, 200), "#365314").save(segment)
    (root / "review.json").write_text(
        json.dumps(
            {
                "schema": "vaw-evidence-review-v1",
                "episode": {
                    "suite": "libero_90",
                    "task_id": 2,
                    "seed": 1,
                    "task": "place the object stably",
                    "env_success": False,
                },
                "findings": [
                    {
                        "evidence": "segments/segment_01.png",
                        "observation": "物体在释放后停在容器边沿。",
                        "insight": "释放前应依据当前视觉把物体投影移到开口内部。",
                        "review": {"accepted": True, "reason": "图中可见。"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _evolver_responses() -> list[ModelResponse]:
    markdown = """---
name: Stable Container Placement
description: 当已抓持物准备放入开口容器且边沿关系不确定时使用。
---

# Stable Container Placement

释放前从当前多视角确认被抓物投影位于开口内部；若压住边沿，先上抬分离，再做小幅水平调整。
"""
    return [
        ModelResponse(
            tool_calls=(
                ToolCall("c1", "inspect_evidence", {"evidence_id": "e001"}),
            )
        ),
        ModelResponse(
            text=json.dumps(
                {
                    "decision": "mutate",
                    "operation": "add",
                    "skill_id": "stable-container-placement",
                    "evidence_ids": ["e001"],
                    "rationale": "经审核的失败视觉证据显示边沿放置需要通用恢复指导。",
                    "skill_markdown": markdown,
                    "references": [
                        {
                            "source_id": "e001",
                            "state": "failure",
                            "view": "multi-action segment",
                            "when_to_use": "物体接近开口边沿且释放结果不确定时",
                            "visual_cue": "物体投影压在容器边沿而非开口内部",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ),
    ]


def _review_response(accepted: bool = True) -> ModelResponse:
    return ModelResponse(
        text=json.dumps(
            {
                "decision": "accept" if accepted else "reject",
                "reason": "候选与真实视觉失败证据一致，且表达为跨任务原则。",
            },
            ensure_ascii=False,
        )
    )


def _day_index(
    root: Path,
    *,
    generation: str,
    outcomes: dict[tuple[int, int], bool],
    consulted: bool,
    turns: int = 20,
    tokens: int = 1000,
    wall_time_s: float = 1.0,
) -> Path:
    root.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema": "vaw-day-run-v1",
                "experiment_id": "skill-evolution",
                "suite": "libero_90",
                "split": "gate",
                "tasks": sorted({task for task, _seed in outcomes}),
                "seeds": sorted({seed for _task, seed in outcomes}),
                "generation_id": generation,
                "skill_digest": generation,
                "runner_args": {"model": "fake"},
            }
        ),
        encoding="utf-8",
    )
    episodes = []
    for task_seed, success in sorted(outcomes.items()):
        task, seed = task_seed
        trace = root / "traces" / f"task{task}_s{seed}"
        trace.mkdir(parents=True)
        call = (
            {
                "name": "consult_mmskill",
                "arguments": {"skill_id": "stable-container-placement"},
            }
            if consulted
            else {"name": "detect_region", "arguments": {"query": "object"}}
        )
        (trace / "steps.jsonl").write_text(
            json.dumps({"function_call": call}) + "\n",
            encoding="utf-8",
        )
        episodes.append(
            {
                "schema": "vaw-day-episode-v1",
                "episode_key": f"libero_90:t{task}:s{seed}",
                "suite": "libero_90",
                "task_id": task,
                "seed": seed,
                "generation_id": generation,
                "skill_digest": generation,
                "trace_dir": str(trace.resolve()),
                "status": "completed",
                "env_success": success,
                "turns": turns,
                "tokens": tokens,
                "wall_time_s": wall_time_s,
                "terminate_mode": "goal",
                "error": None,
            }
        )
    path = root / "day_index.json"
    path.write_text(
        json.dumps({"schema": "vaw-day-index-v1", "episodes": episodes, "summary": {}}),
        encoding="utf-8",
    )
    return path


def _effect_report(
    mutation: SkillMutation,
    baseline: Path,
    candidate: Path,
    outcomes: dict[tuple[int, int], bool],
    *,
    effect: SkillEffect = SkillEffect.IMPROVED,
) -> SkillEffectReport:
    return SkillEffectReport(
        mutation_id=mutation.mutation_id,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=str(baseline.resolve()),
        candidate_day_index=str(candidate.resolve()),
        reviews=tuple(
            SkillEffectReview(
                task_id=task_id,
                seed=seed,
                effect=effect,
                remaining_failure=RemainingFailure.NONE,
                evidence_ids=(f"t{task_id}_s{seed}_e01",),
                reason="配对视觉证据支持该结论。",
            )
            for task_id, seed in sorted(outcomes)
        ),
    )


def test_evolver_reviewer_reference_gate_promotion_and_audit(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _skill(skills)
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), skills)
    m3 = _m3(tmp_path / "m3")
    evolver = FakeProvider(_evolver_responses())
    reviewer = FakeProvider([_review_response()])

    m4 = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m001",
        m3_outputs=[m3],
        output_dir=store.root / "m4" / "m001",
        candidate_dir=store.root / "candidates" / "m001",
        evolver_provider=evolver,
        reviewer_provider=reviewer,
        run_metadata={
            "schema": "vaw-skill-evolver-run-v1",
            "evolver": {"model": "fake-evolver"},
            "reviewer": {"model": "fake-reviewer"},
        },
    )

    result = json.loads(m4.read_text(encoding="utf-8"))
    assert result["status"] == "candidate"
    assert (store.root / "candidates/m001/evolver/turns/turn_01/request.json").is_file()
    assert (store.root / "candidates/m001/review/report.json").is_file()
    assert json.loads(
        (store.root / "candidates/m001/evolver/run.json").read_text()
    )["model"] == "fake-evolver"
    candidate = store.root / "candidates" / "m001"
    store.create_generation("g001", parent_generation="g000", candidate=candidate)
    assert store.active_generation() == "g000"

    library = MMSkillLibrary.from_root(store.generation_path("g001") / "skills")
    skill = library.get("stable-container-placement")
    assert len(skill.references) == 1
    assert skill.references[0].state == "failure"
    provenance = json.loads(
        (
            store.generation_path("g001")
            / "skills/stable-container-placement/references/provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["references"][0]["source_id"] == "e001"

    revise = EvolverDecision(
        decision="mutate",
        rationale="保留仍有用的历史参考并收敛正文。",
        operation=MutationOperation.REVISE,
        skill_id="stable-container-placement",
        evidence_ids=("e001",),
        skill_markdown=(
            "---\nname: Stable Container Placement\n"
            "description: 当容器边沿关系不确定时使用。\n---\n\n"
            "# Stable Container Placement\n\n先观察，再做最小调整。\n"
        ),
        references=(ReferenceChoice(source_id="current:r001"),),
    )
    revise_reviewer = FakeProvider([_review_response()])
    assert MutationReviewer(revise_reviewer).run(
        revise,
        EvidencePool.from_m3_outputs([m3]),
        library,
        tmp_path / "review-retained-reference",
    )
    reviewer_content = revise_reviewer.requests[0][0][-1]["content"]
    assert sum(part.get("type") == "image_url" for part in reviewer_content) == 2
    packet = ContextCompiler().compile(_workspace(), active_mmskill=skill)
    assert packet.skill_reference is not None
    assert packet.skill_reference.skill_id == skill.skill_id
    assert "REFERENCE" not in json.dumps(packet.manifest())

    baseline = _day_index(
        tmp_path / "baseline",
        generation="g000",
        outcomes={(0, 1): False, (0, 2): False, (1, 1): False, (1, 2): False},
        consulted=False,
    )
    candidate_index = _day_index(
        tmp_path / "candidate-day",
        generation="g001",
        outcomes={(0, 1): True, (0, 2): False, (1, 1): True, (1, 2): False},
        consulted=True,
    )
    mutation = store.read_manifest("g001").mutation
    assert mutation is not None
    report, metrics = evaluate_gate(
        mutation=mutation,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=baseline,
        candidate_day_index=candidate_index,
        policy=store.read_spec().gate_policy,
        expected_tasks=(0, 1),
        skill_effect_report=_effect_report(
            mutation,
            baseline,
            candidate_index,
            {(0, 1): True, (0, 2): True, (1, 1): True, (1, 2): True},
        ),
    )
    assert report.decision is GateDecision.PASSED
    assert metrics.recoveries == 2
    report_path, markdown = write_gate_report(store.root / "gates" / "m001", report, metrics)
    assert "| 0 | 1 | primary |" in markdown.read_text(encoding="utf-8")

    store.approve("g001", report_path)
    store.promote("g001", gate_report=report_path, candidate=candidate)
    assert store.active_generation() == "g001"

    parent_audit = _day_index(
        tmp_path / "parent-audit",
        generation="g000",
        outcomes={(0, 1): True, (0, 3): True},
        consulted=False,
    )
    current_audit = _day_index(
        tmp_path / "current-audit",
        generation="g001",
        outcomes={(0, 1): False, (0, 3): False},
        consulted=True,
    )
    audit = audit_generation(
        store=store,
        generation_id="g001",
        parent_day_index=parent_audit,
        current_day_index=current_audit,
        output_path=store.root / "audits" / "g001.json",
    )
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "confirmed_degradation"
    assert store.active_generation() == "g000"


def test_evolver_no_change_and_reviewer_reject_do_not_create_candidate(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _skill(skills)
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), skills)
    m3 = _m3(tmp_path / "m3")
    no_change = FakeProvider(
        [ModelResponse(text='{"decision":"no_change","rationale":"证据不足。"}')]
    )
    result = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m-none",
        m3_outputs=[m3],
        output_dir=tmp_path / "m4-none",
        candidate_dir=tmp_path / "candidate-none",
        evolver_provider=no_change,
        reviewer_provider=FakeProvider([]),
    )
    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "no_change"
    assert not (tmp_path / "candidate-none").exists()

    reviewer = FakeProvider([_review_response(False)])
    rejected = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m-reject",
        m3_outputs=[m3],
        output_dir=tmp_path / "m4-reject",
        candidate_dir=tmp_path / "candidate-reject",
        evolver_provider=FakeProvider(_evolver_responses()),
        reviewer_provider=reviewer,
    )
    assert json.loads(rejected.read_text(encoding="utf-8"))["status"] == "rejected"
    assert not (tmp_path / "candidate-reject").exists()
    reviewer_content = reviewer.requests[0][0][1]["content"]
    review_payload = json.loads(reviewer_content[0]["text"])
    assert review_payload["skill_index"] == [
        {
            "skill_id": "base-skill",
            "name": "base-skill",
            "description": "当需要基础指导时使用。",
        }
    ]


def test_evidence_pool_ignores_rejected_findings(tmp_path: Path) -> None:
    root = _m3(tmp_path / "m3")
    payload = json.loads((root / "review.json").read_text(encoding="utf-8"))
    payload["findings"].append(
        {
            "evidence": "segments/segment_01.png",
            "observation": "unsupported",
            "insight": "unsupported",
            "review": {"accepted": False, "reason": "not visible"},
        }
    )
    (root / "review.json").write_text(json.dumps(payload), encoding="utf-8")

    pool = EvidencePool.from_m3_outputs([root])

    assert [item.evidence_id for item in pool.items] == ["e001"]


def test_evidence_pool_accepts_cycle_m3_result_path(tmp_path: Path) -> None:
    root = _m3(tmp_path / "m3")
    result = root / "m3.json"
    result.write_text(
        json.dumps(
            {
                "schema": "vaw-m3-run-result-v1",
                "investigation": "investigation/findings.json",
                "review": "review.json",
            }
        ),
        encoding="utf-8",
    )

    pool = EvidencePool.from_m3_outputs([result])

    assert [item.evidence_id for item in pool.items] == ["e001"]
    assert pool.items[0].m3_output == root.resolve()


def test_m4_without_accepted_evidence_is_explicit_no_change(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _skill(skills)
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), skills)
    root = _m3(tmp_path / "m3")
    payload = json.loads((root / "review.json").read_text(encoding="utf-8"))
    payload["findings"][0]["review"]["accepted"] = False
    (root / "review.json").write_text(json.dumps(payload), encoding="utf-8")

    result = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m-empty",
        m3_outputs=[root],
        output_dir=tmp_path / "m4-empty",
        candidate_dir=tmp_path / "candidate-empty",
        evolver_provider=FakeProvider([]),
        reviewer_provider=FakeProvider([]),
    )

    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["status"] == "no_change"
    assert "Reviewer" in document["decision"]["rationale"]
    assert not (tmp_path / "candidate-empty").exists()


def test_repeated_inspection_consumes_budget_and_forces_final_decision(
    tmp_path: Path,
) -> None:
    skills = tmp_path / "skills"
    _skill(skills)
    store = GenerationStore.initialize(tmp_path / "experiment", _spec(), skills)
    m3 = _m3(tmp_path / "m3")
    repeated = ModelResponse(
        tool_calls=(ToolCall("c1", "inspect_evidence", {"evidence_id": "e001"}),)
    )
    evolver = FakeProvider(
        [
            repeated,
            repeated,
            ModelResponse(
                text=(
                    '{"decision":"no_change","rationale":'
                    '"检查预算内没有获得足以支持通用技能变化的新证据。"}'
                )
            ),
        ]
    )

    result = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m-repeat",
        m3_outputs=[m3],
        output_dir=tmp_path / "m4-repeat",
        candidate_dir=tmp_path / "candidate-repeat",
        evolver_provider=evolver,
        reviewer_provider=FakeProvider([]),
        max_inspections=2,
    )

    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "no_change"
    final_request = json.loads(
        (
            tmp_path
            / "m4-repeat"
            / "evolver"
            / "turns"
            / "turn_03"
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    assert final_request["tools"] == []


def test_gate_rejects_wrong_generation_and_compares_cost_on_common_success(
    tmp_path: Path,
) -> None:
    baseline = _day_index(
        tmp_path / "baseline",
        generation="g000",
        outcomes={(0, 1): False, (0, 2): False, (1, 1): True, (1, 2): True},
        consulted=False,
        turns=10,
        tokens=100,
        wall_time_s=2.0,
    )
    candidate = _day_index(
        tmp_path / "candidate",
        generation="g001",
        outcomes={(0, 1): True, (0, 2): True, (1, 1): True, (1, 2): True},
        consulted=True,
        turns=20,
        tokens=200,
        wall_time_s=3.0,
    )
    skill_mutation = SkillMutation(
        mutation_id="m001",
        operation=MutationOperation.ADD,
        skill_id="stable-container-placement",
        evidence_ids=("e001",),
        rationale="测试共同成功 episode 的成本门控。",
    )

    report, metrics = evaluate_gate(
        mutation=skill_mutation,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=baseline,
        candidate_day_index=candidate,
        policy=GatePolicy(max_turn_ratio=1.5, max_token_ratio=1.5),
        expected_tasks=(0, 1),
        skill_effect_report=_effect_report(
            skill_mutation,
            baseline,
            candidate,
            {(0, 1): True, (0, 2): True, (1, 1): True, (1, 2): True},
        ),
    )
    assert report.decision is GateDecision.REJECTED
    assert metrics.turn_ratio == 2.0
    assert metrics.token_ratio == 2.0
    assert metrics.wall_time_ratio == 1.5

    with pytest.raises(ValueError, match="baseline DAY run"):
        evaluate_gate(
            mutation=skill_mutation,
            baseline_generation="wrong",
            candidate_generation="g001",
            baseline_day_index=baseline,
            candidate_day_index=candidate,
            policy=GatePolicy(),
        )


def test_gate_keeps_common_pairs_when_one_day_index_is_incomplete(tmp_path: Path) -> None:
    baseline = _day_index(
        tmp_path / "baseline-incomplete",
        generation="g000",
        outcomes={(0, 1): False, (1, 1): False},
        consulted=False,
    )
    candidate = _day_index(
        tmp_path / "candidate-incomplete",
        generation="g001",
        outcomes={(0, 1): False, (1, 1): False},
        consulted=True,
    )
    mutation = SkillMutation(
        mutation_id="m-incomplete",
        operation=MutationOperation.ADD,
        skill_id="stable-container-placement",
        evidence_ids=("e001",),
        rationale="验证中断的配对评测仍能保存已有事实。",
    )
    effects = _effect_report(
        mutation,
        baseline,
        candidate,
        {(0, 1): True},
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["episodes"] = payload["episodes"][:1]
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    report, _ = evaluate_gate(
        mutation=mutation,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=baseline,
        candidate_day_index=candidate,
        policy=GatePolicy(primary_seeds=(1,), confirmation_seeds=(2,)),
        skill_effect_report=effects,
    )

    assert report.decision is GateDecision.INCONCLUSIVE
    assert [(pair.task_id, pair.seed) for pair in report.pairs] == [(0, 1)]
    assert [review.key for review in report.skill_effects] == [(0, 1)]


def test_skill_effect_reviewer_inspects_real_pair_before_judging(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline-trace"
    candidate_root = tmp_path / "candidate-trace"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline, _ = _trace(baseline_root)
    candidate, _ = _trace(candidate_root)
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "inspect-pair-1",
                        "inspect_pair",
                        {
                            "baseline_start": 1,
                            "baseline_end": 2,
                            "candidate_start": 1,
                            "candidate_end": 2,
                            "question": "候选轨迹是否保持了目标关系？",
                        },
                    ),
                )
            ),
            ModelResponse(
                text=json.dumps(
                    {
                        "effect": "improved",
                        "remaining_failure": "other",
                        "evidence_ids": ["t0_s1_e01"],
                        "reason": "candidate 在目标接触问题上更稳定，但终局仍有另一问题。",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )

    review = SkillEffectReviewer(provider, max_inspections=2).run(
        task_id=0,
        seed=1,
        baseline_trace=baseline,
        candidate_trace=candidate,
        baseline_outcome="failure",
        candidate_outcome="failure",
        mutation={"operation": "revise", "skill_id": "base-skill"},
        previous_skill="old",
        candidate_skill="new",
        baseline_consulted=True,
        candidate_consulted=True,
        output_dir=tmp_path / "effect-review",
    )

    assert review.effect is SkillEffect.IMPROVED
    assert review.remaining_failure is RemainingFailure.OTHER
    evidence = tmp_path / "effect-review/evidence/t0_s1_e01.png"
    assert Image.open(evidence).size == (2400, 1010)
    assert provider.requests[0][1][0]["function"]["name"] == "inspect_pair"
    assert provider.requests[1][1] is not None
    assert len(provider.requests[1][0][-1]["content"]) >= 3


def test_skill_effect_reviewer_forces_final_after_duplicate_inspection(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline-trace"
    candidate_root = tmp_path / "candidate-trace"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline, _ = _trace(baseline_root)
    candidate, _ = _trace(candidate_root)
    inspect = ToolCall(
        "inspect-pair",
        "inspect_pair",
        {
            "baseline_start": 1,
            "baseline_end": 2,
            "candidate_start": 1,
            "candidate_end": 2,
            "question": "候选轨迹是否改善目标接触？",
        },
    )
    provider = FakeProvider(
        [
            ModelResponse(tool_calls=(inspect,)),
            ModelResponse(tool_calls=(inspect,)),
            ModelResponse(
                text=json.dumps(
                    {
                        "effect": "unchanged",
                        "remaining_failure": "related",
                        "evidence_ids": ["t0_s1_e01"],
                        "reason": "两侧都没有形成目标接触。",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )

    review = SkillEffectReviewer(provider, max_inspections=4).run(
        task_id=0,
        seed=1,
        baseline_trace=baseline,
        candidate_trace=candidate,
        baseline_outcome="failure",
        candidate_outcome="failure",
        mutation={"operation": "revise", "skill_id": "base-skill"},
        previous_skill="old",
        candidate_skill="new",
        baseline_consulted=True,
        candidate_consulted=True,
        output_dir=tmp_path / "duplicate-effect-review",
    )

    assert review.effect is SkillEffect.UNCHANGED
    assert len(provider.requests) == 3
    assert provider.requests[-1][1] is None
