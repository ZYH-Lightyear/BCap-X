"""用确定性小样本跑通 Skill Evolution 的完整工程生命周期。

这个入口不替代真实机器人评测。它用可审计的脚本化模型回复和 episode 结果，验证
M3 证据交接、M4 调查、独立审核、Visual Reference、generation、Gate、人工批准语义、
promotion 与 post-promotion audit 能在一次运行中正确连接。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.evolution.artifacts import read_json, write_json
from vaw.evolution.audit import audit_generation
from vaw.evolution.candidate import CandidatePackage
from vaw.evolution.domain import (
    EvolutionSpec,
    GatePolicy,
    RemainingFailure,
    SkillEffect,
    SkillEffectReport,
    SkillEffectReview,
)
from vaw.evolution.gate import evaluate_gate, write_gate_report
from vaw.evolution.skill_agent import run_skill_evolver
from vaw.evolution.store import GenerationStore


class _ScriptedProvider:
    """只服务工程 smoke；每次真实请求和回复仍由 M4 正常落盘。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        del messages, tools
        if not self._responses:
            raise RuntimeError("scripted smoke provider 没有剩余响应")
        return self._responses.pop(0)


def _write_m3_fixture(root: Path) -> Path:
    """构造最小 accepted finding；图片明确标注为 smoke，而不冒充真实 rollout。"""

    segment = root / "investigation/segments/segment_01.png"
    segment.parent.mkdir(parents=True)
    image = Image.new("RGB", (960, 540), "#e7eef8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 80, 800, 460), outline="#2f65d9", width=8)
    draw.rectangle((410, 210, 550, 420), fill="#d9a62e")
    draw.text((190, 105), "POLICY-VISIBLE ENGINEERING SMOKE", fill="#16233a")
    image.save(segment)
    write_json(
        root / "review.json",
        {
            "schema": "vaw-evidence-review-v1",
            "episode": {
                "suite": "libero_90",
                "task_id": 0,
                "seed": 1,
                "task": "place an object into an open container",
                "env_success": False,
            },
            "findings": [
                {
                    "span": [2, 4],
                    "evidence": "segments/segment_01.png",
                    "observation": "释放前物体投影仍压在容器边沿。",
                    "insight": "边沿干涉时应先分离，再依据开口做小幅水平调整。",
                    "review": {"accepted": True, "reason": "smoke evidence accepted"},
                }
            ],
        },
    )
    return root


def _evolver_provider() -> _ScriptedProvider:
    markdown = """---
name: Container Rim Recovery
description: 当被抓物接近开口容器边沿且释放关系不确定时使用。
---

# Container Rim Recovery

若物体压住边沿，先小幅上抬至与边沿分离，再依据当前多视角把物体移向开口内部；释放后重新检查真实结果。
"""
    decision = {
        "decision": "mutate",
        "operation": "add",
        "skill_id": "smoke-container-rim-recovery",
        "evidence_ids": ["e001"],
        "rationale": "一条已审核的视觉失败证据支持补充通用边沿恢复知识。",
        "skill_markdown": markdown,
        "references": [
            {
                "source_id": "e001",
                "state": "rim_interference",
                "view": "policy-visible segment",
                "when_to_use": "被抓物接近容器开口但释放关系不确定时",
                "visual_cue": "物体投影与容器边沿重叠",
            }
        ],
    }
    return _ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("inspect-1", "inspect_evidence", {"evidence_id": "e001"}),)
            ),
            ModelResponse(text=json.dumps(decision, ensure_ascii=False)),
        ]
    )


def _reviewer_provider() -> _ScriptedProvider:
    return _ScriptedProvider(
        [
            ModelResponse(
                text=json.dumps(
                    {
                        "decision": "accept",
                        "reason": "技能由可见边沿干涉支持，表达为可迁移的相对视觉原则。",
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )


def _day_index(
    root: Path,
    *,
    generation: str,
    outcomes: dict[tuple[int, int], bool],
    consulted: bool,
) -> Path:
    """写出与正式 DAY 相同的索引形状，便于 smoke 检查 Gate 和归因。"""

    root.mkdir(parents=True)
    tasks = sorted({task for task, _seed in outcomes})
    seeds = sorted({seed for _task, seed in outcomes})
    write_json(
        root / "run.json",
        {
            "schema": "vaw-day-run-v1",
            "experiment_id": "skill-evolver-engineering-smoke",
            "suite": "libero_90",
            "split": "gate",
            "tasks": tasks,
            "seeds": seeds,
            "generation_id": generation,
            "skill_digest": generation,
            "runner_args": {"model": "scripted-smoke", "temperature": 0},
        },
    )
    episodes: list[dict[str, Any]] = []
    for (task_id, seed), success in sorted(outcomes.items()):
        trace = root / "traces" / f"task{task_id}_s{seed}"
        trace.mkdir(parents=True)
        call = {
            "name": "consult_mmskill" if consulted else "detect_region",
            "arguments": (
                {"skill_id": "smoke-container-rim-recovery"}
                if consulted
                else {"query": "object"}
            ),
        }
        (trace / "steps.jsonl").write_text(
            json.dumps({"function_call": call}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        episodes.append(
            {
                "schema": "vaw-day-episode-v1",
                "episode_key": f"libero_90:t{task_id}:s{seed}",
                "suite": "libero_90",
                "task_id": task_id,
                "seed": seed,
                "generation_id": generation,
                "skill_digest": generation,
                "trace_dir": str(trace.resolve()),
                "status": "completed",
                "env_success": success,
                "turns": 12,
                "tokens": 1000,
                "wall_time_s": 1.0,
                "terminate_mode": "goal",
                "error": None,
            }
        )
    return write_json(
        root / "day_index.json",
        {"schema": "vaw-day-index-v1", "episodes": episodes, "summary": {}},
    )


def run_smoke(output_dir: str | Path) -> Path:
    """跑通一次可 promotion 的确定性闭环，并把每个边界产物保存在输出目录。"""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = EvolutionSpec(
        experiment_id="skill-evolver-engineering-smoke",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(2,),
        gate_tasks=(0, 1),
        reserve_tasks=(3,),
        heldout_suites=("libero_object_task", "libero_spatial_task", "libero_goal_task"),
        rollout_config={"seeds": [1, 2, 3], "runner_args": {"model": "scripted-smoke"}},
        gate_policy=GatePolicy(),
    )
    skill_root = Path(__file__).resolve().parents[1] / "mmskills"
    store = GenerationStore.initialize(output / "experiment", spec, skill_root)
    m3 = _write_m3_fixture(output / "m3")
    m4_result = run_skill_evolver(
        store=store,
        parent_generation="g000",
        mutation_id="m001",
        m3_outputs=[m3],
        output_dir=output / "m4",
        candidate_dir=store.root / "candidates/m001",
        evolver_provider=_evolver_provider(),
        reviewer_provider=_reviewer_provider(),
        run_metadata={
            "schema": "vaw-skill-evolver-run-v1",
            "evolver": {"model": "scripted-engineering-smoke"},
            "reviewer": {"model": "scripted-engineering-smoke"},
        },
    )
    candidate = store.root / "candidates/m001"
    store.create_generation("g001", parent_generation="g000", candidate=candidate)

    # 两个 recovery 检查收益；两个共同成功 pair 同时让成本护栏得到真实计算。
    baseline_outcomes = {(0, 1): False, (0, 2): True, (1, 1): False, (1, 2): True}
    candidate_outcomes = {(0, 1): True, (0, 2): True, (1, 1): True, (1, 2): True}
    baseline = _day_index(
        store.root / "runs/gate/baseline",
        generation="g000",
        outcomes=baseline_outcomes,
        consulted=False,
    )
    candidate_index = _day_index(
        store.root / "runs/gate/candidate",
        generation="g001",
        outcomes=candidate_outcomes,
        consulted=True,
    )
    mutation = store.read_manifest("g001").mutation
    assert mutation is not None
    effect_report = SkillEffectReport(
        mutation_id=mutation.mutation_id,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=str(baseline.resolve()),
        candidate_day_index=str(candidate_index.resolve()),
        reviews=tuple(
            SkillEffectReview(
                task_id=task_id,
                seed=seed,
                effect=SkillEffect.IMPROVED,
                remaining_failure=RemainingFailure.NONE,
                evidence_ids=(f"t{task_id}_s{seed}_e01",),
                reason="脚本化 smoke 的候选结果显示目标行为得到改善。",
            )
            for task_id, seed in sorted(baseline_outcomes)
        ),
    )
    report, metrics = evaluate_gate(
        mutation=mutation,
        baseline_generation="g000",
        candidate_generation="g001",
        baseline_day_index=baseline,
        candidate_day_index=candidate_index,
        policy=spec.gate_policy,
        expected_tasks=spec.gate_tasks,
        skill_effect_report=effect_report,
    )
    gate_json, gate_markdown = write_gate_report(store.root / "gates/m001", report, metrics)
    store.ledger.append(
        "generation_gated",
        generation_id="g001",
        payload={"report": str(gate_json), "decision": report.decision.value},
    )
    store.approve("g001", gate_json)
    store.promote("g001", gate_report=gate_json, candidate=candidate)

    audit_parent = _day_index(
        store.root / "runs/audit/parent",
        generation="g000",
        outcomes={(0, 1): True, (0, 3): True},
        consulted=False,
    )
    audit_current = _day_index(
        store.root / "runs/audit/current",
        generation="g001",
        outcomes={(0, 1): True, (0, 3): True},
        consulted=True,
    )
    audit = audit_generation(
        store=store,
        generation_id="g001",
        parent_day_index=audit_parent,
        current_day_index=audit_current,
        output_path=store.root / "audits/g001.json",
    )
    candidate_digest = CandidatePackage.open(candidate).digest
    report_path = output / "SMOKE_REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# Skill Evolution Engineering Smoke",
                "",
                "该目录使用脚本化模型响应与 episode outcome 验证生产接口的完整连通性；",
                "它不代表候选技能在真实 LIBERO 上产生了收益。",
                "",
                "```text",
                "M3 accepted evidence → Evolver → Reviewer → CandidatePackage",
                "→ inactive g001 → paired Gate → approval → promotion → audit",
                "```",
                "",
                "## Result",
                "",
                f"- active generation: `{store.active_generation()}`",
                f"- candidate digest: `{candidate_digest}`",
                f"- recoveries / regressions: `{metrics.recoveries} / {metrics.regressions}`",
                f"- attributed recoveries: `{metrics.attributed_recoveries}`",
                f"- success: `{metrics.baseline_successes} → {metrics.candidate_successes}`",
                f"- audit: `{read_json(audit)['status']}`",
                "",
                "## Trace Map",
                "",
                "- `m4/evolver/turns/`: 每轮 Agent 输入摘要、工具调用和响应。",
                "- `m4/review/`: 独立 Reviewer 的 request、response 和结论。",
                "- `experiment/candidates/m001/`: 密封候选与 evidence provenance。",
                "- `experiment/generations/`: 父代与 inactive/active generation 快照。",
                "- `experiment/runs/gate/`: 相同 task/seed 的 paired DAY fixture。",
                "- `experiment/gates/m001/`: 逐 episode Gate 事实和可读报告。",
                "- `experiment/ledger.jsonl`: 创建、批准、晋升和 audit 账本。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return write_json(
        output / "smoke_summary.json",
        {
            "schema": "vaw-skill-evolution-smoke-v1",
            "kind": "deterministic-engineering-smoke",
            "active_generation": store.active_generation(),
            "m4": str(m4_result.relative_to(output)),
            "candidate": str(candidate.relative_to(output)),
            "gate_json": str(gate_json.relative_to(output)),
            "gate_markdown": str(gate_markdown.relative_to(output)),
            "audit": str(audit.relative_to(output)),
            "ledger": str(store.ledger.path.relative_to(output)),
            "report": str(report_path.relative_to(output)),
            "note": "脚本化回复和结果只验证工程闭环，不代表真实任务收益。",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_smoke(args.output_dir)
    print(f"[skill-evolution-smoke] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_smoke"]
