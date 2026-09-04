"""VAW Skill Evolution 的显式、可审计命令入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vaw.agents.providers.base import ModelProvider
from vaw.evolution.artifacts import read_json
from vaw.evolution.audit import audit_generation
from vaw.evolution.candidate import CandidatePackage
from vaw.evolution.domain import EvolutionSpec
from vaw.evolution.effect_reviewer import run_skill_effect_reviews
from vaw.evolution.gate import evaluate_gate, write_gate_report
from vaw.evolution.skill_agent import run_skill_evolver
from vaw.evolution.store import GenerationStore


def _provider(args: argparse.Namespace, model: str) -> ModelProvider:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider

    provider: ModelProvider = OpenAIProvider(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    return TextProtocolProvider(provider) if args.protocol == "text" else provider


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--review-model")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=300.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="创建 schema-v2 evolution experiment")
    init.add_argument("--experiment-root", type=Path, required=True)
    init.add_argument("--skill-root", type=Path, required=True)
    init.add_argument("--experiment-id", required=True)
    init.add_argument("--base-generation", default="g000")
    init.add_argument("--suite", default="libero_90")
    init.add_argument("--evolve-tasks", required=True)
    init.add_argument("--gate-tasks", required=True)
    init.add_argument("--reserve-tasks", default="")
    init.add_argument(
        "--heldout-suites",
        default="libero_object_task,libero_spatial_task,libero_goal_task",
    )
    init.add_argument("--rollout-config", type=Path, required=True)

    propose = commands.add_parser("propose", help="运行 Evolver 与独立 Reviewer")
    propose.add_argument("--experiment-root", type=Path, required=True)
    propose.add_argument("--parent-generation")
    propose.add_argument("--mutation-id", required=True)
    propose.add_argument("--m3-output", type=Path, action="append", required=True)
    propose.add_argument("--output-dir", type=Path, required=True)
    propose.add_argument("--candidate-dir", type=Path)
    propose.add_argument("--max-inspections", type=int, default=6)
    _add_provider_args(propose)

    materialize = commands.add_parser("materialize", help="创建 inactive generation")
    materialize.add_argument("--experiment-root", type=Path, required=True)
    materialize.add_argument("--candidate", type=Path, required=True)
    materialize.add_argument("--generation", required=True)
    materialize.add_argument("--parent-generation")

    effects = commands.add_parser(
        "review-effects",
        help="独立比较 baseline/candidate 的局部技能效果",
    )
    effects.add_argument("--experiment-root", type=Path, required=True)
    effects.add_argument("--generation", required=True)
    effects.add_argument("--baseline-index", type=Path, required=True)
    effects.add_argument("--candidate-index", type=Path, required=True)
    effects.add_argument("--output-dir", type=Path, required=True)
    effects.add_argument("--max-inspections", type=int, default=4)
    _add_provider_args(effects)

    gate = commands.add_parser("gate", help="读取两份 DAY index 生成配对 Gate")
    gate.add_argument("--experiment-root", type=Path, required=True)
    gate.add_argument("--generation", required=True)
    gate.add_argument("--baseline-index", type=Path, required=True)
    gate.add_argument("--candidate-index", type=Path, required=True)
    gate.add_argument("--effect-report", type=Path, required=True)
    gate.add_argument("--output-dir", type=Path, required=True)
    gate.add_argument("--smoke", action="store_true")

    approve = commands.add_parser("approve", help="记录人工批准")
    approve.add_argument("--experiment-root", type=Path, required=True)
    approve.add_argument("--generation", required=True)
    approve.add_argument("--gate-report", type=Path, required=True)

    promote = commands.add_parser("promote", help="核验后激活 generation")
    promote.add_argument("--experiment-root", type=Path, required=True)
    promote.add_argument("--generation", required=True)
    promote.add_argument("--gate-report", type=Path, required=True)
    promote.add_argument("--candidate", type=Path, required=True)

    audit = commands.add_parser("audit", help="执行 post-promotion paired audit")
    audit.add_argument("--experiment-root", type=Path, required=True)
    audit.add_argument("--generation", required=True)
    audit.add_argument("--parent-index", type=Path, required=True)
    audit.add_argument("--current-index", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    rollback = commands.add_parser("rollback", help="切回一个历史 generation")
    rollback.add_argument("--experiment-root", type=Path, required=True)
    rollback.add_argument("--generation", required=True)

    status = commands.add_parser("status", help="显示 generation 与 ledger 状态")
    status.add_argument("--experiment-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        rollout = read_json(args.rollout_config)
        spec = EvolutionSpec(
            experiment_id=args.experiment_id,
            base_generation=args.base_generation,
            evolve_suite=args.suite,
            evolve_tasks=_integers(args.evolve_tasks),
            gate_tasks=_integers(args.gate_tasks),
            reserve_tasks=_integers(args.reserve_tasks),
            heldout_suites=_strings(args.heldout_suites),
            rollout_config=rollout,
        )
        store = GenerationStore.initialize(args.experiment_root, spec, args.skill_root)
        print(f"[evolution] initialized {store.root} active={store.active_generation()}")
        return 0

    store = GenerationStore.open(args.experiment_root)
    if args.command == "propose":
        parent = args.parent_generation or store.active_generation()
        candidate_dir = args.candidate_dir or store.root / "candidates" / args.mutation_id
        result = run_skill_evolver(
            store=store,
            parent_generation=parent,
            mutation_id=args.mutation_id,
            m3_outputs=args.m3_output,
            output_dir=args.output_dir,
            candidate_dir=candidate_dir,
            evolver_provider=_provider(args, args.model),
            reviewer_provider=_provider(args, args.review_model or args.model),
            max_inspections=args.max_inspections,
            run_metadata={
                "schema": "vaw-skill-evolver-run-v1",
                "evolver": {
                    "model": args.model,
                    "server_url": args.server_url,
                    "protocol": args.protocol,
                    "max_tokens": args.max_tokens,
                    "timeout_s": args.timeout_s,
                    "max_inspections": args.max_inspections,
                },
                "reviewer": {
                    "model": args.review_model or args.model,
                    "server_url": args.server_url,
                    "protocol": args.protocol,
                    "max_tokens": args.max_tokens,
                    "timeout_s": args.timeout_s,
                },
            },
        )
        print(f"[evolution] M4 result={result}")
        return 0
    if args.command == "materialize":
        parent = args.parent_generation or store.active_generation()
        manifest = store.create_generation(
            args.generation,
            parent_generation=parent,
            candidate=CandidatePackage.open(args.candidate),
        )
        print(f"[evolution] inactive generation={manifest.generation_id}")
        return 0
    if args.command == "review-effects":
        result = run_skill_effect_reviews(
            store=store,
            candidate_generation=args.generation,
            baseline_day_index=args.baseline_index,
            candidate_day_index=args.candidate_index,
            output_dir=args.output_dir,
            provider=_provider(args, args.model),
            max_inspections=args.max_inspections,
        )
        print(f"[evolution] skill-effect report={result}")
        return 0
    if args.command == "gate":
        manifest = store.read_manifest(args.generation)
        if manifest.mutation is None or manifest.parent_generation is None:
            raise SystemExit("Gate 只能评估 candidate generation")
        report, metrics = evaluate_gate(
            mutation=manifest.mutation,
            baseline_generation=manifest.parent_generation,
            candidate_generation=args.generation,
            baseline_day_index=args.baseline_index,
            candidate_day_index=args.candidate_index,
            policy=store.read_spec().gate_policy,
            expected_tasks=None if args.smoke else store.read_spec().gate_tasks,
            skill_effect_report=args.effect_report,
        )
        json_path, _markdown = write_gate_report(args.output_dir, report, metrics)
        store.ledger.append(
            "generation_gated",
            generation_id=args.generation,
            payload={"report": str(json_path), "decision": report.decision.value},
        )
        print(f"[evolution] gate={report.decision.value} report={json_path}")
        return 0
    if args.command == "approve":
        store.approve(args.generation, args.gate_report)
        print(f"[evolution] approved generation={args.generation}")
        return 0
    if args.command == "promote":
        store.promote(
            args.generation,
            gate_report=args.gate_report,
            candidate=args.candidate,
        )
        print(f"[evolution] promoted active={store.active_generation()}")
        return 0
    if args.command == "audit":
        result = audit_generation(
            store=store,
            generation_id=args.generation,
            parent_day_index=args.parent_index,
            current_day_index=args.current_index,
            output_path=args.output,
        )
        print(f"[evolution] audit={result} active={store.active_generation()}")
        return 0
    if args.command == "rollback":
        store.rollback(args.generation)
        print(f"[evolution] active={store.active_generation()}")
        return 0
    if args.command == "status":
        manifests = [
            store.read_manifest(path.name).to_dict()
            for path in sorted(store.generations_dir.iterdir())
            if path.is_dir()
        ]
        print(
            json.dumps(
                {
                    "active_generation": store.active_generation(),
                    "generations": manifests,
                    "ledger": list(store.ledger.read()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
