"""RoboMEx CLI — Phase 1: planner-only entry point."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from robomex.core.logging import configure_logging


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _plan(args: argparse.Namespace) -> int:
    """Drive the Reactive Planner and print each ActionIntent it decides on."""
    from robomex.contracts import IntentFeedback
    from robomex.planner import ReactivePlanner
    from robomex.trace import PlannerTracer

    output_dir = Path(args.output_root) / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=output_dir / "run.log")

    from robomex.core.coder.policy import LLMCodePolicy

    policy = LLMCodePolicy(
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.3,
        max_tokens=4096,
    )
    planner = ReactivePlanner(policy, max_intents=args.max_intents)
    tracer = PlannerTracer(output_dir)

    if not args.single:
        # 闭环的另一半没接通时,连续多步只能演示开环序列。把这一点说在前面,
        # 免得把这种输出误当成目标范式的成果。
        print(
            "警告:观测通道尚未接通,连续多步运行只能演示开环序列,"
            "不代表目标范式。单步请加 --single。",
            file=sys.stderr,
        )

    history: list[tuple] = []
    steps_taken = 0
    while steps_taken < args.max_intents:
        step = planner.step(task=args.task, history=tuple(history))
        tracer.record(steps_taken, planner, step)
        steps_taken += 1

        if step.done:
            print(json.dumps({
                "done": True,
                "reason": step.reason,
                "thought": step.thought,
                "steps_taken": steps_taken,
            }, ensure_ascii=False, indent=2))
            break

        assert step.intent is not None
        print(json.dumps({
            "intent_id": step.intent.intent_id,
            "instruction": step.intent.instruction,
            "expected_effect": step.intent.expected_effect,
            "thought": step.thought,
        }, ensure_ascii=False, indent=2))

        if args.single:
            break

        feedback = IntentFeedback(
            intent_id=step.intent.intent_id,
            status="not_executed",
            summary="执行层尚未接通,该动作意图未被真实执行",
        )
        history.append((step.intent, feedback))

    print(f"\nTrace written to: {output_dir}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robomex",
        description="RoboMEx Reactive Planner (Phase 1)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan",
        help="decide ActionIntents for a task using the Reactive Planner",
    )
    plan.add_argument("--task", required=True, help="task description in natural language")
    plan.add_argument("--model", default="vapi/claude-opus-4-8")
    plan.add_argument("--server-url", default="http://localhost:8110/chat/completions")
    plan.add_argument("--api-key", default=None)
    plan.add_argument("--output-root", default="outputs/robomex_planner")
    plan.add_argument("--max-intents", type=int, default=20)
    plan.add_argument(
        "--single",
        action="store_true",
        help="decide only the next ActionIntent and exit (honest Phase 1 mode)",
    )
    plan.set_defaults(handler=_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"robomex: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


__all__ = ["build_parser", "main"]
