"""RoboMEx CLI —— 驱动 Reactive Planner 在 LIBERO-pro 上跑闭环。

只有一个子命令 ``plan``:开一个 LIBERO-pro 任务,拿真实画面喂 planner,把它每一步
决定的 ActionIntent 打出来。执行层尚未接通,所以每个意图都只会得到「没有执行
成功」的反馈,场景不会改变。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from robomex.logging import configure_logging


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _plan(args: argparse.Namespace) -> int:
    from robomex.env import LiberoEnv
    from robomex.planner import ReactivePlanner
    from robomex.policy import LLMPolicy
    from robomex.trace import PlannerTracer

    output_dir = Path(args.output_root) / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=output_dir / "run.log")

    env = LiberoEnv(
        suite_name=args.suite,
        task_id=args.task_id,
        image_dir=output_dir / "observations",
        camera=args.camera,
        seed=args.seed,
    )
    observation = env.reset()

    # LIBERO 自带的任务语言是最权威的任务描述;--task 只在想覆盖它时才给。
    task = args.task or env.task_prompt
    if not task:
        print("robomex: 该任务没有自带任务描述,请用 --task 显式给出", file=sys.stderr)
        return 2
    print(f"task: {task}", file=sys.stderr)

    policy = LLMPolicy(
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
    )
    planner = ReactivePlanner(policy, max_intents=args.max_intents)
    tracer = PlannerTracer(output_dir)

    if not args.single:
        # 执行层缺失时,画面不会变,planner 会反复给出同一个动作意图。这是闭环
        # 的正确表现而非 bug,但不预告一声很容易被当成模型卡死。
        print(
            "提示:执行层尚未接通,场景不会因动作意图而改变,"
            "连续多步很可能反复得到同一个意图 —— 这正是闭环应有的表现。",
            file=sys.stderr,
        )

    history: list[tuple] = []
    steps_taken = 0
    try:
        while steps_taken < args.max_intents:
            step = planner.step(task=task, observation=observation, history=tuple(history))
            tracer.record(steps_taken, planner, step, observation)
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
                # 纯展示用的行号,方便对着 trace 的 step-N 目录看;不是契约字段。
                "step": steps_taken - 1,
                "instruction": step.intent.instruction,
                "expected_effect": step.intent.expected_effect,
                "thought": step.thought,
            }, ensure_ascii=False, indent=2))

            if args.single:
                break

            feedback, observation = env.apply(step.intent)
            history.append((step.intent, feedback))
    finally:
        env.close()

    print(f"\nTrace written to: {output_dir}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robomex",
        description="RoboMEx Reactive Planner",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser(
        "plan",
        help="在 LIBERO-pro 任务上驱动 Reactive Planner 决定 ActionIntent",
    )
    plan.add_argument(
        "--task",
        default=None,
        help="任务描述;默认用 LIBERO 任务自带的 task language",
    )
    plan.add_argument("--suite", default="libero_object_swap", help="LIBERO 套件名")
    plan.add_argument("--task-id", type=int, default=0, help="套件内任务序号")
    plan.add_argument("--camera", default="agentview", help="观测相机名")
    plan.add_argument("--seed", type=int, default=None, help="trial 序号,决定初始摆放")
    plan.add_argument("--model", default="vapi/claude-opus-4-8")
    plan.add_argument("--server-url", default="http://localhost:8110/chat/completions")
    plan.add_argument("--api-key", default=None)
    plan.add_argument("--output-root", default="outputs/robomex_planner")
    plan.add_argument("--max-intents", type=int, default=20)
    plan.add_argument(
        "--single",
        action="store_true",
        help="只决定下一个 ActionIntent 就退出",
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
