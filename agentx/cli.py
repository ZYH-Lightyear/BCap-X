"""agentx 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentx.core import RunConfig, TurnSummary


def _run(args: argparse.Namespace) -> int:
    from agentx.agent import CodingAgent
    from agentx.providers import OpenAIProvider

    def on_turn(index: int, summary: TurnSummary) -> None:
        # 进度打到 stderr,让 stdout 只有最终答复,便于管道使用。
        for record in summary.tool_records:
            mark = "!" if record.result.is_error else "·"
            print(
                f"  [{index}] {mark} {record.call.name} ({record.duration_ms:.0f}ms)",
                file=sys.stderr,
            )
        if summary.text.strip() and not summary.tool_records:
            print(f"  [{index}] {summary.text.strip()[:200]}", file=sys.stderr)

    agent = CodingAgent(
        provider=OpenAIProvider(
            model=args.model,
            server_url=args.server_url,
            api_key=args.api_key,
        ),
        root=args.root,
        protocol=args.protocol,
        skill_roots=args.skills,
        run_config=RunConfig(max_turns=args.max_turns, on_turn=on_turn),
    )

    if agent.skills:
        names = ", ".join(skill.name for skill in agent.skills)
        print(f"skills: {names}", file=sys.stderr)
    print(f"task: {args.task}  [protocol={args.protocol}]", file=sys.stderr)
    result = agent.run(args.task)

    print(result.text or "(模型没有给出最终答复)")
    print(
        f"\n[{result.terminate_mode.value}] {result.turns} 轮,"
        f"{len(result.tool_records)} 次工具调用,"
        f"{result.usage.get('total_tokens', 0)} tokens"
        + (f" — {result.detail}" if result.detail else ""),
        file=sys.stderr,
    )
    return 0 if result.succeeded else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentx",
        description="通用多模态 Coding Agent",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="执行一个任务")
    run.add_argument("task", help="任务描述")
    run.add_argument("--root", default=".", help="工作目录(工具被约束在此目录内)")
    run.add_argument("--model", default="vapi/claude-opus-4-8")
    run.add_argument("--server-url", default="http://localhost:8110/chat/completions")
    run.add_argument("--api-key", default=None)
    run.add_argument("--max-turns", type=int, default=40)
    run.add_argument(
        "--skills",
        action="append",
        default=None,
        metavar="DIR",
        help="技能根目录,可重复。同名技能取先出现的那个",
    )
    run.add_argument(
        "--protocol",
        choices=("native", "text"),
        default="native",
        help="native=原生 function calling;text=<tool_call> 文本协议(端点不支持工具时用)",
    )
    run.set_defaults(handler=_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"agentx: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


__all__ = ["build_parser", "main"]
