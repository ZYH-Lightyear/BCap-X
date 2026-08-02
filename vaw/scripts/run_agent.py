"""Run one real VLM-controlled VAW episode on LIBERO-PRO.

The model always receives ``workspace PNG + state_summary`` and always emits
the existing structured workspace operations.  ``--renderer`` changes only the
visual observation format.

Example:

    python -m vaw.scripts.run_agent \
      --suite libero_object_swap --task-id 0 \
      --model vapi/qwen3.5-plus --protocol text \
      --renderer web --compare-renderers
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from vaw.scripts.scripted_pick import preflight_services


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="vapi/qwen3.5-plus")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="text")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-time-s", type=float, default=1800.0)
    parser.add_argument("--max-physical-ops", type=int, default=30)
    parser.add_argument("--canvas-window-k", type=int, default=3)
    parser.add_argument("--renderer", choices=("pil", "web"), default="pil")
    parser.add_argument("--compare-renderers", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--trace-dir", type=pathlib.Path, default=None)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def main() -> int:
    args = _parse_args()
    preflight_services()

    # Heavy simulation imports stay below --help and the service preflight.
    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider
    from vaw.agents.runtime import RunConfig, run_episode
    from vaw.renderers import build_renderer

    env = FrankaLiberoTask(
        suite_name=args.suite,
        task_id=args.task_id,
        privileged=False,
        seed=args.seed,
    )
    instruction = str(env.handle.task_language)
    api = FrankaLiberoApiReduced(env)
    trace_dir = args.trace_dir or (
        pathlib.Path(__file__).resolve().parent.parent
        / "out"
        / "agent_runs"
        / f"{_safe_name(args.model)}_{args.suite}_t{args.task_id}_s{args.seed}_{args.renderer}"
    )
    renderer = build_renderer(
        args.renderer,
        headed=args.headed,
        compare_dir=(trace_dir / "_render_compare") if args.compare_renderers else None,
    )
    provider = OpenAIProvider(
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.protocol == "text":
        provider = TextProtocolProvider(provider)

    def on_turn(turn, record) -> None:
        flag = "ok" if record.ok else "ERROR"
        print(f"[turn {turn:02d}] {flag:<5} {record.op} {record.args} -> {record.receipt}")

    print(f"[task] {args.suite}:{args.task_id} seed={args.seed}: {instruction}")
    print(
        f"[agent] model={args.model} protocol={args.protocol} "
        f"renderer={renderer.name}"
    )
    result = run_episode(
        provider,
        api,
        instruction,
        trace_dir=str(trace_dir),
        config=RunConfig(
            max_turns=args.max_turns,
            max_time_s=args.max_time_s,
            canvas_window_k=args.canvas_window_k,
            on_turn=on_turn,
        ),
        max_physical_ops=args.max_physical_ops,
        env_check=env.task_completed,
        renderer=renderer,
    )
    print(
        f"[result] mode={result.terminate_mode.value} turns={result.turns} "
        f"claimed={result.claimed_success} env_success={result.env_success}"
    )
    if result.detail:
        print(f"[detail] {result.detail}")
    print(f"[trace] {result.trace_dir}")
    return 1 if result.terminate_mode.value == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
