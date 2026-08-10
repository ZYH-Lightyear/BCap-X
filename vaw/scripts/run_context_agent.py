"""Run the VAW M1.4 Context Runtime on a real LIBERO-PRO task.

The default ``agent`` mode runs the Main/Imagination dual-agent contract.
``scripted`` is an environment wiring/visual trace smoke, not an agent policy.

Examples:

    source .venv-libero/bin/activate
    python -m vaw.scripts.run_context_agent --mode scripted
    python -m vaw.scripts.run_context_agent --mode agent --model vapi/qwen3.5-plus
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from vaw.context_runtime.services import preflight_services


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("agent", "scripted"), default="agent")
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="vapi/qwen3.5-plus", help="Main Agent model")
    parser.add_argument(
        "--imagination-model",
        default=None,
        help="Imagination Agent model (default: reuse --model)",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="text")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-time-s", type=float, default=1800.0)
    parser.add_argument("--max-physical-ops", type=int, default=30)
    parser.add_argument("--max-imagination-turns", type=int, default=6)
    parser.add_argument(
        "--motion-backend",
        choices=("pyroki", "curobo"),
        default="curobo",
        help=(
            "private imagination planner (default: curobo); "
            "delta_move/rotate remain virtual until commit"
        ),
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save agentview, wrist, and policy-visible Context MP4 artifacts",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--context-video-fps", type=int, default=2)
    parser.add_argument("--trace-dir", type=pathlib.Path, default=None)
    parser.add_argument("--object-query", default="the alphabet soup can")
    parser.add_argument("--point-query", default="the center of the alphabet soup can")
    parser.add_argument(
        "--scripted-refinement",
        action="store_true",
        help=(
            "in scripted mode, compose a +3 cm base-Z delta and a +5 degree "
            "tool-Z rotation, then review and commit the resulting action"
        ),
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def main() -> int:
    args = _parse_args()
    try:
        preflight_services()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.context_runtime.trace import ContextTraceLogger
    from vaw.context_runtime.video import save_episode_videos
    from vaw.context_runtime.web_renderer import ContextWebRenderer

    env = FrankaLiberoTask(
        suite_name=args.suite,
        task_id=args.task_id,
        privileged=False,
        seed=args.seed,
    )
    _, reset_info = env.reset(seed=args.seed)
    task_prompt = str(reset_info["task_prompt"])
    api = FrankaLiberoApiReduced(env)
    condition = "scripted" if args.mode == "scripted" else _safe_name(args.model)
    trace_dir = args.trace_dir or (
        pathlib.Path(__file__).resolve().parent.parent
        / "out"
        / "context_runs"
        / f"{condition}_{args.motion_backend}_{args.suite}_t{args.task_id}_s{args.seed}"
    )
    trace = ContextTraceLogger(trace_dir)
    video_capture_enabled = False
    if args.record_video:
        try:
            env.enable_video_capture(True, clear=True, wrist_camera=True)
            video_capture_enabled = True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            trace.log_meta({"video_capture_error": message})
            print(f"[video] capture unavailable: {message}")
    renderer = ContextWebRenderer(headed=args.headed)
    print(
        f"[task] {args.suite}:{args.task_id} seed={args.seed} "
        f"motion={args.motion_backend}: {task_prompt}"
    )
    try:
        if args.mode == "scripted":
            return _run_scripted(
                api,
                env,
                task_prompt,
                renderer,
                trace,
                object_query=args.object_query,
                point_query=args.point_query,
                motion_backend=args.motion_backend,
                scripted_refinement=args.scripted_refinement,
            )
        return _run_agent(api, env, task_prompt, renderer, trace, args)
    finally:
        if args.record_video:
            try:
                videos = save_episode_videos(
                    trace.dir,
                    env,
                    environment_fps=args.video_fps,
                    context_fps=args.context_video_fps,
                )
                trace.log_meta({"videos": videos["artifacts"]})
                if videos.get("errors"):
                    trace.log_meta({"video_errors": videos["errors"]})
                for name, artifact in videos["artifacts"].items():
                    print(
                        f"[video] {name}: {trace.dir / artifact['path']} "
                        f"({artifact['frames']} frames @ {artifact['fps']} fps)"
                    )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                trace.log_meta({"video_export_error": message})
                print(f"[video] export failed: {message}")
            finally:
                if video_capture_enabled:
                    try:
                        env.enable_video_capture(False, clear=True, wrist_camera=True)
                    except Exception as exc:
                        print(f"[video] failed to disable capture: {exc}")
        renderer.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_agent(
    api: Any, env: Any, task_prompt: str, renderer: Any, trace: Any, args: argparse.Namespace
) -> int:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider
    from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime
    from vaw.context_runtime.workspace import ContextWorkspace

    def make_provider(model: str) -> Any:
        provider: Any = OpenAIProvider(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return TextProtocolProvider(provider) if args.protocol == "text" else provider

    provider = make_provider(args.model)
    imagination_provider = make_provider(args.imagination_model or args.model)

    def on_turn(turn: int, record: Any) -> None:
        flag = "ok" if record.ok else "ERROR"
        if record.thought:
            print(f"[basis {turn:02d}] {record.thought}")
        print(f"[turn {turn:02d}] {flag:<5} {record.op} {record.args} -> {record.result}")

    runtime = ContextRuntime(
        provider,
        ContextWorkspace(api, task_prompt, motion_backend=args.motion_backend),
        renderer,
        imagination_provider=imagination_provider,
        config=ContextRunConfig(
            max_main_turns=args.max_turns,
            max_imagination_turns=args.max_imagination_turns,
            max_time_s=args.max_time_s,
            max_physical_ops=args.max_physical_ops,
            on_turn=on_turn,
        ),
        trace=trace,
        env_check=env.task_completed,
    )
    result = runtime.run()
    print(
        f"[result] mode={result.terminate_mode.value} turns={result.turns} "
        f"claimed={result.claimed_success} env_success={result.env_success}"
    )
    print(f"[trace] {result.trace_dir}")
    return 1 if result.terminate_mode.value == "error" else 0


def _run_scripted(
    api: Any,
    env: Any,
    task_prompt: str,
    renderer: Any,
    trace: Any,
    *,
    object_query: str,
    point_query: str,
    motion_backend: str,
    scripted_refinement: bool,
) -> int:
    from vaw.context_runtime.packet import CONTEXT_SCHEMA, ContextCompiler
    from vaw.context_runtime.workspace import ContextWorkspace

    workspace = ContextWorkspace(api, task_prompt, motion_backend=motion_backend)
    compiler = ContextCompiler()
    trace.log_meta(
        {
            "mode": "scripted",
            "context_schema": CONTEXT_SCHEMA,
            "task_prompt": task_prompt,
            "renderer": renderer.name,
            "motion_backend": workspace.motion_backend_name,
        }
    )
    turn = 0

    def step(name: str, **arguments: Any) -> dict[str, Any]:
        nonlocal turn
        owner = workspace.state.owner
        packet = compiler.compile(workspace)
        image = renderer.render(packet)
        result = workspace.execute(name, **arguments)
        if owner == "main" and name != "commit" and result.ok:
            workspace.consume_main_context()
        env_success = bool(env.task_completed()) if name in workspace.PHYSICAL_FUNCTIONS else None
        trace.log_turn(
            turn=turn,
            agent_owner=owner,
            image=image,
            packet=packet,
            function_call={"name": name, "arguments": arguments},
            step=result,
            thought="scripted smoke",
            env_success=env_success,
            done=workspace.finished,
        )
        flag = "ok" if result.ok else "ERROR"
        print(f"[step {turn:02d}] {flag:<5} {name} {arguments} -> {result.result}")
        turn += 1
        if not result.ok:
            raise RuntimeError(result.result["error"])
        return result.result

    workspace.set_refinement_goal("只把虚拟夹爪目标设为 open")
    step("open_gripper")
    action_id = step("finish_imagination", status="ready")["action_id"]
    step("commit", action_id=action_id)
    region_id = step("detection_and_sam", query=object_query)["region_id"]
    step("locate_point", query=point_query, within_region_id=region_id)
    seed_ids = step("propose_grasps", region_id=region_id)["seed_ids"]
    if not seed_ids:
        raise RuntimeError("scripted smoke received no grasp candidates")
    workspace.set_refinement_goal(
        f"使两指围绕 {object_query} 形成可审查的对称接触几何"
    )
    step("select", seed_id=seed_ids[0])
    action_id = step("finish_imagination", status="ready")["action_id"]
    step("commit", action_id=action_id)
    workspace.set_refinement_goal("只把虚拟夹爪目标设为 closed")
    step("close_gripper")
    action_id = step("finish_imagination", status="ready")["action_id"]
    step("commit", action_id=action_id)
    if scripted_refinement:
        workspace.set_refinement_goal("验证累计小幅平移与旋转在 Contact Focus 中清晰可见")
        step(
            "delta_move",
            delta_xyz_m=[0.0, 0.0, 0.03],
            frame="base",
        )
        step(
            "rotate",
            axis="z",
            angle_deg=5.0,
            frame="tool",
        )
        action_id = step("finish_imagination", status="ready")["action_id"]
        step("commit", action_id=action_id)
    step("done", success=False)

    final_packet = compiler.compile(workspace)
    final_image = renderer.render(final_packet)
    trace.log_turn(
        turn=turn,
        agent_owner="runtime",
        image=final_image,
        packet=final_packet,
        function_call=None,
        step=None,
        thought="final post-action Context",
        env_success=bool(env.task_completed()),
        done=True,
    )
    trace.log_meta(
        {
            "turns": turn,
            "claimed_success": workspace.claimed_success,
            "env_success": bool(env.task_completed()),
        }
    )
    print(f"[trace] {trace.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
