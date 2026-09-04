"""Run the Main-ReAct VAW Runtime on a real LIBERO-PRO task.

The default ``agent`` mode lets Main synchronously delegate local refinement.
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
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from vaw.context_runtime.perception_defaults import (
    DEFAULT_GROUNDING_MODEL,
    DEFAULT_POINT_MODEL,
)
from vaw.context_runtime.services import preflight_services
from vaw.mmskill import MMSkillLibrary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("agent", "scripted"), default="agent")
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="vapi/qwen3.5-plus", help="Main Agent model")
    parser.add_argument(
        "--grounding-model",
        default=DEFAULT_GROUNDING_MODEL,
        help=(
            "VLM used only by detect_region candidate generation/review "
            f"(default: {DEFAULT_GROUNDING_MODEL})"
        ),
    )
    parser.add_argument(
        "--grounding-coord-space",
        choices=("auto", "pixel", "norm1000", "norm1000_yxyx"),
        default="auto",
        help="detect_region box coordinate protocol (default: infer from model)",
    )
    parser.add_argument(
        "--point-model",
        default=DEFAULT_POINT_MODEL,
        help=f"VLM used only by locate_point (default: {DEFAULT_POINT_MODEL})",
    )
    parser.add_argument(
        "--point-coord-space",
        choices=("auto", "pixel_xy", "norm1000_xy", "norm1000_yx"),
        default="auto",
        help="locate_point coordinate protocol (default: infer from model)",
    )
    parser.add_argument(
        "--imagination-model",
        default=None,
        help="Imagination Agent model (default: reuse --model)",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument(
        "--protocol",
        choices=("native", "text"),
        default="native",
        help=(
            "native 将标准 tools 交给服务端按模型 chat template 编译；"
            "text 仅用于不支持工具通道的显式回退"
        ),
    )
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
            "coarse spatial planner (default: curobo); when curobo is selected, "
            "Imagination targets near the observed TCP use PyRoki while farther "
            "targets remain on CuRobo"
        ),
    )
    parser.add_argument(
        "--preview-gripper",
        choices=("fk-mesh", "semantic-wireframe"),
        default="fk-mesh",
        help=(
            "virtual gripper rendering only: exact returned-joints FK mesh "
            "(default) or a symmetric metric wireframe for visual ablation"
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
    parser.add_argument(
        "--semantic-render-scale",
        type=float,
        default=2.0,
        help=(
            "private agentview render scale used only by semantic grounding "
            "(default: 2.0; Canvas and RGB-D geometry remain unchanged)"
        ),
    )
    parser.add_argument("--trace-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--skill-root",
        type=pathlib.Path,
        default=None,
        help=(
            "技能目录或 evolution 实验根目录；实验根目录默认读取 active generation"
        ),
    )
    parser.add_argument(
        "--skill-generation",
        default=None,
        help="显式选择 --skill-root 实验中的 generation，用于候选评测",
    )
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


def _resolve_skill_library(
    skill_root: pathlib.Path | None,
    skill_generation: str | None,
) -> MMSkillLibrary:
    """解析 Runtime 技能来源；候选 generation 只能通过显式参数加载。"""

    from vaw.evolution.store import GenerationStore

    if skill_root is None:
        if skill_generation is not None:
            raise ValueError("--skill-generation 必须与 --skill-root 一起使用")
        return MMSkillLibrary.builtin()

    root = skill_root.resolve()
    if (root / "experiment.json").is_file():
        store = GenerationStore.open(root)
        generation_id = skill_generation or store.active_generation()
        manifest = store.read_manifest(generation_id)
        return MMSkillLibrary.from_root(
            store.generation_path(generation_id) / "skills",
            source=str(root),
            generation_id=generation_id,
            expected_digest=manifest.skill_digest,
        )
    if skill_generation is not None:
        raise ValueError("普通 --skill-root 不支持 --skill-generation")
    return MMSkillLibrary.from_root(root)


def main() -> int:
    args = _parse_args()
    try:
        skill_library = _resolve_skill_library(
            args.skill_root,
            args.skill_generation,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        preflight_services()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.context_runtime.contact_camera import (
        LiberoContactCameraProvider,
        LiberoOppositeSceneCameraProvider,
    )
    from vaw.context_runtime.libero_sensor import make_libero_semantic_rgb_provider
    from vaw.context_runtime.point_grounding import resolve_point_coord_space
    from vaw.context_runtime.semantic_grounding import resolve_grounding_coord_space
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
    grounding_coord_space = None
    point_coord_space = None
    if args.grounding_model:
        grounding_coord_space = resolve_grounding_coord_space(
            args.grounding_model,
            args.grounding_coord_space,
        )
    if args.point_model:
        point_coord_space = resolve_point_coord_space(
            args.point_model,
            args.point_coord_space,
        )
    if args.grounding_model or args.point_model:
        api.configure_vlm_backend(server_url=args.server_url, api_key=args.api_key)
    semantic_rgb_provider = make_libero_semantic_rgb_provider(
        env,
        scale=args.semantic_render_scale,
    )
    contact_camera_provider = LiberoContactCameraProvider(env)
    opposite_scene_camera_provider = LiberoOppositeSceneCameraProvider(env)
    condition = "scripted" if args.mode == "scripted" else _safe_name(args.model)
    trace_dir = args.trace_dir or (
        pathlib.Path(__file__).resolve().parent.parent
        / "out"
        / "context_runs"
        / f"{condition}_{args.motion_backend}_{args.suite}_t{args.task_id}_s{args.seed}"
    )
    trace = ContextTraceLogger(trace_dir)
    trace.log_meta(
        {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "model": args.model,
            "grounding_model": args.grounding_model,
            "grounding_coord_space": grounding_coord_space,
            "point_model": args.point_model,
            "point_coord_space": point_coord_space,
            "imagination_model": args.imagination_model or args.model,
            "semantic_render_scale": args.semantic_render_scale,
            "contact_camera": "mujoco-direct-simulation-only",
            "opposite_scene_camera": "mujoco-direct-simulation-only",
            "preview_gripper": args.preview_gripper,
        }
    )
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
    local_motion = "pyroki" if args.motion_backend == "curobo" else args.motion_backend
    print(
        f"[task] {args.suite}:{args.task_id} seed={args.seed} "
        f"motion={args.motion_backend} local={local_motion}: {task_prompt}"
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
                grounding_model=args.grounding_model,
                grounding_coord_space=grounding_coord_space,
                point_model=args.point_model,
                point_coord_space=point_coord_space,
                scripted_refinement=args.scripted_refinement,
                semantic_rgb_provider=semantic_rgb_provider,
                contact_camera_provider=contact_camera_provider,
                opposite_scene_camera_provider=opposite_scene_camera_provider,
                preview_gripper_style=args.preview_gripper,
            )
        return _run_agent(
            api,
            env,
            task_prompt,
            renderer,
            trace,
            args,
            skill_library=skill_library,
            grounding_model=args.grounding_model,
            grounding_coord_space=grounding_coord_space,
            point_model=args.point_model,
            point_coord_space=point_coord_space,
            semantic_rgb_provider=semantic_rgb_provider,
            contact_camera_provider=contact_camera_provider,
            opposite_scene_camera_provider=opposite_scene_camera_provider,
        )
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
    api: Any,
    env: Any,
    task_prompt: str,
    renderer: Any,
    trace: Any,
    args: argparse.Namespace,
    *,
    skill_library: MMSkillLibrary,
    grounding_model: str | None,
    grounding_coord_space: str | None,
    point_model: str | None,
    point_coord_space: str | None,
    semantic_rgb_provider: Any,
    contact_camera_provider: Any,
    opposite_scene_camera_provider: Any,
) -> int:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider
    from vaw.context_runtime.packet import ContextCompiler
    from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime
    from vaw.context_runtime.workspace import ContextWorkspace
    from vaw.observatory.media import ActionMediaRecorder

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
        try:
            # ffmpeg children flip the shared stdout/stderr fds to
            # non-blocking; restore before printing so a full tee pipe
            # blocks instead of raising BlockingIOError.
            os.set_blocking(sys.stdout.fileno(), True)
            if record.thought:
                print(f"[basis {turn:02d}] {record.thought}")
            print(f"[turn {turn:02d}] {flag:<5} {record.op} {record.args} -> {record.result}")
        except OSError:
            pass

    runtime = ContextRuntime(
        provider,
        ContextWorkspace(
            api,
            task_prompt,
            motion_backend=args.motion_backend,
            grounding_model=grounding_model,
            grounding_coord_space=grounding_coord_space,
            point_model=point_model,
            point_coord_space=point_coord_space,
            semantic_rgb_provider=semantic_rgb_provider,
            contact_camera_provider=contact_camera_provider,
            opposite_scene_camera_provider=opposite_scene_camera_provider,
        ),
        renderer,
        imagination_provider=imagination_provider,
        compiler=ContextCompiler(preview_gripper_style=args.preview_gripper),
        config=ContextRunConfig(
            max_main_turns=args.max_turns,
            max_imagination_turns=args.max_imagination_turns,
            max_time_s=args.max_time_s,
            max_physical_ops=args.max_physical_ops,
            on_turn=on_turn,
        ),
        trace=trace,
        action_media_recorder=(
            ActionMediaRecorder(env, trace, fps=args.video_fps)
            if args.record_video
            else None
        ),
        env_check=env.task_completed,
        env_terminal_check=lambda: bool(getattr(env, "_current_done", False)),
        skill_library=skill_library,
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
    grounding_model: str | None,
    grounding_coord_space: str | None,
    point_model: str | None,
    point_coord_space: str | None,
    scripted_refinement: bool,
    semantic_rgb_provider: Any,
    contact_camera_provider: Any | None = None,
    opposite_scene_camera_provider: Any | None = None,
    preview_gripper_style: str = "fk-mesh",
) -> int:
    from vaw.context_runtime.packet import CONTEXT_SCHEMA, ContextCompiler
    from vaw.context_runtime.protocol import main_function_registry
    from vaw.context_runtime.workspace import ContextWorkspace

    workspace = ContextWorkspace(
        api,
        task_prompt,
        motion_backend=motion_backend,
        grounding_model=grounding_model,
        grounding_coord_space=grounding_coord_space,
        point_model=point_model,
        point_coord_space=point_coord_space,
        semantic_rgb_provider=semantic_rgb_provider,
        contact_camera_provider=contact_camera_provider,
        opposite_scene_camera_provider=opposite_scene_camera_provider,
    )
    compiler = ContextCompiler(preview_gripper_style=preview_gripper_style)
    trace.log_meta(
        {
            "mode": "scripted",
            "context_schema": CONTEXT_SCHEMA,
            "task_prompt": task_prompt,
            "renderer": renderer.name,
            "motion_backend": workspace.motion_backend_name,
            "local_motion_backend": workspace.local_motion_backend_name,
            "preview_gripper": preview_gripper_style,
        }
    )
    turn = 0
    function_registry = main_function_registry()

    def step(name: str, *, trace_owner: str = "main", **arguments: Any) -> dict[str, Any]:
        nonlocal turn
        packet = (
            compiler.compile_imagination(workspace)
            if workspace.state.imagination is not None
            else compiler.compile(workspace)
        )
        image = renderer.render(packet)
        result = (
            workspace.execute_imagination(name, **arguments)
            if trace_owner == "imagination"
            else workspace.execute(name, **arguments)
        )
        spec = function_registry.get(name) if trace_owner == "main" else None
        env_success = (
            bool(env.task_completed())
            if spec is not None and spec.world_effect == "physical"
            else None
        )
        trace.log_turn(
            turn=turn,
            agent_owner=trace_owner,
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

    step("open_gripper")
    region_id = step("detect_region", query=object_query)["region_id"]
    step("locate_point", query=point_query, within_region_id=region_id)
    seed_ids = step("propose_grasps", region_id=region_id)["seed_ids"]
    if not seed_ids:
        raise RuntimeError("scripted smoke received no grasp candidates")
    action_id = step("preview_grasp", seed_id=seed_ids[0])["action_id"]
    workspace.begin_imagination(
        f"使两指围绕 {object_query} 形成可审查的对称接触几何",
        action_id,
    )
    if scripted_refinement:
        step(
            "delta_move",
            trace_owner="imagination",
            delta_xyz_m=[0.0, 0.0, 0.03],
            frame="base",
        )
        step(
            "rotate",
            trace_owner="imagination",
            axis="z",
            angle_deg=5.0,
            frame="tool",
        )
    action_id = step(
        "finish_imagination", trace_owner="imagination", status="ready"
    )["action_id"]
    step("execute_action", action_id=action_id)
    step("close_gripper")
    step("finish_task", success=False)

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
