"""Compare PyRoki and CuRobo for the same local TCP corrections.

This diagnostic deliberately bypasses Main, grounding, and Imagination.  Each
backend receives the same LIBERO task, seed, reset state, and base-frame TCP
deltas.  The report therefore measures the motion layer rather than VLM policy
quality.

Example:

    .venv-libero/bin/python -m vaw.diagnostics.compare_local_motion_backends \
        --suite libero_object --task-id 1 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


DEFAULT_DELTAS = (
    (-0.025, 0.007, 0.0),
    (0.020, 0.006, 0.0),
    (-0.030, 0.000, 0.0),
)
STRICT_CUROBO_THRESHOLD_M = 0.002


class _ThresholdedCuroboApi:
    """Diagnostic-only proxy that tightens CuRobo's Cartesian goal tolerance."""

    def __init__(self, api: Any, threshold_m: float) -> None:
        self._api = api
        self._threshold_m = float(threshold_m)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._api, name)

    def plan_grasp_trajectory(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["position_threshold"] = self._threshold_m
        kwargs["position_threshold_z"] = self._threshold_m
        return self._api.plan_grasp_trajectory(*args, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--backends",
        default="pyroki,curobo,curobo_strict_2mm",
        help="comma-separated subset of pyroki,curobo,curobo_strict_2mm",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="output directory (default: timestamped vaw/out/diagnostics directory)",
    )
    return parser.parse_args()


def _vector(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)[:3]


def _tcp(workspace: Any) -> np.ndarray:
    robot = workspace.state.robot
    if robot is None or robot.tcp_pose is None:
        raise RuntimeError("workspace has no observed TCP pose")
    return _vector(robot.tcp_pose.position_xyz)


def _agentview(api: Any) -> np.ndarray | None:
    observation = api.get_observation()
    camera = observation.get("agentview") if isinstance(observation, dict) else None
    if not isinstance(camera, dict):
        return None
    images = camera.get("images")
    if not isinstance(images, dict) or "rgb" not in images:
        return None
    image = np.asarray(images["rgb"])
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    return np.asarray(image[:, :, :3], dtype=np.uint8)


def _save_pair(
    path: pathlib.Path,
    before: np.ndarray | None,
    after: np.ndarray | None,
    *,
    title: str,
    detail: str,
) -> None:
    if before is None or after is None:
        return
    height = max(before.shape[0], after.shape[0])
    width = before.shape[1] + after.shape[1]
    header = 72
    canvas = Image.new("RGB", (width, height + header), (20, 24, 31))
    canvas.paste(Image.fromarray(before), (0, header))
    canvas.paste(Image.fromarray(after), (before.shape[1], header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(240, 244, 252))
    draw.text((12, 34), detail, fill=(174, 190, 214))
    draw.text((12, header + 8), "BEFORE", fill=(255, 255, 255))
    draw.text((before.shape[1] + 12, header + 8), "AFTER", fill=(255, 255, 255))
    canvas.save(path)


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()
        return
    handle = getattr(env, "handle", None)
    inner = getattr(handle, "env", None)
    close = getattr(inner, "close", None)
    if callable(close):
        close()


def _run_backend(
    backend: str,
    *,
    suite: str,
    task_id: int,
    seed: int,
    out_dir: pathlib.Path,
) -> dict[str, Any]:
    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.context_runtime.motion import CuroboMotionBackend
    from vaw.context_runtime.workspace import ContextWorkspace

    env = FrankaLiberoTask(
        suite_name=suite,
        task_id=task_id,
        privileged=False,
        seed=seed,
    )
    _, reset_info = env.reset(seed=seed)
    api = FrankaLiberoApiReduced(env)
    local_motion: str | Any = backend
    if backend == "curobo_strict_2mm":
        local_motion = CuroboMotionBackend(
            _ThresholdedCuroboApi(api, STRICT_CUROBO_THRESHOLD_M),
            waypoint_tolerance_rad=0.005,
            final_joint_tolerance_rad=0.002,
            final_settle_max_steps=240,
        )
    workspace = ContextWorkspace(
        api,
        task_prompt=str(reset_info.get("task_prompt", f"{suite}:{task_id}")),
        motion_backend="curobo",
        local_motion_backend=local_motion,
    )
    backend_dir = out_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "backend": backend,
        "suite": suite,
        "task_id": task_id,
        "seed": seed,
        "initial_tcp_xyz_m": _tcp(workspace).tolist(),
        "steps": [],
    }
    try:
        for index, requested in enumerate(DEFAULT_DELTAS, start=1):
            before_tcp = _tcp(workspace)
            target_tcp = before_tcp + np.asarray(requested, dtype=np.float64)
            before_rgb = _agentview(api)
            started = time.perf_counter()
            step = workspace.execute(
                "move_tcp_delta",
                delta_xyz_m=list(requested),
                frame="base",
            )
            elapsed_s = time.perf_counter() - started
            after_tcp = _tcp(workspace)
            after_rgb = _agentview(api)
            achieved_delta = after_tcp - before_tcp
            endpoint_error = after_tcp - target_tcp
            action = workspace.state.last_physical_action
            item = {
                "index": index,
                "requested_delta_xyz_m": list(requested),
                "before_tcp_xyz_m": before_tcp.tolist(),
                "target_tcp_xyz_m": target_tcp.tolist(),
                "after_tcp_xyz_m": after_tcp.tolist(),
                "achieved_delta_xyz_m": achieved_delta.tolist(),
                "endpoint_error_xyz_m": endpoint_error.tolist(),
                "position_error_m": float(np.linalg.norm(endpoint_error)),
                "elapsed_s": elapsed_s,
                "revision_before": step.revision_before,
                "revision_after": step.revision_after,
                "function_result": step.result,
                "physical_outcome": getattr(action, "outcome", None),
                "physical_error": getattr(action, "error_detail", None),
                "planner_backend": (
                    step.trace_diagnostics.get("direct_delta_move", {}).get(
                        "motion_backend"
                    )
                    if step.trace_diagnostics
                    else None
                ),
            }
            report["steps"].append(item)
            _save_pair(
                backend_dir / f"step_{index:02d}.png",
                before_rgb,
                after_rgb,
                title=f"{backend.upper()} local TCP correction {index}",
                detail=(
                    f"requested={np.round(requested, 4).tolist()} m  "
                    f"achieved={np.round(achieved_delta, 4).tolist()} m  "
                    f"error={item['position_error_m'] * 1000.0:.1f} mm"
                ),
            )
            (out_dir / "report.partial.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    finally:
        _close_env(env)

    errors = [float(step["position_error_m"]) for step in report["steps"]]
    report["summary"] = {
        "completed_steps": len(report["steps"]),
        "mean_position_error_m": float(np.mean(errors)) if errors else None,
        "max_position_error_m": float(np.max(errors)) if errors else None,
        "unsettled_or_failed_steps": sum(
            step["physical_outcome"] != "completed" for step in report["steps"]
        ),
        "total_elapsed_s": float(
            sum(float(step["elapsed_s"]) for step in report["steps"])
        ),
    }
    return report


def main() -> int:
    args = _parse_args()
    requested_backends = tuple(
        item.strip() for item in str(args.backends).split(",") if item.strip()
    )
    known_backends = {"pyroki", "curobo", "curobo_strict_2mm"}
    unknown_backends = sorted(set(requested_backends) - known_backends)
    if not requested_backends or unknown_backends:
        raise SystemExit(
            "--backends must contain only pyroki,curobo,curobo_strict_2mm; "
            f"unknown={unknown_backends}"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (
        pathlib.Path(__file__).resolve().parent.parent
        / "out"
        / "diagnostics"
        / f"local_motion_ab_{args.suite}_t{args.task_id}_s{args.seed}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {
        "experiment": "same-state local TCP backend A/B",
        "suite": args.suite,
        "task_id": args.task_id,
        "seed": args.seed,
        "deltas_xyz_m": [list(delta) for delta in DEFAULT_DELTAS],
        "backends": {},
    }
    for backend in requested_backends:
        print(f"[A/B] running {backend} ...", flush=True)
        try:
            result = _run_backend(
                backend,
                suite=args.suite,
                task_id=args.task_id,
                seed=args.seed,
                out_dir=out_dir,
            )
        except Exception as exc:
            result = {
                "backend": backend,
                "fatal_error": f"{type(exc).__name__}: {exc}",
            }
        combined["backends"][backend] = result
        (out_dir / "report.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary = result.get("summary", {})
        print(
            f"[A/B] {backend}: mean_error="
            f"{summary.get('mean_position_error_m')} m, "
            f"failures={summary.get('unsettled_or_failed_steps')}",
            flush=True,
        )
    print(f"[A/B] report: {out_dir / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
