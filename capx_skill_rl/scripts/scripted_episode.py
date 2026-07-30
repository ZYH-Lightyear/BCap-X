"""M2 scripted pick-and-place episode using only the ten public tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from capx_skill_rl.backends import LiberoBackendConfig, create_libero_backend
from capx_skill_rl.env import StepResult, ToolEnv

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "libero_pro.yaml"


class ScriptedEpisode:
    """Small deterministic controller over the model-facing tool boundary."""

    def __init__(self, env: ToolEnv, output_dir: Path) -> None:
        self.env = env
        self.output_dir = output_dir
        self.trace: list[dict[str, Any]] = []
        self.frame_index = 0

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        frame_label: str | None = None,
    ) -> StepResult:
        if self.env.done:
            raise RuntimeError(f"episode ended before {name}")
        step = self.env.step({"name": name, "arguments": arguments})
        entry = {
            "action": {"name": name, "arguments": arguments},
            "result": step.result,
            "reward": step.reward,
            "done": step.done,
        }
        self.trace.append(entry)
        print(
            f"[{len(self.trace):02d}] {name}: {step.result} reward={step.reward} done={step.done}",
            flush=True,
        )
        if "error" in step.result:
            raise RuntimeError(f"{name} failed: {step.result['error']}")
        if step.observation is not None and frame_label is not None:
            self.save_frame(frame_label, step.observation.rgb)
        return step

    def move_pose(
        self,
        label: str,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> StepResult:
        _validate_workspace(position, label)
        ik = self.call(
            "solve_ik",
            {
                "position": position.tolist(),
                "quaternion": quaternion.tolist(),
            },
        ).result
        return self.call(
            "move_to_joints",
            {"joints": ik["joints"]},
            frame_label=label,
        )

    def save_frame(self, label: str, rgb: np.ndarray | None = None) -> Path:
        if rgb is None:
            assert self.env.context is not None
            rgb = self.env.context.frame.rgb
        path = self.output_dir / f"{self.frame_index:02d}_{label}.png"
        self.frame_index += 1
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
        return path

    def save_grounding(
        self,
        bbox: list[float],
        point: list[float],
    ) -> Path:
        assert self.env.context is not None
        image = Image.fromarray(self.env.context.frame.rgb.copy())
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(bbox), outline=(255, 220, 0), width=4)
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 40, 40))
        path = self.output_dir / f"{self.frame_index:02d}_target_grounding.png"
        self.frame_index += 1
        image.save(path)
        return path


def run_scripted_episode(
    env: ToolEnv,
    *,
    output_dir: Path,
    target_query: str,
    receptacle_query: str,
    seed: int,
    approach_height: float,
    lift_height: float,
    transfer_clearance: float,
    release_clearance: float,
) -> dict[str, Any]:
    runner = ScriptedEpisode(env, output_dir)
    observation = env.reset(seed=seed)
    runner.save_frame("initial", observation.rgb)
    print(f"task: {observation.task}", flush=True)

    receptacle_point = runner.call(
        "vlm_point_detection",
        {"query": receptacle_query},
    ).result["point"]
    receptacle_mask = runner.call(
        "sam3",
        {"point": receptacle_point},
    ).result["mask_id"]
    receptacle_obb = runner.call(
        "get_obb",
        {"mask_id": receptacle_mask},
    ).result

    bbox = runner.call(
        "vlm_bbox_detection",
        {"query": target_query},
    ).result["bbox"]
    point = runner.call(
        "vlm_point_detection",
        {"query": target_query},
    ).result["point"]
    bbox_point_consistent = _point_near_bbox(point, bbox, tolerance=16.0)
    if not bbox_point_consistent:
        print(
            "warning: bbox and point detectors selected different instances; "
            "using the point-prompt mask",
            flush=True,
        )
    runner.save_grounding(bbox, point)

    target_mask = runner.call("sam3", {"point": point}).result["mask_id"]
    target_obb = runner.call(
        "get_obb",
        {"mask_id": target_mask},
    ).result
    grasp = runner.call(
        "plan_grasp",
        {"mask_id": target_mask},
    ).result

    target_center = np.asarray(target_obb["center"], dtype=np.float64)
    grasp_position = np.asarray(grasp["position"], dtype=np.float64)
    grasp_quaternion = np.asarray(grasp["quaternion"], dtype=np.float64)
    receptacle_center = np.asarray(receptacle_obb["center"], dtype=np.float64)
    receptacle_top = _obb_top_z(receptacle_obb)
    _validate_grounding(target_center, grasp_position, receptacle_center)

    pregrasp = grasp_position + np.array([0.0, 0.0, approach_height])
    lift = grasp_position + np.array([0.0, 0.0, lift_height])
    transfer_z = max(lift[2], receptacle_top + transfer_clearance)
    transfer = np.array(
        [receptacle_center[0], receptacle_center[1], transfer_z],
        dtype=np.float64,
    )
    release = np.array(
        [
            receptacle_center[0],
            receptacle_center[1],
            receptacle_top + release_clearance,
        ],
        dtype=np.float64,
    )
    geometry = {
        "bbox_point_consistent": bbox_point_consistent,
        "target_obb": target_obb,
        "receptacle_obb": receptacle_obb,
        "receptacle_top_z": receptacle_top,
        "grasp": grasp,
        "pregrasp_position": pregrasp.tolist(),
        "lift_position": lift.tolist(),
        "transfer_position": transfer.tolist(),
        "release_position": release.tolist(),
    }
    print(f"geometry: {geometry}", flush=True)

    runner.call("open_gripper", {}, frame_label="gripper_open")
    runner.move_pose("pregrasp", pregrasp, grasp_quaternion)
    runner.move_pose("at_grasp", grasp_position, grasp_quaternion)
    runner.call("close_gripper", {}, frame_label="gripper_closed")
    runner.move_pose("lifted", lift, grasp_quaternion)
    runner.move_pose("over_receptacle", transfer, grasp_quaternion)
    runner.move_pose("release_pose", release, grasp_quaternion)
    release_step = runner.call(
        "open_gripper",
        {},
        frame_label="released",
    )

    success = release_step.reward == 1.0
    if not success and not env.done:
        retreat = release + np.array([0.0, 0.0, approach_height])
        retreat_step = runner.move_pose("retreat", retreat, grasp_quaternion)
        success = retreat_step.reward == 1.0
    if not success and not env.done:
        home_step = runner.call("go_home", {}, frame_label="home")
        success = home_step.reward == 1.0

    return {
        "task": observation.task,
        "seed": seed,
        "target_query": target_query,
        "receptacle_query": receptacle_query,
        "success": success,
        "geometry": geometry,
        "trace": runner.trace,
    }


def _point_near_bbox(
    point: list[float],
    bbox: list[float],
    *,
    tolerance: float,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance


def _obb_top_z(obb: dict[str, Any]) -> float:
    center = np.asarray(obb["center"], dtype=np.float64)
    extent = np.asarray(obb["extent"], dtype=np.float64)
    rotation = Rotation.from_quat(obb["quaternion"]).as_matrix()
    vertical_radius = float(np.abs(rotation[2]) @ (extent / 2.0))
    return float(center[2] + vertical_radius)


def _validate_grounding(
    target_center: np.ndarray,
    grasp_position: np.ndarray,
    receptacle_center: np.ndarray,
) -> None:
    _validate_workspace(grasp_position, "grasp")
    if not (0.10 <= target_center[0] <= 0.70):
        raise RuntimeError(f"target center x={target_center[0]:.3f} is outside the safe workspace")
    if np.linalg.norm(grasp_position[:2] - target_center[:2]) > 0.12:
        raise RuntimeError("planned grasp is more than 12 cm from the target OBB center in XY")
    if not (0.10 <= receptacle_center[0] <= 0.70):
        raise RuntimeError(
            f"receptacle center is outside the safe workspace: {receptacle_center.tolist()}"
        )


def _validate_workspace(position: np.ndarray, label: str) -> None:
    lower = np.array([-0.10, -0.50, 0.005])
    upper = np.array([0.75, 0.50, 0.90])
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RuntimeError(f"{label} position must contain three finite values")
    if np.any(position < lower) or np.any(position > upper):
        raise RuntimeError(f"{label} position {position.tolist()} is outside IK workspace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-name")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target", default="alphabet soup can")
    parser.add_argument("--receptacle", default="basket")
    parser.add_argument("--approach-height", type=float, default=0.075)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--transfer-clearance", type=float, default=0.22)
    parser.add_argument("--release-clearance", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    raw = _load_yaml(args.config)
    config = _backend_config(raw, args.suite_name, args.task_id)
    output_dir = args.output_dir or Path("outputs/capx_skill_rl/m2") / (
        f"{config.suite_name}_{config.task_id}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = create_libero_backend(config)
    env = ToolEnv(backend, max_steps=int(raw.get("max_tool_steps", 32)))
    report: dict[str, Any]
    try:
        report = run_scripted_episode(
            env,
            output_dir=output_dir,
            target_query=args.target,
            receptacle_query=args.receptacle,
            seed=args.seed,
            approach_height=args.approach_height,
            lift_height=args.lift_height,
            transfer_clearance=args.transfer_clearance,
            release_clearance=args.release_clearance,
        )
    except Exception as exc:
        report = {
            "task": getattr(env, "task", ""),
            "seed": args.seed,
            "target_query": args.target,
            "receptacle_query": args.receptacle,
            "success": False,
            "error": str(exc),
        }
        raise
    finally:
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report: {report_path}", flush=True)
        backend.close()

    if not report["success"]:
        raise SystemExit(1)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def _backend_config(
    raw: dict[str, Any],
    suite_name: str | None,
    task_id: int | None,
) -> LiberoBackendConfig:
    vlm = raw.get("vlm") or {}
    return LiberoBackendConfig(
        suite_name=suite_name or raw.get("suite_name", "libero_object_swap"),
        task_id=task_id if task_id is not None else int(raw.get("task_id", 0)),
        max_sim_steps=int(raw.get("max_sim_steps", 8000)),
        vlm_model=str(vlm.get("model", "vapi/gpt-5.5")),
        vlm_server_url=str(vlm.get("server_url", "http://localhost:8110/chat/completions")),
        vlm_api_key=vlm.get("api_key"),
        vlm_coord_space=str(vlm.get("coord_space", "pixel")),
    )


if __name__ == "__main__":
    main()
