"""M1 live validation for the ten-tool LIBERO-PRO action space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from capx_skill_rl.backends import (
    CapXLiberoBackend,
    LiberoBackendConfig,
    create_libero_backend,
)
from capx_skill_rl.env import StepResult, ToolEnv
from capx_skill_rl.tools import TOOL_NAMES

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "libero_pro.yaml"
EXPECTED_TOOLS = (
    "vlm_bbox_detection",
    "vlm_point_detection",
    "sam3",
    "get_obb",
    "plan_grasp",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "go_home",
)


class LiveValidator:
    """Run live calls while retaining a compact machine-readable report."""

    def __init__(self, env: ToolEnv) -> None:
        self.env = env
        self.checks: list[dict[str, Any]] = []

    @property
    def passed(self) -> bool:
        return all(bool(check["ok"]) for check in self.checks)

    def check(self, name: str, condition: bool, error: str) -> None:
        if condition:
            self._record(name, True)
        else:
            self._record(name, False, error=error)

    def tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        label: str | None = None,
        physical: bool = False,
    ) -> dict[str, Any] | None:
        check_name = label or name
        assert self.env.context is not None
        revision_before = self.env.context.frame.revision
        try:
            step = self.env.step({"name": name, "arguments": arguments})
        except Exception as exc:
            self._record(check_name, False, error=str(exc))
            return None

        if "error" in step.result:
            self._record(check_name, False, error=str(step.result["error"]))
            return None

        error = self._step_contract_error(
            step,
            physical=physical,
            revision_before=revision_before,
        )
        if error is not None:
            self._record(check_name, False, error=error)
            return None

        self._record(check_name, True, result=step.result)
        return step.result

    def stale_mask(self, mask_id: str) -> None:
        try:
            step = self.env.step({"name": "get_obb", "arguments": {"mask_id": mask_id}})
        except Exception as exc:
            self._record("mask_invalid_after_motion", False, error=str(exc))
            return
        error = str(step.result.get("error") or "")
        self.check(
            "mask_invalid_after_motion",
            bool(error) and ("mask_id" in error),
            f"old mask unexpectedly remained usable: {step.result}",
        )

    def _step_contract_error(
        self,
        step: StepResult,
        *,
        physical: bool,
        revision_before: int,
    ) -> str | None:
        assert self.env.context is not None
        if not physical:
            if step.observation is not None:
                return "non-physical tool unexpectedly refreshed the observation"
            if self.env.context.frame.revision != revision_before:
                return "non-physical tool unexpectedly changed frame revision"
            return None
        if step.result != {}:
            return f"successful physical tool must return {{}}, got {step.result}"
        if step.observation is None:
            return "physical tool did not return a refreshed RGB observation"
        if self.env.context.frame.revision != revision_before + 1:
            return "physical tool did not advance frame revision by one"
        return None

    def _record(
        self,
        name: str,
        ok: bool,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        check: dict[str, Any] = {"name": name, "ok": ok}
        if result:
            check["result"] = result
        if error:
            check["error"] = error
        self.checks.append(check)
        suffix = f": {result}" if result else f": {error}" if error else ""
        print(f"{'PASS' if ok else 'FAIL'} {name}{suffix}", flush=True)


def run_validation(
    env: ToolEnv,
    backend: CapXLiberoBackend,
    *,
    config: LiberoBackendConfig,
    query: str,
    seed: int,
) -> dict[str, Any]:
    validator = LiveValidator(env)
    observation = env.reset(seed=seed)
    print(f"task: {observation.task}", flush=True)
    print(
        f"rgb: shape={observation.rgb.shape}, dtype={observation.rgb.dtype}",
        flush=True,
    )
    validator.check(
        "tool_list",
        TOOL_NAMES == EXPECTED_TOOLS,
        f"expected {EXPECTED_TOOLS}, got {TOOL_NAMES}",
    )

    bbox = validator.tool(
        "vlm_bbox_detection",
        {"query": query},
    )
    point = validator.tool(
        "vlm_point_detection",
        {"query": query},
    )

    mask_ids: list[str] = []
    if bbox is not None:
        result = validator.tool(
            "sam3",
            {"bbox": bbox["bbox"]},
            label="sam3[bbox]",
        )
        if result is not None:
            mask_ids.append(str(result["mask_id"]))
    else:
        validator.check("sam3[bbox]", False, "bbox detection failed")

    if point is not None:
        result = validator.tool(
            "sam3",
            {"point": point["point"]},
            label="sam3[point]",
        )
        if result is not None:
            mask_ids.append(str(result["mask_id"]))
    else:
        validator.check("sam3[point]", False, "point detection failed")

    result = validator.tool(
        "sam3",
        {"text": query},
        label="sam3[text]",
    )
    if result is not None:
        mask_ids.append(str(result["mask_id"]))

    grasp: dict[str, Any] | None = None
    if mask_ids:
        mask_id = mask_ids[0]
        validator.tool("get_obb", {"mask_id": mask_id})
        grasp = validator.tool("plan_grasp", {"mask_id": mask_id})
    else:
        validator.check("get_obb", False, "no valid SAM3 mask")
        validator.check("plan_grasp", False, "no valid SAM3 mask")

    if grasp is not None:
        validator.tool(
            "solve_ik",
            {
                "position": grasp["position"],
                "quaternion": grasp["quaternion"],
            },
        )
    else:
        validator.check("solve_ik", False, "grasp planning failed")

    try:
        robot_state = np.asarray(
            backend.api.get_observation()["robot_joint_pos"],
            dtype=np.float64,
        ).reshape(-1)
        if robot_state.size < 7 or not np.isfinite(robot_state[:7]).all():
            raise ValueError("current robot state has no seven finite arm joints")
        current_joints = [float(value) for value in robot_state[:7]]
        validator.tool(
            "move_to_joints",
            {"joints": current_joints},
            physical=True,
        )
    except Exception as exc:
        validator.check("move_to_joints", False, str(exc))

    if mask_ids:
        validator.stale_mask(mask_ids[0])
    else:
        validator.check(
            "mask_invalid_after_motion",
            False,
            "no mask was created before motion",
        )

    validator.tool("close_gripper", {}, physical=True)
    validator.tool("open_gripper", {}, physical=True)
    validator.tool("go_home", {}, physical=True)

    return {
        "suite_name": config.suite_name,
        "task_id": config.task_id,
        "task": observation.task,
        "passed": validator.passed,
        "checks": validator.checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-name")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--query", default="alphabet soup")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    raw = _load_yaml(args.config)
    config = _backend_config(raw, args.suite_name, args.task_id)
    report_path = args.report or Path("outputs/capx_skill_rl/m1") / (
        f"{config.suite_name}_{config.task_id}.json"
    )
    backend = create_libero_backend(config)
    env = ToolEnv(backend, max_steps=int(raw.get("max_tool_steps", 32)))
    try:
        report = run_validation(
            env,
            backend,
            config=config,
            query=args.query,
            seed=args.seed,
        )
    finally:
        backend.close()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"report: {report_path}", flush=True)
    if not report["passed"]:
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
