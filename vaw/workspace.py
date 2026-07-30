"""Workspace facade: binds a Cap-X control API to the action state, executes
ops, renders the canvas, and logs every step in the training-data format.

Usage (M1, inside a Cap-X launch):

    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.workspace import Workspace

    api = FrankaLiberoApiReduced(env)
    ws = Workspace(api, instruction="put the red mug on the plate",
                   trace_dir="runs/vaw/ep0")
    result = ws.step("observe")
    result = ws.step("ground", text="red mug")
    ...
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from vaw.cloud import SceneCloud, build_scene_cloud
from vaw.ops import OPS, OpError
from vaw.render import render_canvas
from vaw.state import ActionState
from vaw.types import Pose, StepResult


class TraceLogger:
    """One episode = steps.jsonl + canvas_XXXX.png. Teacher traces and student
    rollouts share this format; SFT/RL data builders consume it directly."""

    def __init__(self, trace_dir: str | pathlib.Path) -> None:
        self.dir = pathlib.Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.dir / "steps.jsonl"
        self._index = 0
        # One directory holds exactly one episode. Without this, rerunning into
        # the same directory appended a second episode's steps while the canvas
        # indices restarted at 0 — the jsonl and the images silently disagreed,
        # and a data builder reading the file would see one impossible episode.
        for stale in (
            [self._steps_path, self.dir / "meta.json"] + sorted(self.dir.glob("canvas_*.png"))
        ):
            stale.unlink(missing_ok=True)

    def log(self, result: StepResult) -> None:
        canvas_name = None
        if result.canvas is not None:
            canvas_name = f"canvas_{self._index:04d}.png"
            Image.fromarray(result.canvas).save(self.dir / canvas_name)
        record = {
            "index": self._index,
            "time": time.time(),
            "op": result.op,
            "args": result.args,
            "ok": result.ok,
            "physical": result.physical,
            "receipt": result.receipt_text,
            "canvas": canvas_name,
            "state": result.state_summary,
        }
        with self._steps_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._index += 1

    def log_meta(self, meta: dict[str, Any]) -> None:
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))


class Workspace:
    """The visual action workspace an agent operates.

    ``api`` is duck-typed: any object exposing the FrankaLiberoApiReduced
    surface used by the ops (get_observation, vlm_bbox_detection,
    segment_sam3_box_prompt, plan_grasp, solve_ik, move_to_joints,
    open_gripper/close_gripper, get_oriented_bounding_box_from_3d_points).
    """

    def __init__(
        self,
        api: Any,
        instruction: str,
        *,
        trace_dir: str | pathlib.Path | None = None,
        max_physical_ops: int = 30,
        env_check: Callable[[], bool] | None = None,
    ) -> None:
        """
        :param env_check: Optional privileged success verdict (e.g. LIBERO's
            ``env.task_completed``). Called once when the episode ends and
            written to ``meta.json`` only — never into a receipt or prompt.
            It is the ground truth that trace filtering (M4) and the
            claimed-vs-actual gap metric need; ``claimed_success`` is just the
            agent's own belief.
        """
        self.api = api
        self.state = ActionState(instruction=instruction)
        self.camera_name: str = getattr(api, "camera_name", "agentview")
        self.wrist_camera_name: str = getattr(api, "wrist_camera_name", "robot0_eye_in_hand")
        self.obs: dict[str, Any] | None = None
        self._cloud: SceneCloud | None = None
        self.finished = False
        self.claimed_success = False
        self.env_success: bool | None = None
        self.max_physical_ops = max_physical_ops
        self._physical_ops = 0
        self._env_check = env_check
        self.trace = TraceLogger(trace_dir) if trace_dir else None
        if self.trace:
            self.trace.log_meta({"instruction": instruction})

    # ------------------------------------------------------------------ #
    def refresh_observation(self) -> dict[str, Any]:
        self.obs = self.api.get_observation()
        self.state.bump_revision()
        # Proprioception into the state: the canvas, the panel and (later) the
        # held-object test all read the arm from there, never from raw obs.
        cart = np.asarray(self.obs.get("robot_cartesian_pos", []), dtype=np.float64)
        if cart.size >= 8:
            self.state.ee_pose = Pose(cart[:3], cart[3:7])
            self.state.gripper_opening = float(cart[7])
            self.state.gripper_open = bool(cart[7] > 0.5)
        joints = np.asarray(self.obs.get("robot_joint_pos", []), dtype=np.float64).reshape(-1)
        if joints.size >= 7 and np.isfinite(joints[:7]).all():
            self.state.arm_joint_positions_rad = joints[:7].copy()
        else:
            self.state.arm_joint_positions_rad = None
        self._cloud = None
        return self.obs

    def scene_cloud(self) -> SceneCloud:
        """Fused world-frame cloud for the current observation, cached.

        One cloud per observation serves the main view, the focus inset and the
        object tints; rebuilding it per render would triple the cost of a step
        that changed nothing.
        """
        if self._cloud is None or self._cloud.revision != self.state.obs_revision:
            self._cloud = build_scene_cloud(
                self.obs,
                (self.camera_name, self.wrist_camera_name),
                revision=self.state.obs_revision,
            )
        return self._cloud

    def render(self) -> np.ndarray:
        return render_canvas(
            self.state,
            self.obs,
            camera_name=self.camera_name,
            wrist_camera_name=self.wrist_camera_name,
            cloud=self.scene_cloud(),
        )

    # ------------------------------------------------------------------ #
    def step(self, op_name: str, **kwargs: Any) -> StepResult:
        """Execute one workspace operation and return (canvas, receipt, state).

        Op failures (bad ids, empty tool results) come back as agent-visible
        error receipts, never exceptions: recovery is the agent's job.
        """
        spec = OPS.get(op_name)
        if spec is None:
            return self._result(False, op_name, kwargs, f"unknown op '{op_name}'; available: {sorted(OPS)}")
        if self.finished:
            return self._result(False, op_name, kwargs, "episode already ended")
        if spec.physical and op_name != "done" and self._physical_ops >= self.max_physical_ops:
            return self._result(False, op_name, kwargs, "physical operation budget exhausted; call done")
        if self.obs is None and op_name != "observe":
            self.refresh_observation()

        try:
            receipt_text = spec.fn(self, **kwargs)
            ok = True
        except (OpError, KeyError, ValueError, TypeError) as exc:
            receipt_text = f"ERROR: {exc}"
            ok = False
            self.state.log(f"{op_name} failed: {exc}")
        if ok and spec.physical:
            self._physical_ops += 1
        if self.finished:
            self._settle_episode()

        return self._result(ok, op_name, kwargs, receipt_text, physical=spec.physical)

    def _settle_episode(self) -> None:
        """Record the episode verdict once, at the moment ``done`` lands.

        The env check is read here and not earlier so it sees the final world
        state, and the result goes to ``meta.json`` only: feeding it into a
        receipt would leak privileged state into the agent's context and the
        training data.
        """
        if self._env_check is not None and self.env_success is None:
            try:
                self.env_success = bool(self._env_check())
            except Exception as exc:  # a broken checker must not kill the trace
                self.state.log(f"env_check failed: {exc}")
        if self.trace:
            self.trace.log_meta(
                {
                    "instruction": self.state.instruction,
                    "claimed_success": self.claimed_success,
                    "env_success": self.env_success,
                    "physical_ops": self._physical_ops,
                }
            )

    def reject(self, op_name: str, args: dict[str, Any], reason: str) -> StepResult:
        """Record an action that never reached an op: unparseable arguments, an
        unknown op name, a protocol violation.

        It goes through the trace like any other step. The agent's malformed
        output and how it recovers are part of the episode; dropping them would
        train the student on a world where it never makes protocol mistakes.
        """
        if self.obs is None:
            self.refresh_observation()
        return self._result(False, op_name or "invalid", args, f"ERROR: {reason}")

    def _result(
        self,
        ok: bool,
        op_name: str,
        args: dict[str, Any],
        receipt_text: str,
        *,
        physical: bool = False,
    ) -> StepResult:
        result = StepResult(
            ok=ok,
            op=op_name,
            args={k: v for k, v in args.items()},
            receipt_text=receipt_text,
            canvas=self.render(),
            state_summary=self.state.summary(),
            physical=physical,
        )
        if self.trace:
            self.trace.log(result)
        return result
