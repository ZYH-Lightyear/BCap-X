"""Revision lifecycle and dispatch for the VAW Context Runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from vaw.context_runtime.errors import ContextFunctionError
from vaw.context_runtime.functions import ContextFunctions
from vaw.context_runtime.geometry import (
    DEFAULT_TCP_TO_HAND_LOCAL_XYZ,
    tcp_position_from_hand_pose,
)
from vaw.context_runtime.model import (
    ActionSeed,
    ContextState,
    PointEvidence,
    Pose,
    RegionEvidence,
    RobotState,
)
from vaw.context_runtime.motion import MotionBackend, create_motion_backend
from vaw.context_runtime.private import PresentationEvent, PrivateEnvContext
from vaw.context_runtime.protocol import parse_action


@dataclass(frozen=True)
class ContextStepResult:
    function_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    revision_before: int
    revision_after: int
    manifest: dict[str, Any]
    trace_diagnostics: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return "error" not in self.result


class ContextWorkspace:
    """Own one episode's public state, private sensors and function dispatcher."""

    MAX_GRASP_CANDIDATES = 5
    # Waypoint editors never mutate the environment.  Runtime physical-op
    # accounting and environment-success checks share this single boundary.
    PHYSICAL_FUNCTIONS = frozenset({"commit"})

    def __init__(
        self,
        api: Any,
        task_prompt: str,
        *,
        motion_backend: str | MotionBackend = "curobo",
        tcp_to_hand_local_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        self.api = api
        self.camera_name = str(getattr(api, "camera_name", "agentview"))
        self.wrist_camera_name = str(
            getattr(api, "wrist_camera_name", "robot0_eye_in_hand")
        )
        self.motion = (
            create_motion_backend(motion_backend, api)
            if isinstance(motion_backend, str)
            else motion_backend
        )
        self.motion_backend_name = str(self.motion.name)
        backend_offset = (
            getattr(
                self.motion,
                "tcp_to_hand_local_xyz",
                getattr(api, "_TCP_OFFSET", DEFAULT_TCP_TO_HAND_LOCAL_XYZ),
            )
            if tcp_to_hand_local_xyz is None
            else tcp_to_hand_local_xyz
        )
        self._tcp_to_hand_local_xyz = _finite_vector3(
            backend_offset, "tcp_to_hand_local_xyz"
        )
        self.state = ContextState(task_prompt=task_prompt)
        self._private = PrivateEnvContext()
        self.finished = False
        self.claimed_success = False
        self._functions = ContextFunctions(self)
        self.refinement_goal = task_prompt
        self.refresh_observation()

    def set_refinement_goal(self, text: str) -> None:
        """Set the session-local goal used if the next call starts imagination."""

        normalized = " ".join(str(text).split())
        self.refinement_goal = normalized or f"为任务“{self.state.task_prompt}”检查并调整动作"

    def refresh_observation(self) -> dict[str, Any]:
        observation = self._call_backend("get_observation")
        if not isinstance(observation, dict):
            raise ContextFunctionError("get_observation did not return a mapping")
        revision = self.state.begin_revision()
        self._private.begin_revision(observation)
        self.state.robot = _robot_state(
            observation,
            revision,
            self._tcp_to_hand_local_xyz,
        )
        return observation

    def execute(self, function_name: str, **arguments: Any) -> ContextStepResult:
        before = self.state.observation_revision
        self._private.begin_function_call()
        handler = self._functions.handlers.get(function_name)
        if handler is None:
            result: dict[str, Any] = {"error": f"unknown function '{function_name}'"}
        elif self.finished:
            result = {"error": "episode already ended"}
        else:
            try:
                inspect.signature(handler).bind(**arguments)
            except TypeError as exc:
                result = {"error": str(exc)}
            else:
                try:
                    result = handler(**arguments)
                except ContextFunctionError as exc:
                    result = {"error": str(exc)}
        return self._record(function_name, arguments, result, before)

    def execute_action(self, payload: str | dict[str, Any]) -> ContextStepResult:
        try:
            name, arguments = parse_action(payload)
        except ValueError as exc:
            shown_payload = payload if isinstance(payload, str) else dict(payload)
            return self.reject("invalid", {"payload": shown_payload}, str(exc))
        return self.execute(name, **arguments)

    def limit_imagination(self) -> ContextStepResult:
        """Return control to Main when the Imagination turn budget is spent."""

        before = self.state.observation_revision
        self._private.begin_function_call()
        try:
            result = self._functions.limit_imagination()
        except ContextFunctionError as exc:
            result = {"error": str(exc)}
        return self._record("imagination_limit", {}, result, before)

    def reject(
        self,
        function_name: str,
        arguments: dict[str, Any],
        error: str,
    ) -> ContextStepResult:
        """Record a protocol-level failure without dispatching a function."""

        self._private.begin_function_call()
        return self._record(
            function_name,
            arguments,
            {"error": str(error)},
            self.state.observation_revision,
        )

    def _record(
        self,
        function_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        revision_before: int,
    ) -> ContextStepResult:
        revision_after = self.state.observation_revision
        self._private.presentation_event = PresentationEvent(
            function_name=function_name,
            result=dict(result),
            error=str(result["error"]) if "error" in result else None,
        )
        return ContextStepResult(
            function_name=function_name,
            arguments=dict(arguments),
            result=result,
            revision_before=revision_before,
            revision_after=revision_after,
            manifest=self.state.manifest(),
            trace_diagnostics=(
                dict(self._private.trace_diagnostics)
                if self._private.trace_diagnostics
                else None
            ),
        )

    def _call_backend(self, name: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.api, name, None)
        if not callable(function):
            raise ContextFunctionError(f"backend function '{name}' is unavailable")
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # external service/controller boundary
            raise ContextFunctionError(f"{name} failed: {exc}") from exc

    def _camera(self) -> dict[str, Any]:
        try:
            return self._private.camera(self.camera_name)
        except RuntimeError as exc:
            raise ContextFunctionError(str(exc)) from exc

    def _crop_rgb(
        self,
        rgb: np.ndarray,
        within_region_id: str | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if within_region_id is None:
            return rgb, (0, 0)
        region = self._current_region(within_region_id)
        x1, y1, x2, y2 = region.bbox_xyxy_px
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(rgb.shape[1], int(np.ceil(x2)))
        bottom = min(rgb.shape[0], int(np.ceil(y2)))
        if right <= left or bottom <= top:
            raise ContextFunctionError(f"region '{within_region_id}' has an empty crop")
        return rgb[top:bottom, left:right], (left, top)

    def _current_region(self, region_id: str) -> RegionEvidence:
        region = self.state.regions.get(region_id)
        if region is None or region.source_revision != self.state.observation_revision:
            raise ContextFunctionError(f"unknown or expired region_id '{region_id}'")
        return region

    def _current_point(self, point_id: str) -> PointEvidence:
        point = self.state.points.get(point_id)
        if point is None or point.source_revision != self.state.observation_revision:
            raise ContextFunctionError(f"unknown or expired point_id '{point_id}'")
        return point

    def _current_seed(self, seed_id: str) -> ActionSeed:
        seed = self.state.seeds.get(seed_id)
        if seed is None or seed.source_revision != self.state.observation_revision:
            raise ContextFunctionError(f"unknown or expired seed_id '{seed_id}'")
        return seed


def _robot_state(
    observation: dict[str, Any],
    revision: int,
    tcp_to_hand_local_xyz: np.ndarray,
) -> RobotState:
    cartesian = np.asarray(
        observation.get("robot_cartesian_pos", []), dtype=np.float64
    ).reshape(-1)
    ee_pose = None
    tcp_pose = None
    opening = None
    if cartesian.size >= 7 and np.isfinite(cartesian[:7]).all():
        ee_pose = Pose(
            tuple(float(value) for value in cartesian[:3]),
            tuple(float(value) for value in np.roll(cartesian[3:7], -1)),
        )
        try:
            tcp_position = tcp_position_from_hand_pose(
                ee_pose.position_xyz,
                ee_pose.quaternion_xyzw,
                tcp_to_hand_local_xyz,
            )
        except ValueError:
            tcp_pose = None
        else:
            tcp_pose = Pose(
                tuple(float(value) for value in tcp_position),
                ee_pose.quaternion_xyzw,
            )
    if cartesian.size >= 8 and np.isfinite(cartesian[7]):
        opening = float(cartesian[7])
    joints_array = np.asarray(
        observation.get("robot_joint_pos", []), dtype=np.float64
    ).reshape(-1)
    joints = (
        tuple(float(value) for value in joints_array[:7])
        if joints_array.size >= 7 and np.isfinite(joints_array[:7]).all()
        else None
    )
    return RobotState(
        ee_pose=ee_pose,
        tcp_pose=tcp_pose,
        joint_positions_rad=joints,
        gripper_opening=opening,
        source_revision=revision,
    )


def _finite_vector3(values: Any, label: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain three numbers") from exc
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain finite numbers")
    return vector.copy()


__all__ = ["ContextFunctionError", "ContextStepResult", "ContextWorkspace"]
