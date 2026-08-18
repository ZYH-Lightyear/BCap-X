"""Revision lifecycle and dispatch for the VAW Context Runtime."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime.errors import ContextFunctionError
from vaw.context_runtime.functions import ContextFunctions
from vaw.context_runtime.geometry import (
    DEFAULT_TCP_TO_HAND_LOCAL_XYZ,
    tcp_position_from_hand_pose,
)
from vaw.context_runtime.memory import record_physical_transaction
from vaw.context_runtime.model import (
    ActionSeed,
    ContextState,
    PointEvidence,
    Pose,
    RefinementSession,
    RegionEvidence,
    RobotState,
)
from vaw.context_runtime.motion import (
    MotionBackend,
    MotionBackendError,
    MotionPlan,
    PyrokiMotionBackend,
    create_motion_backend,
)
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
    # PyRoki is useful for genuinely local Cartesian corrections.  A small
    # *edit* to a far-away proposal is still a large robot motion and must stay
    # on the collision-aware coarse planner.
    LOCAL_TRANSLATION_LIMIT_M = 0.06
    LOCAL_ROTATION_LIMIT_DEG = 20.0
    # Main may execute simple local controls directly.  ``commit`` is reserved
    # for a spatial action that Imagination explicitly returned as ready.
    PHYSICAL_FUNCTIONS = frozenset(
        {"delta_move", "open_gripper", "close_gripper", "commit"}
    )

    def __init__(
        self,
        api: Any,
        task_prompt: str,
        *,
        motion_backend: str | MotionBackend = "curobo",
        local_motion_backend: str | MotionBackend | None = None,
        tcp_to_hand_local_xyz: tuple[float, float, float] | None = None,
        semantic_rgb_provider: Callable[[], np.ndarray] | None = None,
        contact_camera_provider: Any | None = None,
        opposite_scene_camera_provider: Any | None = None,
    ) -> None:
        self.api = api
        self.camera_name = str(getattr(api, "camera_name", "agentview"))
        self.wrist_camera_name = str(getattr(api, "wrist_camera_name", "robot0_eye_in_hand"))
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
        self._tcp_to_hand_local_xyz = _finite_vector3(backend_offset, "tcp_to_hand_local_xyz")
        if local_motion_backend is None:
            self.local_motion = (
                PyrokiMotionBackend(
                    api,
                    public_tcp_to_hand_local_xyz=tuple(self._tcp_to_hand_local_xyz),
                )
                if isinstance(motion_backend, str)
                and str(motion_backend).strip().lower() == "curobo"
                else self.motion
            )
        else:
            self.local_motion = (
                PyrokiMotionBackend(
                    api,
                    public_tcp_to_hand_local_xyz=tuple(self._tcp_to_hand_local_xyz),
                )
                if isinstance(local_motion_backend, str)
                and local_motion_backend.strip().lower() == "pyroki"
                else create_motion_backend(local_motion_backend, api)
                if isinstance(local_motion_backend, str)
                else local_motion_backend
            )
        self.local_motion_backend_name = str(self.local_motion.name)
        self._motion_executors = {
            str(self.motion.name): self.motion,
            str(self.local_motion.name): self.local_motion,
        }
        self._semantic_rgb_provider = semantic_rgb_provider
        self.state = ContextState(task_prompt=task_prompt)
        self._private = PrivateEnvContext(
            contact_camera_provider=contact_camera_provider,
            opposite_scene_camera_provider=opposite_scene_camera_provider,
        )
        self.finished = False
        self.claimed_success = False
        self._functions = ContextFunctions(self)
        self.refresh_observation()

    def execute_motion_plan(self, plan: MotionPlan, target: Pose) -> None:
        """Execute through the backend that produced the cached plan."""

        backend = self._motion_executors.get(str(plan.backend))
        if backend is None:
            raise MotionBackendError(
                f"no executor is registered for motion plan backend '{plan.backend}'"
            )
        backend.execute(plan, target)

    def motion_backend_for_target(
        self,
        target: Pose,
    ) -> tuple[MotionBackend, dict[str, Any]]:
        """Route by the full observed-TCP-to-target displacement.

        Imagination edits are expressed relative to a virtual target, so the
        latest edit magnitude is not a valid proxy for execution distance.
        Keep distant or substantially rotated targets on the coarse planner;
        use the local planner only after the real robot is already near the
        requested pose.
        """

        robot = self.state.robot
        current = robot.tcp_pose if robot is not None else None
        if current is None or self.local_motion is self.motion:
            return self.motion, {
                "backend": str(self.motion.name),
                "reason": "single_backend" if current is not None else "tcp_unavailable",
            }

        current_position = np.asarray(current.position_xyz, dtype=np.float64)
        target_position = np.asarray(target.position_xyz, dtype=np.float64)
        translation_m = float(np.linalg.norm(target_position - current_position))
        current_rotation = Rotation.from_quat(
            np.asarray(current.quaternion_xyzw, dtype=np.float64)
        )
        target_rotation = Rotation.from_quat(
            np.asarray(target.quaternion_xyzw, dtype=np.float64)
        )
        rotation_deg = float(
            np.rad2deg((current_rotation.inv() * target_rotation).magnitude())
        )
        use_local = (
            translation_m <= self.LOCAL_TRANSLATION_LIMIT_M
            and rotation_deg <= self.LOCAL_ROTATION_LIMIT_DEG
        )
        backend = self.local_motion if use_local else self.motion
        return backend, {
            "backend": str(backend.name),
            "translation_m": round(translation_m, 6),
            "rotation_deg": round(rotation_deg, 3),
            "local_translation_limit_m": self.LOCAL_TRANSLATION_LIMIT_M,
            "local_rotation_limit_deg": self.LOCAL_ROTATION_LIMIT_DEG,
            "reason": "local_target" if use_local else "coarse_target",
        }

    def begin_refinement(self, instruction: str, action_id: str) -> str:
        """Open one synchronous Imagination task without yielding Main ownership."""

        if self.state.refinement is not None:
            raise ContextFunctionError("a refinement task is already active")
        normalized = " ".join(str(instruction).split())
        if not normalized:
            raise ContextFunctionError("instruction must not be empty")
        action = self.state.pending_action
        if action is None or action.action_id != action_id:
            raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
        # A Contact Camera pair is selected once at refinement entry and then
        # remains fixed while Imagination edits the virtual target.  Re-entering
        # refinement intentionally performs a fresh visibility selection.
        self._private.clear_contact_camera_lock()
        self.state.refinement = RefinementSession(
            action_id=action.action_id,
            instruction=normalized,
            initial_target=action.target,
            created_action=False,
        )
        return action.action_id

    def discard_pending_action(self) -> None:
        """Discard the current virtual action without changing the real world."""

        self.state.pending_action = None
        self.state.refinement = None
        self._private.action_artifacts = None
        self._private.clear_contact_camera_lock()

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
        pending_intent = (
            self.state.pending_action.intent
            if self.state.pending_action is not None
            else None
        )
        previous_physical = self.state.last_physical_action
        self._private.begin_function_call()
        handler = self._functions.main_handlers.get(function_name)
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
        return self._record(
            function_name,
            arguments,
            result,
            before,
            pending_intent=pending_intent,
            previous_physical=previous_physical,
        )

    def execute_imagination(
        self, function_name: str, **arguments: Any
    ) -> ContextStepResult:
        """Dispatch one editor on the private Imagination tool surface."""

        before = self.state.observation_revision
        self._private.begin_function_call()
        handler = self._functions.imagination_handlers.get(function_name)
        if handler is None:
            result: dict[str, Any] = {
                "error": f"unknown imagination function '{function_name}'"
            }
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
            name, arguments = parse_action(payload, allowed=tuple(self._functions.main_handlers))
        except ValueError as exc:
            shown_payload = payload if isinstance(payload, str) else dict(payload)
            return self.reject("invalid", {"payload": shown_payload}, str(exc))
        return self.execute(name, **arguments)

    def limit_refinement(self) -> ContextStepResult:
        """Fail the active Imagination task when its private budget is spent."""

        before = self.state.observation_revision
        self._private.begin_function_call()
        try:
            result = self._functions.limit_imagination()
        except ContextFunctionError as exc:
            result = {"error": str(exc)}
        return self._record("refinement_limit", {}, result, before)

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
        *,
        pending_intent: str | None = None,
        previous_physical: Any | None = None,
    ) -> ContextStepResult:
        revision_after = self.state.observation_revision
        physical = self.state.last_physical_action
        if (
            function_name in self.PHYSICAL_FUNCTIONS
            and physical is not None
            and physical is not previous_physical
        ):
            record_physical_transaction(
                self.state.task_memory,
                function_name=function_name,
                arguments=arguments,
                intent=pending_intent,
                physical_outcome=physical.outcome,
            )
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
                dict(self._private.trace_diagnostics) if self._private.trace_diagnostics else None
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

    def _semantic_rgb(self) -> np.ndarray:
        """Return one cached, revision-local RGB used only for semantic grounding."""

        cached = self._private.semantic_rgb
        if cached is not None:
            return cached
        if self._semantic_rgb_provider is None:
            image = np.asarray(self._camera()["images"]["rgb"])
        else:
            try:
                image = np.asarray(self._semantic_rgb_provider())
            except Exception as exc:
                raise ContextFunctionError(f"semantic RGB capture failed: {exc}") from exc
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ContextFunctionError(f"semantic RGB must have shape (H,W,3/4), got {image.shape}")
        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ContextFunctionError("semantic RGB is empty")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
                raise ContextFunctionError("semantic RGB contains non-finite values")
            image = (
                np.clip(image * 255.0, 0.0, 255.0)
                if float(np.max(image)) <= 1.0
                else np.clip(image, 0.0, 255.0)
            ).astype(np.uint8)
        image = np.ascontiguousarray(image[:, :, :3])
        self._private.semantic_rgb = image
        return image

    def _semantic_crop(self, within_region_id: str | None) -> np.ndarray:
        """Crop the semantic view to the same observed region as the RGB-D view."""

        semantic = self._semantic_rgb()
        if within_region_id is None:
            return semantic
        observed = np.asarray(self._camera()["images"]["rgb"])
        observed_crop, (left, top) = self._crop_rgb(observed, within_region_id)
        right = left + observed_crop.shape[1]
        bottom = top + observed_crop.shape[0]
        scale_x = semantic.shape[1] / observed.shape[1]
        scale_y = semantic.shape[0] / observed.shape[0]
        semantic_left = max(0, int(np.floor(left * scale_x)))
        semantic_top = max(0, int(np.floor(top * scale_y)))
        semantic_right = min(semantic.shape[1], int(np.ceil(right * scale_x)))
        semantic_bottom = min(semantic.shape[0], int(np.ceil(bottom * scale_y)))
        crop = semantic[semantic_top:semantic_bottom, semantic_left:semantic_right]
        if crop.size == 0:
            raise ContextFunctionError(f"region '{within_region_id}' has an empty semantic crop")
        return crop

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
    cartesian = np.asarray(observation.get("robot_cartesian_pos", []), dtype=np.float64).reshape(-1)
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
    joints_array = np.asarray(observation.get("robot_joint_pos", []), dtype=np.float64).reshape(-1)
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
