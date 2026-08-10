"""Private motion backends for the VAW Context Runtime.

The policy never calls these objects directly.  They translate an
``ActionTarget`` pose into either the legacy one-waypoint PyRoki command or
a collision-aware trajectory produced by CaP-X's shared CuRobo planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

from vaw.context_runtime.model import ActionPrediction, Pose

PYROKI_TCP_TO_HAND_LOCAL_XYZ = (0.0, 0.0, -0.1)
# Must match capx.integrations.motion.curobo_api.FRANKA_HAND_TO_FINGERTIP_Z_M.
# Importing that module here would eagerly initialise CuRobo/warp in offline tests.
CUROBO_TCP_TO_HAND_LOCAL_XYZ = (0.0, 0.0, -(0.0584 * 2.0))


class MotionBackendError(RuntimeError):
    """An expected planner/controller boundary failure."""


@dataclass(frozen=True)
class MotionPlan:
    """Episode-private plan associated with an imagined action target."""

    backend: str
    prediction: ActionPrediction
    trajectory_rad: np.ndarray | None = None
    goalset_index: int | None = None

    def __post_init__(self) -> None:
        trajectory = self.trajectory_rad
        if trajectory is None:
            return
        values = np.asarray(trajectory, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 7:
            raise ValueError(f"motion trajectory must be (T, 7), got {values.shape}")
        values = values[:, :7].copy()
        if not np.isfinite(values).all():
            raise ValueError("motion trajectory must contain finite joint values")
        values.setflags(write=False)
        object.__setattr__(self, "trajectory_rad", values)


class MotionBackend(Protocol):
    """Internal interface shared by PyRoki and CuRobo implementations."""

    name: str
    tcp_to_hand_local_xyz: tuple[float, float, float]

    def preview(self, target: Pose) -> ActionPrediction: ...

    def plan_pose(
        self,
        target: Pose,
        *,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan: ...

    def plan_grasp(
        self,
        target: Pose,
        *,
        object_name: str,
        object_mask: np.ndarray,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan: ...

    def execute(self, plan: MotionPlan, target: Pose) -> None: ...


class PyrokiMotionBackend:
    """Compatibility backend matching the pre-CuRobo M1.3 behaviour."""

    name = "pyroki"

    def __init__(self, api: Any) -> None:
        self.api = api
        offset = getattr(api, "_TCP_OFFSET", PYROKI_TCP_TO_HAND_LOCAL_XYZ)
        self.tcp_to_hand_local_xyz = tuple(
            float(value) for value in _vector(offset, 3, "PyRoki TCP offset")
        )

    def preview(self, target: Pose) -> ActionPrediction:
        return _solve_ik_prediction(self.api, target)

    def plan_pose(
        self,
        target: Pose,
        *,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan:
        prediction = preview or self.preview(target)
        trajectory = _one_waypoint(prediction)
        return MotionPlan(self.name, prediction, trajectory)

    def plan_grasp(
        self,
        target: Pose,
        *,
        object_name: str,
        object_mask: np.ndarray,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan:
        del object_name, object_mask
        return self.plan_pose(target, preview=preview)

    def execute(self, plan: MotionPlan, target: Pose) -> None:
        trajectory = plan.trajectory_rad
        # Preserve M1.3's useful retry semantics when proposal-time IK was
        # unavailable, without changing CuRobo's exact-plan contract.
        if trajectory is None:
            retry = self.preview(target)
            trajectory = _one_waypoint(retry)
        if trajectory is None:
            detail = plan.prediction.detail or "solve_ik returned no joints"
            raise MotionBackendError(detail)
        _call(self.api, "move_to_joints", trajectory[-1])


class CuroboMotionBackend:
    """Adapter over the shared CaP-X LIBERO CuRobo planner."""

    name = "curobo"
    tcp_to_hand_local_xyz = CUROBO_TCP_TO_HAND_LOCAL_XYZ

    def __init__(
        self,
        api: Any,
        *,
        use_world_collision: bool = True,
        trajectory_subsample: int = 2,
        waypoint_tolerance_rad: float = 0.01,
        max_steps_per_waypoint: int = 120,
        final_joint_tolerance_rad: float = 0.02,
    ) -> None:
        if trajectory_subsample < 1:
            raise ValueError("trajectory_subsample must be at least one")
        self.api = api
        self.use_world_collision = bool(use_world_collision)
        self.trajectory_subsample = int(trajectory_subsample)
        self.waypoint_tolerance_rad = float(waypoint_tolerance_rad)
        self.max_steps_per_waypoint = int(max_steps_per_waypoint)
        self.final_joint_tolerance_rad = float(final_joint_tolerance_rad)

    def preview(self, target: Pose) -> ActionPrediction:
        # Candidate previews remain cheap and explicitly unchecked.  Selected
        # actions are re-planned from the observed joints by CuRobo below.
        return _solve_ik_prediction(self.api, target)

    def plan_pose(
        self,
        target: Pose,
        *,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan:
        del preview
        try:
            world = _call(self.api, "update_curobo_world")
            return self._plan(
                target,
                object_name="vaw_pose_target",
                object_mask=np.zeros((1, 1), dtype=bool),
                world_config=world,
            )
        except MotionBackendError as exc:
            return _failed_plan(self.name, exc)

    def plan_grasp(
        self,
        target: Pose,
        *,
        object_name: str,
        object_mask: np.ndarray,
        preview: ActionPrediction | None = None,
    ) -> MotionPlan:
        del preview
        try:
            return self._plan(
                target,
                object_name=object_name,
                object_mask=np.asarray(object_mask, dtype=bool),
                world_config=None,
            )
        except MotionBackendError as exc:
            return _failed_plan(self.name, exc)

    def _plan(
        self,
        target: Pose,
        *,
        object_name: str,
        object_mask: np.ndarray,
        world_config: Any,
    ) -> MotionPlan:
        _ensure_warp_torch_compat()
        position = np.asarray(target.position_xyz, dtype=np.float64)
        quaternion_wxyz = _xyzw_to_wxyz(target.quaternion_xyzw)
        result = _call(
            self.api,
            "plan_grasp_trajectory",
            object_name,
            object_mask=object_mask,
            grasp_poses=[(position, quaternion_wxyz)],
            top_k_grasps=1,
            use_world_collision=self.use_world_collision,
            world_config=world_config,
            grasp_pose_is_fingertip=True,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise MotionBackendError(
                "plan_grasp_trajectory returned an invalid result"
            )
        success, trajectory, goalset_index = result
        if not success or trajectory is None:
            raise MotionBackendError("CuRobo found no collision-free trajectory")
        try:
            values = np.asarray(trajectory, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            final_joints = _joint_tuple(values[-1])
            plan = MotionPlan(
                backend=self.name,
                prediction=ActionPrediction(
                    solve_ik="returned",
                    joint_positions_rad=final_joints,
                    trajectory_checked=True,
                    collision_checked=self.use_world_collision,
                ),
                trajectory_rad=values,
                goalset_index=(
                    int(goalset_index) if goalset_index is not None else None
                ),
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise MotionBackendError(f"CuRobo returned an invalid trajectory: {exc}") from exc
        return plan

    def execute(self, plan: MotionPlan, target: Pose) -> None:
        del target
        trajectory = plan.trajectory_rad
        if trajectory is None:
            detail = plan.prediction.detail or "CuRobo proposal has no cached trajectory"
            raise MotionBackendError(detail)
        status = _call(
            self.api,
            "execute_joint_trajectory",
            trajectory,
            subsample=self.trajectory_subsample,
            tolerance=self.waypoint_tolerance_rad,
            max_steps=self.max_steps_per_waypoint,
        )
        if isinstance(status, dict) and status.get("all_converged") is False:
            raise MotionBackendError("execute_joint_trajectory did not converge")

        observation = _call(self.api, "get_observation")
        if not isinstance(observation, dict):
            raise MotionBackendError("get_observation returned no joint state")
        achieved = _vector(
            observation.get("robot_joint_pos", []),
            minimum_length=7,
            label="observed robot joints",
        )[:7]
        residual = float(np.linalg.norm(achieved - trajectory[-1]))
        if residual > self.final_joint_tolerance_rad:
            raise MotionBackendError(
                "CuRobo trajectory execution did not converge: "
                f"joint residual {residual:.6f} rad exceeds "
                f"{self.final_joint_tolerance_rad:.6f} rad"
            )


def create_motion_backend(name: str, api: Any) -> MotionBackend:
    """Create one of the two supported private motion backends."""

    normalized = str(name).strip().lower()
    if normalized == "pyroki":
        return PyrokiMotionBackend(api)
    if normalized == "curobo":
        return CuroboMotionBackend(api)
    raise ValueError(f"unknown motion backend '{name}'")


def _solve_ik_prediction(api: Any, target: Pose) -> ActionPrediction:
    try:
        solved = _call(
            api,
            "solve_ik",
            np.asarray(target.position_xyz, dtype=np.float64),
            _xyzw_to_wxyz(target.quaternion_xyzw),
            return_info=True,
        )
        if not isinstance(solved, tuple) or len(solved) != 2:
            raise MotionBackendError(
                "solve_ik did not report whether the requested orientation was used"
            )
        joints, info = solved
        if not isinstance(info, dict):
            raise MotionBackendError("solve_ik returned invalid orientation metadata")
        orientation_used = str(info.get("orientation_used", ""))
        if orientation_used != "requested":
            raise MotionBackendError(
                "solve_ik substituted orientation "
                f"'{orientation_used or 'unknown'}' for the requested candidate pose"
            )
        values = _joint_tuple(joints)
    except MotionBackendError as exc:
        return ActionPrediction(solve_ik="error", detail=str(exc))
    return ActionPrediction(solve_ik="returned", joint_positions_rad=values)


def _failed_plan(backend: str, error: Exception) -> MotionPlan:
    return MotionPlan(
        backend=backend,
        prediction=ActionPrediction(solve_ik="error", detail=str(error)),
    )


def _one_waypoint(prediction: ActionPrediction) -> np.ndarray | None:
    joints = prediction.joint_positions_rad
    return None if joints is None else np.asarray([joints], dtype=np.float64)


def _joint_tuple(values: Any) -> tuple[float, ...]:
    joints = _vector(values, 7, "joint result")
    return tuple(float(value) for value in joints)


def _xyzw_to_wxyz(quaternion_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    return np.roll(np.asarray(quaternion_xyzw, dtype=np.float64), 1)


def _vector(
    values: Any,
    length: int | None = None,
    label: str = "vector",
    *,
    minimum_length: int | None = None,
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MotionBackendError(f"{label} is not numeric") from exc
    if length is not None and vector.size != length:
        raise MotionBackendError(f"{label} must contain {length} values")
    if minimum_length is not None and vector.size < minimum_length:
        raise MotionBackendError(
            f"{label} must contain at least {minimum_length} values"
        )
    if not np.isfinite(vector).all():
        raise MotionBackendError(f"{label} must contain finite values")
    return vector.copy()


def _call(api: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(api, name, None)
    if not callable(function):
        raise MotionBackendError(f"backend function '{name}' is unavailable")
    try:
        return function(*args, **kwargs)
    except MotionBackendError:
        raise
    except Exception as exc:
        raise MotionBackendError(f"{name} failed: {exc}") from exc


def _ensure_warp_torch_compat() -> None:
    """Bridge the single legacy ``wp.torch`` lookup used by bundled CuRobo.

    warp-lang 1.15 moved its torch interop helpers to top-level attributes,
    while the bundled CuRobo world-mesh checker still accesses
    ``wp.torch.device_from_torch``.  Keep the compatibility local to the VAW
    CuRobo path so neither CaP-X nor the vendored dependency is modified.
    """

    try:
        import warp as wp
    except ModuleNotFoundError:
        return
    if hasattr(wp, "torch"):
        return
    device_from_torch = getattr(wp, "device_from_torch", None)
    if callable(device_from_torch):
        wp.torch = SimpleNamespace(device_from_torch=device_from_torch)


__all__ = [
    "CUROBO_TCP_TO_HAND_LOCAL_XYZ",
    "CuroboMotionBackend",
    "MotionBackend",
    "MotionBackendError",
    "MotionPlan",
    "PyrokiMotionBackend",
    "create_motion_backend",
]
