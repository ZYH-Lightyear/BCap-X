"""Episode-private sensor, planning and presentation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime.model import ActionTarget, Pose

if TYPE_CHECKING:
    from vaw.context_runtime.motion import MotionPlan


@dataclass
class RegionGeometryArtifact:
    agentview_mask: np.ndarray
    wrist_mask: np.ndarray | None
    object_points_base: np.ndarray
    scene_points_base: np.ndarray
    filtered_object_points_base: np.ndarray


@dataclass(frozen=True)
class PlanningContext:
    source_kind: str
    source_ref: str | None = None
    region_id: str | None = None


@dataclass(frozen=True)
class VisualEdit:
    kind: str
    frame: str | None = None
    reference_pose: Pose | None = None
    delta_xyz_m: tuple[float, float, float] | None = None
    axis: str | None = None
    angle_deg: float | None = None
    gripper_target: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.frame is not None:
            result["frame"] = self.frame
        if self.reference_pose is not None:
            result["reference_pose"] = self.reference_pose.summary()
        if self.delta_xyz_m is not None:
            result["delta_xyz_m"] = [round(float(v), 6) for v in self.delta_xyz_m]
        if self.axis is not None:
            result["axis"] = self.axis
        if self.angle_deg is not None:
            result["angle_deg"] = round(float(self.angle_deg), 6)
        if self.gripper_target is not None:
            result["gripper_target"] = self.gripper_target
        return result

    def command_summary(self) -> dict[str, Any]:
        """Return only the command delta needed to reason about edit history."""

        result = self.summary()
        result.pop("reference_pose", None)
        return result


@dataclass(frozen=True)
class SeedArtifacts:
    planning_context: PlanningContext | None = None
    preview_plan: MotionPlan | None = None


@dataclass(frozen=True)
class ImaginationArtifacts:
    planning_context: PlanningContext | None = None
    preview_plan: MotionPlan | None = None
    initial_target: ActionTarget | None = None
    previous_visual_edit: VisualEdit | None = None
    latest_visual_edit: VisualEdit | None = None
    turn_count: int = 0


@dataclass(frozen=True)
class ActionReviewArtifacts:
    motion_plan: MotionPlan | None = None
    planning_context: PlanningContext | None = None
    termination_reason: str = "agent_ready"
    # Exact command-state summary copied at the Imagination -> Main boundary.
    # Without this, Main only sees the endpoint and loses whether refinement
    # moved in the intended base-frame direction.
    edit_summary: EditSummary | None = None


@dataclass(frozen=True)
class LastPhysicalArtifacts:
    focus_pose: Pose | None = None


@dataclass(frozen=True)
class EditSummary:
    """Minimal command memory for one revision-local imagination session."""

    initial_target: ActionTarget
    current_target: ActionTarget
    total_translation_base_m: tuple[float, float, float] | None
    total_rotation_axis_base: tuple[float, float, float] | None
    total_rotation_deg: float | None
    previous_edit: VisualEdit | None
    last_edit: VisualEdit | None

    def summary(self) -> dict[str, Any]:
        # A quaternion is an exact transport representation, but its four
        # components are not per-axis angles.  Exposing both endpoint
        # quaternions encouraged vision-language models to invent an Euler
        # interpretation and rotate otherwise useful seeds.  The control
        # summary therefore carries position/gripper endpoints plus the exact
        # relative axis-angle that was actually edited.  Full poses remain in
        # private state and trace diagnostics.
        result: dict[str, Any] = {
            "initial_target": _control_target_summary(self.initial_target),
            "current_target": _control_target_summary(self.current_target),
        }
        if self.total_translation_base_m is not None:
            result["total_translation_base_m"] = [
                round(float(value), 6) for value in self.total_translation_base_m
            ]
        if self.total_rotation_deg is not None:
            result["total_rotation_deg"] = round(float(self.total_rotation_deg), 3)
            result["total_rotation_axis_base"] = [
                round(float(value), 6)
                for value in self.total_rotation_axis_base or (0.0, 0.0, 0.0)
            ]
        if self.previous_edit is not None:
            result["previous_edit"] = self.previous_edit.command_summary()
        if self.last_edit is not None:
            result["last_edit"] = self.last_edit.command_summary()
        return result


def _control_target_summary(target: ActionTarget) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if target.pose is not None:
        result["position_xyz"] = [
            round(float(value), 6) for value in target.pose.position_xyz
        ]
    if target.gripper is not None:
        result["gripper_target"] = target.gripper
    return result


def build_edit_summary(
    current_target: ActionTarget,
    artifacts: ImaginationArtifacts | None,
) -> EditSummary:
    """Compile exact cumulative pose change without replaying tool history."""

    initial = (
        artifacts.initial_target
        if artifacts is not None and artifacts.initial_target is not None
        else current_target
    )
    translation: tuple[float, float, float] | None = None
    rotation_axis: tuple[float, float, float] | None = None
    rotation_deg: float | None = None
    if initial.pose is not None and current_target.pose is not None:
        initial_position = np.asarray(initial.pose.position_xyz, dtype=np.float64)
        current_position = np.asarray(current_target.pose.position_xyz, dtype=np.float64)
        translation = tuple(float(value) for value in current_position - initial_position)
        initial_rotation = Rotation.from_quat(initial.pose.quaternion_xyzw)
        current_rotation = Rotation.from_quat(current_target.pose.quaternion_xyzw)
        rotvec = (current_rotation * initial_rotation.inv()).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        rotation_deg = float(np.rad2deg(angle))
        rotation_axis = (
            tuple(float(value) for value in rotvec / angle)
            if angle > 1e-9
            else (0.0, 0.0, 0.0)
        )
    return EditSummary(
        initial_target=initial,
        current_target=current_target,
        total_translation_base_m=translation,
        total_rotation_axis_base=rotation_axis,
        total_rotation_deg=rotation_deg,
        previous_edit=(artifacts.previous_visual_edit if artifacts is not None else None),
        last_edit=(artifacts.latest_visual_edit if artifacts is not None else None),
    )


@dataclass(frozen=True)
class PresentationEvent:
    function_name: str
    result: dict[str, Any]
    error: str | None = None


@dataclass
class PrivateEnvContext:
    observation: dict[str, Any] | None = None
    previous_observation: dict[str, Any] | None = None
    semantic_rgb: np.ndarray | None = None
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    region_geometry: dict[str, RegionGeometryArtifact] = field(default_factory=dict)
    seed_artifacts: dict[str, SeedArtifacts] = field(default_factory=dict)
    imagination_artifacts: ImaginationArtifacts | None = None
    review_artifacts: dict[str, ActionReviewArtifacts] = field(default_factory=dict)
    last_physical_artifacts: LastPhysicalArtifacts | None = None
    presentation_event: PresentationEvent | None = None
    trace_diagnostics: dict[str, Any] = field(default_factory=dict)

    def begin_revision(self, observation: dict[str, Any]) -> None:
        self.previous_observation = self.observation
        self.observation = observation
        self.semantic_rgb = None
        self.region_masks.clear()
        self.region_geometry.clear()
        self.seed_artifacts.clear()
        self.imagination_artifacts = None
        self.review_artifacts.clear()
        self.last_physical_artifacts = None
        self.presentation_event = None
        self.trace_diagnostics.clear()

    def begin_function_call(self) -> None:
        self.trace_diagnostics.clear()

    def camera(self, name: str, *, previous: bool = False) -> dict[str, Any]:
        observation = self.previous_observation if previous else self.observation
        if observation is None:
            raise RuntimeError("observation is unavailable")
        camera = observation.get(name)
        if not isinstance(camera, dict):
            raise RuntimeError(f"observation has no camera '{name}'")
        return camera


__all__ = [
    "EditSummary",
    "ImaginationArtifacts",
    "PlanningContext",
    "PresentationEvent",
    "PrivateEnvContext",
    "ActionReviewArtifacts",
    "LastPhysicalArtifacts",
    "RegionGeometryArtifact",
    "SeedArtifacts",
    "VisualEdit",
    "build_edit_summary",
]
