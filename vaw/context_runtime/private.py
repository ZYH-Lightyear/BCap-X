"""Episode-private sensor, planning and presentation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

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
        return result


@dataclass(frozen=True)
class SeedArtifacts:
    planning_context: PlanningContext | None = None
    preview_plan: MotionPlan | None = None


@dataclass(frozen=True)
class ImaginationArtifacts:
    planning_context: PlanningContext | None = None
    preview_plan: MotionPlan | None = None
    latest_visual_edit: VisualEdit | None = None
    turn_count: int = 0


@dataclass(frozen=True)
class ActionReviewArtifacts:
    motion_plan: MotionPlan | None = None
    planning_context: PlanningContext | None = None
    handoff_reason: str = "completed"


@dataclass(frozen=True)
class PresentationEvent:
    function_name: str
    result: dict[str, Any]
    error: str | None = None


@dataclass
class PrivateEnvContext:
    observation: dict[str, Any] | None = None
    previous_observation: dict[str, Any] | None = None
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    region_geometry: dict[str, RegionGeometryArtifact] = field(default_factory=dict)
    seed_artifacts: dict[str, SeedArtifacts] = field(default_factory=dict)
    imagination_artifacts: ImaginationArtifacts | None = None
    review_artifacts: dict[str, ActionReviewArtifacts] = field(default_factory=dict)
    presentation_event: PresentationEvent | None = None
    trace_diagnostics: dict[str, Any] = field(default_factory=dict)

    def begin_revision(self, observation: dict[str, Any]) -> None:
        self.previous_observation = self.observation
        self.observation = observation
        self.region_masks.clear()
        self.region_geometry.clear()
        self.seed_artifacts.clear()
        self.imagination_artifacts = None
        self.review_artifacts.clear()
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
    "ImaginationArtifacts",
    "PlanningContext",
    "PresentationEvent",
    "PrivateEnvContext",
    "ActionReviewArtifacts",
    "RegionGeometryArtifact",
    "SeedArtifacts",
    "VisualEdit",
]
