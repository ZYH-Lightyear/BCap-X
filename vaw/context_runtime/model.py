"""Public, array-free state for the VAW visual Context Runtime.

Raw RGB-D, masks, camera matrices and point clouds do not belong here.  They
stay in :class:`vaw.context_runtime.workspace.ContextWorkspace` and are keyed
by the short handles defined below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


def _float_list(values: tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in values]


@dataclass(frozen=True)
class Pose:
    """Robot-base pose using the public ``xyzw`` quaternion convention."""

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def summary(self) -> dict[str, list[float]]:
        return {
            "position_xyz": _float_list(self.position_xyz),
            "quaternion_xyzw": _float_list(self.quaternion_xyzw),
        }


@dataclass(frozen=True)
class RegionEvidence:
    region_id: str
    query: str
    bbox_xyxy_px: tuple[float, float, float, float]
    source_revision: int
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "region_id": self.region_id,
            "query": self.query,
            "bbox_xyxy_px": _float_list(self.bbox_xyxy_px),
            "source_revision": self.source_revision,
        }
        if self.within_region_id is not None:
            out["within_region_id"] = self.within_region_id
        return out


@dataclass(frozen=True)
class PointEvidence:
    point_id: str
    query: str
    pixel_xy: tuple[float, float]
    position_xyz: tuple[float, float, float]
    source_revision: int
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "point_id": self.point_id,
            "query": self.query,
            "pixel_xy": _float_list(self.pixel_xy),
            "position_xyz": _float_list(self.position_xyz),
            "source_revision": self.source_revision,
        }
        if self.within_region_id is not None:
            out["within_region_id"] = self.within_region_id
        return out


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    kind: str
    source_ref: str | None
    target_pose: Pose
    source_revision: int
    prediction: ActionPrediction

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "target_pose": self.target_pose.summary(),
            "source_revision": self.source_revision,
            "prediction": self.prediction.summary(),
        }
        if self.source_ref is not None:
            out["source_ref"] = self.source_ref
        return out


@dataclass(frozen=True)
class RobotState:
    # ``ee_pose`` is the backend-observed panda_hand link retained for
    # execution discrepancy checks.  ``tcp_pose`` is the policy-facing
    # fingertip/contact frame shared by ActionProposal targets.
    ee_pose: Pose | None
    tcp_pose: Pose | None
    joint_positions_rad: tuple[float, ...] | None
    gripper_opening: float | None
    source_revision: int

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source_revision": self.source_revision}
        if self.ee_pose is not None:
            out["ee_pose"] = self.ee_pose.summary()
        if self.tcp_pose is not None:
            out["tcp_pose"] = self.tcp_pose.summary()
        if self.joint_positions_rad is not None:
            out["joint_positions_rad"] = _float_list(self.joint_positions_rad)
        if self.gripper_opening is not None:
            out["gripper_opening"] = round(float(self.gripper_opening), 6)
        return out


SolveIKStatus = Literal["returned", "error", "unavailable"]


@dataclass(frozen=True)
class ActionPrediction:
    solve_ik: SolveIKStatus
    joint_positions_rad: tuple[float, ...] | None = None
    trajectory_checked: bool = False
    collision_checked: bool = False
    detail: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "solve_ik": self.solve_ik,
            "trajectory_checked": self.trajectory_checked,
            "collision_checked": self.collision_checked,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


AdjustmentKind = Literal["delta_move", "rotate"]
AdjustmentFrame = Literal["base", "tool"]


@dataclass(frozen=True)
class ActionAdjustment:
    """Latest revision-local edit that produced an Action Proposal.

    This is deliberately a single edit rather than an unbounded ancestry.  It
    gives the trusted presenter enough geometry to explain the current virtual
    waypoint while K=3 history and the trace retain the Function transaction.
    """

    kind: AdjustmentKind
    frame: AdjustmentFrame
    reference_pose: Pose
    parent_action_id: str | None = None
    delta_xyz_m: tuple[float, float, float] | None = None
    axis: Literal["x", "y", "z"] | None = None
    angle_deg: float | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "frame": self.frame,
            "reference_pose": self.reference_pose.summary(),
        }
        if self.parent_action_id is not None:
            out["parent_action_id"] = self.parent_action_id
        if self.delta_xyz_m is not None:
            out["delta_xyz_m"] = _float_list(self.delta_xyz_m)
        if self.axis is not None:
            out["axis"] = self.axis
        if self.angle_deg is not None:
            out["angle_deg"] = round(float(self.angle_deg), 6)
        return out


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    kind: str
    source_ref: str | None
    source_revision: int
    target_pose: Pose
    prediction: ActionPrediction
    adjustment: ActionAdjustment | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "action_id": self.action_id,
            "kind": self.kind,
            "source_revision": self.source_revision,
            "target_pose": self.target_pose.summary(),
            "prediction": self.prediction.summary(),
        }
        if self.source_ref is not None:
            out["source_ref"] = self.source_ref
        if self.adjustment is not None:
            out["adjustment"] = self.adjustment.summary()
        return out


@dataclass(frozen=True)
class SpatialTargetSummary:
    """Most recently executed spatial target, retained only for visual review."""

    action_id: str
    target_pose: Pose
    revision_after: int


@dataclass(frozen=True)
class ExecutionReceipt:
    receipt_id: str
    function_name: str
    revision_before: int
    revision_after: int
    action_id: str | None = None
    position_error_m: float | None = None
    gripper_opening: float | None = None
    discrepancy: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "receipt_id": self.receipt_id,
            "function_name": self.function_name,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
        }
        if self.action_id is not None:
            out["action_id"] = self.action_id
        if self.position_error_m is not None:
            out["position_error_m"] = round(float(self.position_error_m), 6)
        if self.gripper_opening is not None:
            out["gripper_opening"] = round(float(self.gripper_opening), 6)
        if self.discrepancy:
            out["discrepancy"] = dict(self.discrepancy)
        return out


@dataclass(frozen=True)
class FunctionRecord:
    function_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    revision_before: int
    revision_after: int
    action_id: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "function_name": self.function_name,
            "arguments": self.arguments,
            "result": self.result,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "ok": "error" not in self.result,
        }
        if self.action_id is not None:
            out["action_id"] = self.action_id
        return out


@dataclass
class ContextState:
    task_prompt: str
    observation_revision: int = 0
    regions: dict[str, RegionEvidence] = field(default_factory=dict)
    points: dict[str, PointEvidence] = field(default_factory=dict)
    candidates: dict[str, ActionCandidate] = field(default_factory=dict)
    robot: RobotState | None = None
    active_action: ActionProposal | None = None
    last_spatial_target: SpatialTargetSummary | None = None
    last_receipt: ExecutionReceipt | None = None
    recent_calls: list[FunctionRecord] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict, repr=False)

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    def begin_revision(self) -> int:
        self.observation_revision += 1
        self.regions.clear()
        self.points.clear()
        self.candidates.clear()
        self.active_action = None
        return self.observation_revision

    def add_record(self, record: FunctionRecord) -> None:
        self.recent_calls.append(record)
        del self.recent_calls[:-3]

    def manifest(self) -> dict[str, Any]:
        return {
            "revision": self.observation_revision,
            "active_action_id": (
                self.active_action.action_id if self.active_action is not None else None
            ),
            "valid_region_ids": list(self.regions),
            "valid_point_ids": list(self.points),
            "valid_candidate_ids": list(self.candidates),
        }

    def trace_summary(self) -> dict[str, Any]:
        """Array-free semantic state for debugging and offline traces.

        This is deliberately *not* the default policy prompt.  The policy gets
        :meth:`manifest`, the current Context image and recent tool results.
        """

        return {
            "task_prompt": self.task_prompt,
            "manifest": self.manifest(),
            "regions": [item.summary() for item in self.regions.values()],
            "points": [item.summary() for item in self.points.values()],
            "candidates": [item.summary() for item in self.candidates.values()],
            "robot": self.robot.summary() if self.robot is not None else None,
            "active_action": (
                self.active_action.summary() if self.active_action is not None else None
            ),
            "last_receipt": (
                self.last_receipt.summary() if self.last_receipt is not None else None
            ),
            "recent_calls": [item.summary() for item in self.recent_calls],
        }
