"""Small semantic state for the dual-agent Visual Action Workspace.

The records in this module describe evidence and commands, not execution
history.  Sensor arrays, planner results and presentation provenance stay in
``private.py`` and are never serialized into a policy message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GripperTarget = Literal["open", "closed"]
ImaginationOutcome = Literal["review_required", "failed"]


def _floats(values: tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in values]


@dataclass(frozen=True)
class Pose:
    """Robot-base pose using the public ``xyzw`` convention."""

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def summary(self) -> dict[str, list[float]]:
        return {
            "position_xyz": _floats(self.position_xyz),
            "quaternion_xyzw": _floats(self.quaternion_xyzw),
        }


@dataclass(frozen=True)
class ActionTarget:
    """One atomic arm or gripper target under visual review."""

    pose: Pose | None = None
    gripper: GripperTarget | None = None

    def __post_init__(self) -> None:
        if (self.pose is None) == (self.gripper is None):
            raise ValueError("ActionTarget must contain exactly one of pose or gripper")

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.pose is not None:
            result["pose"] = self.pose.summary()
        if self.gripper is not None:
            result["gripper"] = self.gripper
        return result


@dataclass(frozen=True)
class ActionSeed:
    """One revision-local starting point for an imagination session."""

    seed_id: str
    target: ActionTarget
    source_revision: int


@dataclass
class ImaginationState:
    """The one mutable target owned by the Imagination Agent."""

    target: ActionTarget
    refinement_goal: str


@dataclass(frozen=True)
class ActionReview:
    """A final imagination target awaiting an explicit Main-Agent decision."""

    action_id: str
    target: ActionTarget
    intent: str

    def summary(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target": self.target.summary(),
            "intent": self.intent,
        }


@dataclass(frozen=True)
class ImaginationHandoff:
    status: ImaginationOutcome
    action_id: str | None = None
    source_ref: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status}
        if self.action_id is not None:
            result["action_id"] = self.action_id
        if self.source_ref is not None:
            result["source_ref"] = self.source_ref
        return result


@dataclass(frozen=True)
class LastPhysicalAction:
    """Revision-local causal continuity, never a task-effect claim.

    The record is overwrite-only and disappears at the next physical revision.
    It is not a transcript and carries no assertion about grasp/place success.
    """

    intent: str
    executed_stages: Literal["arm", "gripper"]
    outcome: Literal["completed", "arm_failed", "gripper_failed"]
    target_gripper: Literal["open", "closed"] | None = None
    requested_arm_delta_base_m: tuple[float, float, float] | None = None
    failed_action_id: str | None = None
    error_detail: str | None = None
    source_query: str | None = None
    evidence_invalidated: bool = False
    recovery_hint: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "intent": self.intent,
            "executed_stages": self.executed_stages,
            "outcome": self.outcome,
        }
        if self.target_gripper is not None:
            result["target_gripper"] = self.target_gripper
        if self.requested_arm_delta_base_m is not None:
            result["requested_arm_delta_base_m"] = _floats(
                self.requested_arm_delta_base_m
            )
        if self.failed_action_id is not None:
            result["failed_action_id"] = self.failed_action_id
        if self.error_detail:
            result["error_detail"] = self.error_detail
        if self.source_query:
            result["source_query"] = self.source_query
        if self.evidence_invalidated:
            result["evidence_invalidated"] = True
        if self.recovery_hint:
            result["recovery_hint"] = self.recovery_hint
        return result


@dataclass(frozen=True)
class RegionEvidence:
    region_id: str
    query: str
    bbox_xyxy_px: tuple[float, float, float, float]
    source_revision: int
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "region_id": self.region_id,
            "query": self.query,
            "bbox_xyxy_px": _floats(self.bbox_xyxy_px),
        }
        if self.within_region_id is not None:
            result["within_region_id"] = self.within_region_id
        return result


@dataclass(frozen=True)
class PointEvidence:
    point_id: str
    query: str
    pixel_xy: tuple[float, float]
    position_xyz: tuple[float, float, float]
    source_revision: int
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "point_id": self.point_id,
            "query": self.query,
            "pixel_xy": _floats(self.pixel_xy),
            "position_xyz": _floats(self.position_xyz),
        }
        if self.within_region_id is not None:
            result["within_region_id"] = self.within_region_id
        return result


@dataclass(frozen=True)
class RobotState:
    ee_pose: Pose | None
    tcp_pose: Pose | None
    joint_positions_rad: tuple[float, ...] | None
    gripper_opening: float | None
    source_revision: int

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.ee_pose is not None:
            result["ee_pose"] = self.ee_pose.summary()
        if self.tcp_pose is not None:
            result["tcp_pose"] = self.tcp_pose.summary()
        if self.joint_positions_rad is not None:
            result["joint_positions_rad"] = _floats(self.joint_positions_rad)
        if self.gripper_opening is not None:
            result["gripper_opening"] = round(float(self.gripper_opening), 6)
        return result


SolveIKStatus = Literal["returned", "error", "unavailable"]


@dataclass(frozen=True)
class ActionPrediction:
    solve_ik: SolveIKStatus
    joint_positions_rad: tuple[float, ...] | None = None
    trajectory_checked: bool = False
    collision_checked: bool = False
    detail: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "solve_ik": self.solve_ik,
            "trajectory_checked": self.trajectory_checked,
            "collision_checked": self.collision_checked,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass
class ContextState:
    task_prompt: str
    observation_revision: int = 0
    regions: dict[str, RegionEvidence] = field(default_factory=dict)
    points: dict[str, PointEvidence] = field(default_factory=dict)
    seeds: dict[str, ActionSeed] = field(default_factory=dict)
    robot: RobotState | None = None
    imagination: ImaginationState | None = None
    action_review: ActionReview | None = None
    last_handoff: ImaginationHandoff | None = None
    last_physical_action: LastPhysicalAction | None = None
    _counters: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def owner(self) -> Literal["main", "imagination"]:
        return "imagination" if self.imagination is not None else "main"

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    def begin_revision(self) -> int:
        self.observation_revision += 1
        self.regions.clear()
        self.points.clear()
        self.seeds.clear()
        self.imagination = None
        self.action_review = None
        self.last_handoff = None
        self.last_physical_action = None
        return self.observation_revision

    def manifest(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "review_action_id": (
                self.action_review.action_id if self.action_review is not None else None
            ),
            "valid_region_ids": list(self.regions),
            "valid_point_ids": list(self.points),
            "valid_seed_ids": list(self.seeds),
        }

    def trace_summary(self) -> dict[str, Any]:
        return {
            "task_prompt": self.task_prompt,
            "observation_revision": self.observation_revision,
            "owner": self.owner,
            "regions": [item.summary() for item in self.regions.values()],
            "points": [item.summary() for item in self.points.values()],
            "seed_ids": list(self.seeds),
            "robot": self.robot.summary() if self.robot is not None else None,
            "imagination": (
                {
                    "target": self.imagination.target.summary(),
                    "refinement_goal": self.imagination.refinement_goal,
                }
                if self.imagination is not None
                else None
            ),
            "action_review": (
                self.action_review.summary() if self.action_review is not None else None
            ),
            "last_handoff": (
                self.last_handoff.summary() if self.last_handoff is not None else None
            ),
            "last_physical_action": (
                self.last_physical_action.summary()
                if self.last_physical_action is not None
                else None
            ),
        }


__all__ = [
    "ActionPrediction",
    "ActionSeed",
    "ActionTarget",
    "ContextState",
    "GripperTarget",
    "ImaginationHandoff",
    "ImaginationState",
    "LastPhysicalAction",
    "PointEvidence",
    "Pose",
    "ActionReview",
    "RegionEvidence",
    "RobotState",
]
