"""Small semantic state for the Main-ReAct Visual Action Workspace.

The records in this module describe evidence and commands, not execution
history.  Sensor arrays, planner results and presentation provenance stay in
``private.py`` and are never serialized into a policy message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vaw.context_runtime.memory import TaskMemory


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
    """One spatial pose under visual review."""

    pose: Pose

    def summary(self) -> dict[str, Any]:
        return {"pose": self.pose.summary()}


@dataclass(frozen=True)
class ActionSeed:
    """One revision-local starting point for an imagination session."""

    seed_id: str
    target: ActionTarget
    source_revision: int


@dataclass(frozen=True)
class PendingAction:
    """One spatial action awaiting optional refinement or physical commit."""

    action_id: str
    target: ActionTarget
    intent: str
    ready_for_commit: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target": self.target.summary(),
            "intent": self.intent,
            "ready_for_commit": self.ready_for_commit,
        }


@dataclass(frozen=True)
class RefinementSession:
    """A synchronous, episode-private task delegated by Main to Imagination."""

    action_id: str
    instruction: str
    initial_target: ActionTarget
    created_action: bool = False


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
    task_memory: TaskMemory = field(default_factory=TaskMemory)
    observation_revision: int = 0
    regions: dict[str, RegionEvidence] = field(default_factory=dict)
    points: dict[str, PointEvidence] = field(default_factory=dict)
    seeds: dict[str, ActionSeed] = field(default_factory=dict)
    robot: RobotState | None = None
    pending_action: PendingAction | None = None
    refinement: RefinementSession | None = None
    last_physical_action: LastPhysicalAction | None = None
    _counters: dict[str, int] = field(default_factory=dict, repr=False)

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    def begin_revision(self) -> int:
        self.observation_revision += 1
        self.regions.clear()
        self.points.clear()
        self.seeds.clear()
        self.pending_action = None
        self.refinement = None
        self.last_physical_action = None
        return self.observation_revision

    def manifest(self) -> dict[str, Any]:
        return {
            "pending_action": (
                {
                    "action_id": self.pending_action.action_id,
                    "intent": self.pending_action.intent,
                    "state": (
                        "ready" if self.pending_action.ready_for_commit else "coarse"
                    ),
                }
                if self.pending_action is not None
                else None
            ),
            "regions": [
                {"id": region.region_id, "query": region.query}
                for region in self.regions.values()
            ],
            "points": [
                {
                    "id": point.point_id,
                    "query": point.query,
                    **(
                        {"within_region_id": point.within_region_id}
                        if point.within_region_id is not None
                        else {}
                    ),
                }
                for point in self.points.values()
            ],
            "seed_ids": list(self.seeds),
        }

    def trace_summary(self) -> dict[str, Any]:
        return {
            "task_prompt": self.task_prompt,
            "task_memory": self.task_memory.summary(),
            "observation_revision": self.observation_revision,
            "regions": [item.summary() for item in self.regions.values()],
            "points": [item.summary() for item in self.points.values()],
            "seed_ids": list(self.seeds),
            "robot": self.robot.summary() if self.robot is not None else None,
            "refinement": (
                {
                    "action_id": self.refinement.action_id,
                    "instruction": self.refinement.instruction,
                }
                if self.refinement is not None
                else None
            ),
            "pending_action": (
                self.pending_action.summary() if self.pending_action is not None else None
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
    "LastPhysicalAction",
    "PointEvidence",
    "Pose",
    "PendingAction",
    "RefinementSession",
    "RegionEvidence",
    "RobotState",
]
