"""Small semantic state for the Main-ReAct Visual Action Workspace.

The records in this module describe evidence and commands, not execution
history.  Sensor arrays, planner results and presentation provenance stay in
``private.py`` and are never serialized into a policy message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
class ActionProposal:
    """One Main-owned spatial action that has not been executed.

    ``refined`` records provenance only: whether an Imagination session
    delivered the current target.  It is not a commit qualification; commit
    eligibility is decided by the executable cached plan alone.
    """

    action_id: str
    target: ActionTarget
    intent: str
    refined: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target": self.target.summary(),
            "intent": self.intent,
            "refined": self.refined,
        }


@dataclass(frozen=True)
class ImaginationSession:
    """A synchronous, episode-private task delegated by Main to Imagination."""

    action_id: str
    instruction: str
    initial_target: ActionTarget


@dataclass
class ImaginationAttempts:
    """Revision-local Imagination session ledger, never a transcript."""

    total: int = 0
    failed: int = 0
    last_reason: str | None = None
    failed_source_refs: list[str] = field(default_factory=list)

    def record(
        self,
        *,
        failed: bool,
        reason: str | None = None,
        source_ref: str | None = None,
    ) -> None:
        self.total += 1
        if not failed:
            return
        self.failed += 1
        if reason:
            self.last_reason = reason
        if source_ref and source_ref not in self.failed_source_refs:
            self.failed_source_refs.append(source_ref)

    def summary(self) -> dict[str, Any] | None:
        if self.total == 0:
            return None
        result: dict[str, Any] = {"total": self.total, "failed": self.failed}
        if self.last_reason:
            result["last_reason"] = self.last_reason
        if self.failed_source_refs:
            result["failed_source_refs"] = list(self.failed_source_refs)
        return result

@dataclass(frozen=True)
class LastPhysicalAction:
    """Revision-local causal continuity, never a task-effect claim.

    The record is overwrite-only and disappears at the next physical revision.
    It is not a transcript and carries no assertion about grasp/place success.
    """

    intent: str
    executed_stages: Literal["arm", "gripper"]
    outcome: Literal[
        "completed", "arm_failed", "arm_unsettled", "gripper_failed"
    ]
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

# Grounded evidence is retained across physical actions and revalidated
# against each new observation.  ``verified`` means the archived appearance
# still matches the current image; ``occluded`` means the robot body (or a
# closer surface) currently blocks the check, so the entry is kept but its
# geometry could not be re-confirmed.  Evidence with positive proof of change
# is deleted rather than flagged.
EvidenceStatus = Literal["verified", "occluded"]

@dataclass(frozen=True)
class RegionEvidence:
    region_id: str
    query: str
    bbox_xyxy_px: tuple[float, float, float, float]
    source_revision: int
    within_region_id: str | None = None
    status: EvidenceStatus = "verified"

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "region_id": self.region_id,
            "query": self.query,
            "bbox_xyxy_px": _floats(self.bbox_xyxy_px),
        }
        if self.within_region_id is not None:
            result["within_region_id"] = self.within_region_id
        if self.status != "verified":
            result["status"] = self.status
        return result


@dataclass(frozen=True)
class PointEvidence:
    point_id: str
    query: str
    pixel_xy: tuple[float, float]
    position_xyz: tuple[float, float, float]
    source_revision: int
    within_region_id: str | None = None
    status: EvidenceStatus = "verified"

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "point_id": self.point_id,
            "query": self.query,
            "pixel_xy": _floats(self.pixel_xy),
            "position_xyz": _floats(self.position_xyz),
        }
        if self.within_region_id is not None:
            result["within_region_id"] = self.within_region_id
        if self.status != "verified":
            result["status"] = self.status
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
    action_proposal: ActionProposal | None = None
    imagination: ImaginationSession | None = None
    last_physical_action: LastPhysicalAction | None = None
    imagination_attempts: ImaginationAttempts = field(default_factory=ImaginationAttempts)
    # Episode-level behaviour ledger: repeated grasp closures surface as an
    # advisory in the Function Event, never as a gate.
    gripper_close_count: int = 0
    _counters: dict[str, int] = field(default_factory=dict, repr=False)

    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    def begin_revision(self) -> int:
        # Grounded regions/points survive the revision boundary; the workspace
        # revalidates them against the fresh observation immediately after and
        # deletes only entries with positive evidence of change.  Seeds and the
        # proposal stay revision-local: their cached plans start from a robot
        # state that no longer exists.
        self.observation_revision += 1
        self.seeds.clear()
        self.action_proposal = None
        self.imagination = None
        self.last_physical_action = None
        self.imagination_attempts = ImaginationAttempts()
        return self.observation_revision

    def manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action_proposal": (
                {
                    "action_id": self.action_proposal.action_id,
                    "intent": self.action_proposal.intent,
                    "state": (
                        "refined" if self.action_proposal.refined else "planned"
                    ),
                }
                if self.action_proposal is not None
                else None
            ),
            "regions": [
                {
                    "id": region.region_id,
                    "query": region.query,
                    **(
                        {"status": region.status}
                        if region.status != "verified"
                        else {}
                    ),
                }
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
                    **(
                        {"status": point.status}
                        if point.status != "verified"
                        else {}
                    ),
                }
                for point in self.points.values()
            ],
            "seed_ids": list(self.seeds),
        }
        attempts = self.imagination_attempts.summary()
        if attempts is not None:
            result["imagination_attempts"] = attempts
        return result

    def trace_summary(self) -> dict[str, Any]:
        return {
            "task_prompt": self.task_prompt,
            "observation_revision": self.observation_revision,
            "regions": [item.summary() for item in self.regions.values()],
            "points": [item.summary() for item in self.points.values()],
            "seed_ids": list(self.seeds),
            "robot": self.robot.summary() if self.robot is not None else None,
            "imagination": (
                {
                    "action_id": self.imagination.action_id,
                    "instruction": self.imagination.instruction,
                }
                if self.imagination is not None
                else None
            ),
            "action_proposal": (
                self.action_proposal.summary() if self.action_proposal is not None else None
            ),
            "last_physical_action": (
                self.last_physical_action.summary()
                if self.last_physical_action is not None
                else None
            ),
        }


__all__ = [
    "ActionPrediction",
    "ActionProposal",
    "ActionSeed",
    "ActionTarget",
    "ContextState",
    "ImaginationAttempts",
    "ImaginationSession",
    "LastPhysicalAction",
    "PointEvidence",
    "Pose",
    "RegionEvidence",
    "RobotState",
]
