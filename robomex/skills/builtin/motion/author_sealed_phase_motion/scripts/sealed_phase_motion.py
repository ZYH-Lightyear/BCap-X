"""Audited constructor for one immutable RoboMEx v2 arm phase."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    ExecutionPolicy,
    JointPath,
    MotionPlan,
)

_PLAN_KINDS = (
    "transport_to_safe_hover",
    "bounded_correction",
    "descend_to_release",
    "safe_retreat",
)


def build_sealed_phase_motion(
    snapshot: dict[str, object],
    *,
    plan_kind: str,
    joint_waypoints: Sequence[Sequence[float]],
    tcp_frame_id: str = "panda_hand",
    planner_backend: str = "curobo",
    robot_model_digest: str | None = None,
    max_start_deviation_rad: float = 0.02,
    subsample: int = 1,
    timeout_s: float = 30.0,
) -> dict[str, object]:
    """Validate and seal exactly one pre-solved joint-path phase."""

    admitted = AdmissionSnapshot.model_validate(snapshot)
    if plan_kind not in _PLAN_KINDS:
        raise ValueError(f"unsupported sealed phase plan_kind: {plan_kind!r}")
    positions = tuple(tuple(float(value) for value in waypoint) for waypoint in joint_waypoints)
    if not positions:
        raise ValueError("joint_waypoints must include the exact admitted start waypoint")
    if positions[0] != admitted.joint_positions_rad:
        raise ValueError("first joint waypoint must exactly equal the admission snapshot")
    motion = JointPath(
        joint_names=admitted.joint_names,
        positions_rad=positions,
        execution_policy=ExecutionPolicy(subsample=subsample, timeout_s=timeout_s),
    )
    semantic = json.dumps(
        {
            "plan_kind": plan_kind,
            "snapshot": admitted.model_dump(mode="json"),
            "positions": positions,
            "tcp_frame_id": tcp_frame_id,
            "planner_backend": planner_backend,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    plan = MotionPlan(
        plan_id=f"{plan_kind}-" + hashlib.sha256(semantic).hexdigest()[:20],
        plan_kind=plan_kind,
        tcp_frame_id=tcp_frame_id,
        planner_backend=planner_backend,
        robot_model_digest=robot_model_digest or admitted.config_digest,
        expected_snapshot=admitted,
        max_start_deviation_rad=max_start_deviation_rad,
        motion=motion,
        possibly_affected_revisions=("robot.arm", "scene", "attachment"),
    )
    return plan.model_dump(mode="json")


__all__ = ["build_sealed_phase_motion"]
