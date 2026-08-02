"""Endpoint preview before a motion planner is connected.

This stage answers exactly one question: can IK produce the requested terminal
pose without silently substituting another orientation?  It stores that exact
joint solution so the canvas can render the terminal gripper through URDF FK.

It deliberately does *not* invent a trajectory. A Cartesian line from the
reported TCP to the target is not the path a joint-controlled Franka follows.
Trajectory and swept-volume evidence will only return when a motion planner
such as CuRobo supplies the joint sequence that execution will actually
consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaw.state import ActionState
from vaw.types import Candidate, Pose, PreviewResult


def run_preview(
    api: Any,
    state: ActionState,
    cand: Candidate,
) -> PreviewResult:
    """Check terminal IK and prepare its exact-FK gripper overlay."""
    target = cand.pose

    # 1) IK feasibility. solve_ik silently falls back to canned orientations;
    #    return_info exposes that, and a fallback counts as "not the pose the
    #    agent asked for" rather than a pass.
    ik_ok, orientation_used = False, "failed"
    ik_joints = None
    try:
        solved, info = api.solve_ik(
            target.position, target.quat_wxyz, return_info=True
        )
        solved = np.asarray(solved, dtype=np.float64).reshape(-1)
        if solved.size < 7 or not np.isfinite(solved[:7]).all():
            raise ValueError("IK returned no finite seven-joint solution")
        ik_joints = solved[:7].copy()
        orientation_used = str(info.get("orientation_used", "requested"))
        ik_ok = orientation_used == "requested"
    except Exception as exc:
        orientation_used = f"failed ({type(exc).__name__})"

    notes = (
        "endpoint IK solved; trajectory not planned"
        if ik_ok
        else f"endpoint IK failed: {orientation_used}"
    )

    preview = PreviewResult(
        candidate_id=cand.candidate_id,
        ik_ok=ik_ok,
        orientation_used=orientation_used,
        predicted_ee=Pose(target.position.copy(), target.quat_wxyz.copy()),
        joint_positions_rad=ik_joints,
        notes=notes,
    )
    state.add_preview(preview)
    return preview
