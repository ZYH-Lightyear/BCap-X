"""Geometric rollout: the workspace's explicit, physics-grounded preview.

M1 version: IK feasibility (with orientation-fallback detection) + straight
line end-effector path checked for clearance against the scene point cloud
(target object's own points excluded near the goal, since an approach must
get close to what it manipulates).

M2: replace the straight-line path with the cuRobo trajectory already
implemented in ``FrankaLiberoApiReduced.plan_grasp_trajectory`` and render
its swept volume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaw.geometry import (
    CONTACT_TO_IK_TARGET_M,
    depth_to_world_points,
    interpolate_path,
    min_clearance,
    shift_along_approach,
)
from vaw.state import ActionState
from vaw.types import Candidate, Pose, PreviewResult

# Path points closer than this to the scene cloud count as collision ...
CLEARANCE_THRESHOLD_M = 0.01
# ... except within this radius of the goal, where contact is intended.
GOAL_EXCLUSION_RADIUS_M = 0.06


def run_preview(
    api: Any,
    state: ActionState,
    cand: Candidate,
    obs: dict[str, Any],
    *,
    camera_name: str = "agentview",
) -> PreviewResult:
    """Predict the outcome of committing ``cand``. Never moves the robot."""
    target = cand.pose

    # 1) IK feasibility. solve_ik silently falls back to canned orientations;
    #    return_info exposes that, and a fallback counts as "not the pose the
    #    agent asked for" rather than a pass.
    ik_ok, orientation_used = False, "failed"
    try:
        solve_arm_ik(api, target)
        ik_ok, orientation_used = True, "requested"
    except Exception as exc:
        orientation_used = f"failed ({type(exc).__name__})"

    # 2) Straight-line path from current EE to the target position.
    ee = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64)[:3]
    path = interpolate_path(ee, target.position, num=24)

    # 3) Clearance of the path against the scene cloud, excluding the region
    #    around the goal (intended contact) and the target object's points.
    cam = obs[camera_name]
    scene = depth_to_world_points(
        cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"], stride=4
    )
    if cand.object_id and cand.object_id in state.objects:
        obj_pts = state.objects[cand.object_id].points_world
        if obj_pts is not None and len(obj_pts) and len(scene):
            # Cheap exclusion: drop scene points near the object's centroid.
            centroid = np.median(obj_pts, axis=0)
            keep = np.linalg.norm(scene - centroid, axis=1) > GOAL_EXCLUSION_RADIUS_M
            scene = scene[keep]
    goal_dist = np.linalg.norm(path - target.position[None, :], axis=1)
    check_path = path[goal_dist > GOAL_EXCLUSION_RADIUS_M]
    clearance = min_clearance(check_path, scene)
    collision = clearance < CLEARANCE_THRESHOLD_M

    notes = []
    if not ik_ok:
        notes.append(f"IK: {orientation_used}")
    if collision:
        notes.append(f"path clearance {clearance*100:.1f}cm < {CLEARANCE_THRESHOLD_M*100:.0f}cm")

    preview = PreviewResult(
        candidate_id=cand.candidate_id,
        ik_ok=ik_ok,
        orientation_used=orientation_used,
        collision=collision,
        min_clearance_m=clearance,
        predicted_ee=Pose(target.position.copy(), target.quat_wxyz.copy()),
        path_world=path,
        notes="; ".join(notes) or "feasible",
    )
    state.add_preview(preview)
    return preview
