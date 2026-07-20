"""Pure top-grasp offset helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_top_grasp_with_tcp_to_bottom_offset(
    points: np.ndarray,
    *,
    obb: dict[str, Any] | None = None,
    z_grasp_offset: float = 0.04,
    approach_height: float = 0.15,
    lift_height: float = 0.20,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return {"ok": False, "reason": "empty_points"}
    top_z = float(np.percentile(pts[:, 2], 95))
    bottom_z = float(np.percentile(pts[:, 2], 5))
    if obb is not None and "center" in obb:
        center = np.asarray(obb["center"], dtype=float)
    else:
        center = np.median(pts, axis=0)
    pos = np.array([center[0], center[1], top_z - float(z_grasp_offset)], dtype=float)
    object_center = np.array([center[0], center[1], 0.5 * (top_z + bottom_z)], dtype=float)
    lift_pos = pos.copy()
    lift_pos[2] += float(lift_height)
    approach_pos = pos.copy()
    approach_pos[2] += float(approach_height)
    quat = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
    return {
        "ok": True,
        "strategy": "top_grasp_with_tcp_to_bottom_offset",
        "pos": pos.tolist(),
        "position": pos.tolist(),
        "quat": quat.tolist(),
        "quaternion_wxyz": quat.tolist(),
        "approach_position": approach_pos.tolist(),
        "lift_pos": lift_pos.tolist(),
        "lift_position": lift_pos.tolist(),
        "top_z": top_z,
        "bottom_z": bottom_z,
        "tcp_to_bottom": float(pos[2] - bottom_z),
        "object_center_at_grasp": object_center.tolist(),
        "object_center_offset_from_grasp": (object_center - pos).tolist(),
        "num_points": int(len(pts)),
    }
