"""Pure OBB short-axis grasp affordance helper."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _quat_wxyz_from_rotation(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s])
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = np.array([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s])
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = np.array([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s])
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = np.array([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s])
    return _normalize(quat)


def compute_obb_short_axis_grasp(
    points: np.ndarray,
    obb: dict[str, Any],
    *,
    grasp_depth_fraction_from_top: float = 0.5,
    approach_height: float = 0.15,
    lift_height: float = 0.15,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=float)
    center = np.asarray(obb["center"], dtype=float)
    extent = np.asarray(obb["extent"], dtype=float)
    R_obb = np.asarray(obb["R"], dtype=float)
    vertical_axis_index = int(np.argmax(np.abs(R_obb[2, :])))
    horizontal = [i for i in range(3) if i != vertical_axis_index]
    short_axis_index = horizontal[int(np.argmin(extent[horizontal]))]
    short_axis = np.asarray(R_obb[:, short_axis_index], dtype=float)
    short_axis[2] = 0.0
    if np.linalg.norm(short_axis) < 1e-6:
        short_axis = np.array([1.0, 0.0, 0.0])
    short_axis = _normalize(short_axis)
    ee_z_axis = np.array([0.0, 0.0, -1.0])
    ee_x_axis = short_axis
    ee_y_axis = _normalize(np.cross(ee_z_axis, ee_x_axis))
    ee_x_axis = _normalize(np.cross(ee_y_axis, ee_z_axis))
    R_gripper = np.column_stack([ee_x_axis, ee_y_axis, ee_z_axis])
    quat = _quat_wxyz_from_rotation(R_gripper)
    body_height = float(extent[vertical_axis_index])
    top_z = float(np.max(pts[:, 2])) if len(pts) else float(center[2] + 0.5 * body_height)
    pos = center.copy()
    pos[2] = top_z - float(grasp_depth_fraction_from_top) * body_height
    lift_pos = pos.copy()
    lift_pos[2] += float(lift_height)
    approach_pos = pos.copy()
    approach_pos[2] += float(approach_height)
    return {
        "ok": True,
        "strategy": "obb_short_axis_topdown",
        "pos": pos.tolist(),
        "position": pos.tolist(),
        "quat": quat.tolist(),
        "quaternion_wxyz": quat.tolist(),
        "approach_position": approach_pos.tolist(),
        "lift_pos": lift_pos.tolist(),
        "lift_position": lift_pos.tolist(),
        "obb_center": center.tolist(),
        "obb_extent": extent.tolist(),
        "vertical_axis_index": vertical_axis_index,
        "short_axis_index": short_axis_index,
        "short_axis": short_axis.tolist(),
        "num_points": int(len(pts)),
    }
