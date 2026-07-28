"""Pure PCA side-grasp affordance construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_pca_side_grasps(
    points: np.ndarray,
    *,
    approach_distance: float = 0.12,
    lift_height: float = 0.15,
) -> dict[str, Any]:
    """Return two body-centered side-grasp candidates from finite world points."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise ValueError("need at least 20 finite points")
    if approach_distance <= 0.0 or lift_height <= 0.0:
        raise ValueError("approach_distance and lift_height must be positive")

    center = np.median(pts, axis=0)
    covariance = np.cov((pts - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    body_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    body_axis[2] = 0.0
    if np.linalg.norm(body_axis) < 1e-8:
        body_axis = np.array([1.0, 0.0, 0.0])
    body_axis /= np.linalg.norm(body_axis)
    side_axis = np.array([-body_axis[1], body_axis[0], 0.0])

    grasp = center.copy()
    grasp[2] = float(np.clip(center[2], np.percentile(pts[:, 2], 30), np.percentile(pts[:, 2], 70)))
    candidates = []
    for sign in (1.0, -1.0):
        approach_axis = sign * side_axis
        approach = grasp - approach_axis * float(approach_distance)
        lift = grasp + np.array([0.0, 0.0, float(lift_height)])
        quaternion = _side_quaternion(approach_axis)
        candidates.append(
            {
                "strategy": "pca_side",
                "position": grasp.tolist(),
                "pos": grasp.tolist(),
                "quaternion_wxyz": quaternion,
                "quat": quaternion,
                "approach_position": approach.tolist(),
                "lift_position": lift.tolist(),
                "lift_pos": lift.tolist(),
                "approach_axis": approach_axis.tolist(),
                "body_axis": body_axis.tolist(),
                "score": 1.0,
            }
        )
    return {"ok": True, "candidates": candidates, "selected_candidate": candidates[0]}


def _side_quaternion(approach_axis: np.ndarray) -> list[float]:
    """Build WXYZ quaternion whose tool z axis follows the side approach."""

    z_axis = np.asarray(approach_axis, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ]
    else:
        # scipy-free robust conversion for the non-positive-trace branch.
        index = int(np.argmax(np.diag(rotation)))
        next_indices = ((1, 2), (0, 2), (0, 1))
        j, k = next_indices[index]
        scale = math.sqrt(
            max(0.0, 1.0 + rotation[index, index] - rotation[j, j] - rotation[k, k])
        ) * 2.0
        xyz = [0.0, 0.0, 0.0]
        xyz[index] = 0.25 * scale
        xyz[j] = (rotation[j, index] + rotation[index, j]) / scale
        xyz[k] = (rotation[k, index] + rotation[index, k]) / scale
        quat = [
            (rotation[k, j] - rotation[j, k]) / scale,
            xyz[0],
            xyz[1],
            xyz[2],
        ]
    norm = math.sqrt(sum(value * value for value in quat))
    return [float(value / norm) for value in quat]
