"""Pure robust geometry summary for segmented world-frame points."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_object_geometry(points: np.ndarray) -> dict[str, Any]:
    """Estimate robust AABB/PCA geometry without changing world state."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise ValueError("need at least 20 finite object points")

    lower = np.percentile(pts, 2.0, axis=0)
    upper = np.percentile(pts, 98.0, axis=0)
    core = pts[np.all((pts >= lower) & (pts <= upper), axis=1)]
    if len(core) < 20:
        core = pts
    center = np.median(core, axis=0)
    covariance = np.cov((core - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    local = (core - center) @ rotation
    local_lower = np.percentile(local, 2.0, axis=0)
    local_upper = np.percentile(local, 98.0, axis=0)
    extent = local_upper - local_lower
    aabb_min = np.percentile(core, 2.0, axis=0)
    aabb_max = np.percentile(core, 98.0, axis=0)
    height = float(aabb_max[2] - aabb_min[2])
    diameter = float(max(aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1]))
    vertical_alignment = np.abs(rotation[2, :])
    pose_hint = "upright" if float(np.max(vertical_alignment)) >= 0.75 else "lying_or_tilted"
    return {
        "center": center.tolist(),
        "extent": extent.tolist(),
        "rotation_matrix": rotation.tolist(),
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "height_meters": height,
        "diameter_meters": diameter,
        "pose_hint": pose_hint,
        "num_points": int(len(core)),
    }
