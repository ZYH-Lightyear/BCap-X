"""Geometry helpers for open-bowl grasp affordances.

The functions here are intentionally pure NumPy. They do not move the robot; Act can
import them after loading the open-bowl skill to propose top-down rim grasp candidates
from segmented world-frame points.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def propose_open_bowl_grasps(
    points: np.ndarray,
    *,
    top_quantile: float = 0.72,
    z_margins: Sequence[float] | None = None,
    radius_scale: float = 0.97,
    standoff_m: float = 0.075,
    num_angles: int = 8,
    solve_ik_fn: Callable[[np.ndarray, np.ndarray], Any] | None = None,
) -> dict[str, Any]:
    """Return grasp candidates for an upward-facing open bowl.

    Args:
        points: ``(N, 3)`` segmented target points in world frame.
        top_quantile: Quantile used to estimate the rim band.
        z_margins: Candidate vertical offsets below the top rim. Multiple offsets are
            useful because the best pinch depth depends on rim thickness and geometry.
        radius_scale: Radial scale from observed rim support points toward the bowl
            center. Values slightly below 1 keep the grasp just inside the rim instead
            of projecting a synthetic circle outside the object.
        standoff_m: Suggested pregrasp distance along the negative approach axis.
        num_angles: Number of rim directions to sample.
        solve_ik_fn: Optional IK checker. If provided, each candidate gets ``ik_ok``.

    Returns:
        A dict with ``geometry`` and ranked ``candidates``. Candidate quaternions are
        ``wxyz`` and positions are in the same world frame as ``points``.
        The best pose is available as ``out["selected_candidate"]``. Each candidate
        uses these stable keys: ``pos`` for grasp TCP position, ``quat`` for WXYZ
        quaternion, ``object_center`` for the estimated object center at grasp
        height, ``object_center_offset_from_grasp`` for center-aware placement,
        ``pregrasp_pos`` for approach, and ``ik_ok`` / ``ik_error`` for feasibility.
    """

    pts = _validate_points(points)
    bottom_z = float(np.percentile(pts[:, 2], 3))
    top_z = float(np.percentile(pts[:, 2], 97))
    rim_z_cut = float(np.quantile(pts[:, 2], top_quantile))
    rim = pts[pts[:, 2] >= rim_z_cut]
    if len(rim) < 12:
        rim = pts

    center_xy = np.median(rim[:, :2], axis=0)
    all_center_xy = np.median(pts[:, :2], axis=0)
    # Blend rim and full-mask centers. This is more stable when the visible rim is partial.
    center_xy = 0.7 * center_xy + 0.3 * all_center_xy
    rim_xy = rim[:, :2]
    radial = rim_xy - center_xy
    distances = np.linalg.norm(radial, axis=1)
    radius = float(np.percentile(distances, 85))
    radius = max(radius, 0.025)

    rim_height = float(np.percentile(rim[:, 2], 90))
    margins = _normalize_margins(z_margins if z_margins is not None else (0.004, 0.010, 0.016))

    angles = _principal_ordered_angles(rim_xy, center_xy, num_angles)
    support_points = _observed_rim_support_points(rim_xy, center_xy, angles, radius_scale)
    candidates: list[dict[str, Any]] = []
    for margin_rank, margin in enumerate(margins):
        grasp_z = max(bottom_z + 0.025, rim_height - float(margin))
        for angle_rank, support in enumerate(support_points):
            point_xy = support["point_xy"]
            rim_point_xy = support["rim_point_xy"]
            theta = float(support["theta"])
            outward = np.array([point_xy[0] - center_xy[0], point_xy[1] - center_xy[1], 0.0], dtype=float)
            outward = _unit(outward)
            tangent = np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=float)
            approach = np.array([0.0, 0.0, -1.0], dtype=float)
            pos = np.array([point_xy[0], point_xy[1], grasp_z], dtype=float)
            object_center = np.array([center_xy[0], center_xy[1], grasp_z], dtype=float)
            center_offset = object_center - pos
            quat = _quat_from_axes(tangent, outward, approach)
            candidate = {
                "strategy": "open_bowl_rim_topdown",
                "pos": _list(pos),
                "quat": _list(quat),
                "object_center": _list(object_center),
                "object_center_offset_from_grasp": _list(center_offset),
                "approach_axis": _list(approach),
                "jaw_axis": _list(tangent),
                "radial_axis": _list(outward),
                "pregrasp_pos": _list(pos - approach * standoff_m),
                "source_rim_point_xy": _list(rim_point_xy),
                "rim_z_margin": float(margin),
                "score": float(1.0 - 0.035 * angle_rank - 0.025 * margin_rank),
                "ik_ok": True,
                "ik_error": None,
                "notes": [
                    "top-down rim pinch for upward-facing bowl",
                    "approach_axis points downward from above the rim",
                    "jaw_axis is tangent to the rim and orthogonal to radial_axis",
                    "rim_z_margin is the candidate depth below the estimated rim",
                    "for placement, align object_center to the target center, not the rim grasp point",
                ],
            }
            if solve_ik_fn is not None:
                try:
                    solve_ik_fn(np.asarray(candidate["pos"], dtype=float), np.asarray(candidate["quat"], dtype=float))
                    candidate["ik_ok"] = True
                except Exception as exc:  # noqa: BLE001 - IK failure is a candidate attribute.
                    candidate["ik_ok"] = False
                    candidate["ik_error"] = repr(exc)
                    candidate["score"] = float(candidate["score"]) - 0.35
            candidates.append(candidate)

    candidates.sort(key=lambda c: (bool(c.get("ik_ok")), float(c.get("score", 0.0))), reverse=True)
    selected = candidates[0] if candidates else None
    return {
        "ok": bool(candidates),
        "recommended_strategy": "open_bowl_rim_topdown",
        "selected_candidate": selected,
        "candidates": candidates,
        "geometry": {
            "center": _list(
                np.array(
                    [
                        center_xy[0],
                        center_xy[1],
                        float(selected["pos"][2]) if selected is not None else rim_height,
                    ],
                    dtype=float,
                )
            ),
            "center_xy": _list(center_xy),
            "bottom_z": bottom_z,
            "top_z": top_z,
            "rim_z_cut": rim_z_cut,
            "rim_height": rim_height,
            "candidate_z_margins": [float(v) for v in margins],
            "estimated_radius": radius,
            "radius_scale": float(radius_scale),
            "num_points": int(len(pts)),
            "num_rim_points": int(len(rim)),
        },
    }


def _validate_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise ValueError(f"need at least 20 finite points for bowl grasp, got {len(pts)}")
    return pts


def _normalize_margins(values: Sequence[float]) -> tuple[float, ...]:
    margins = sorted({round(float(v), 6) for v in values if np.isfinite(float(v)) and float(v) >= 0.0})
    if not margins:
        return (0.004,)
    return tuple(margins)


def _observed_rim_support_points(
    rim_xy: np.ndarray,
    center_xy: np.ndarray,
    angles: Sequence[float],
    radius_scale: float,
) -> list[dict[str, np.ndarray | float]]:
    """Pick candidate xy positions from observed rim points instead of a synthetic circle."""

    rel = np.asarray(rim_xy, dtype=float) - np.asarray(center_xy, dtype=float)
    dist = np.linalg.norm(rel, axis=1)
    valid = dist > 1e-6
    if not np.any(valid):
        raise ValueError("rim points collapse to center; cannot choose bowl rim grasps")
    rel = rel[valid]
    rim_valid = np.asarray(rim_xy, dtype=float)[valid]
    dist = dist[valid]
    dirs = rel / np.maximum(dist[:, None], 1e-9)
    scale = float(np.clip(radius_scale, 0.80, 1.02))

    supports: list[dict[str, np.ndarray | float]] = []
    for theta in angles:
        target = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        alignment = dirs @ target
        # Prefer points facing this angle, then the outermost observed rim point in that sector.
        sector = alignment >= max(0.35, float(np.cos(np.pi / max(4, len(angles)))))
        candidates = np.flatnonzero(sector)
        if len(candidates) == 0:
            candidates = np.arange(len(rim_valid))
        # Projection rewards the far side of the observed rim; perpendicular penalty avoids
        # choosing a distant point from another sector when the mask is partial.
        projection = rel[candidates] @ target
        perpendicular = np.abs(rel[candidates] @ np.array([-target[1], target[0]], dtype=float))
        local_score = projection - 0.25 * perpendicular
        idx = candidates[int(np.argmax(local_score))]
        rim_point = rim_valid[idx]
        point_xy = np.asarray(center_xy, dtype=float) + (rim_point - center_xy) * scale
        actual_theta = float(np.arctan2(point_xy[1] - center_xy[1], point_xy[0] - center_xy[0]))
        supports.append({"theta": actual_theta, "point_xy": point_xy, "rim_point_xy": rim_point})
    return supports


def _principal_ordered_angles(xy: np.ndarray, center_xy: np.ndarray, num_angles: int) -> list[float]:
    rel = np.asarray(xy, dtype=float) - np.asarray(center_xy, dtype=float)
    if len(rel) >= 3:
        cov = np.cov(rel.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        base = float(np.arctan2(axis[1], axis[0]))
    else:
        base = 0.0
    step = 2.0 * np.pi / max(1, num_angles)
    # Alternate opposing sides first, then fill intermediate angles.
    order: list[float] = []
    for k in range((num_angles + 1) // 2):
        order.append(base + k * step)
        if len(order) < num_angles:
            order.append(base + np.pi + k * step)
    return order[:num_angles]


def _quat_from_axes(x_axis: np.ndarray, y_hint: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    z = _unit(z_axis)
    x = np.asarray(x_axis, dtype=float)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(y_hint, z)
    x = _unit(x)
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    r = np.column_stack([x, y, z])
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r[:, 0] *= -1.0
    return _matrix_to_quat_wxyz(r)


def _matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / max(1e-9, float(np.linalg.norm(quat)))


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    return arr / max(1e-9, float(np.linalg.norm(arr)))


def _list(arr: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(arr, dtype=float).reshape(-1)]
