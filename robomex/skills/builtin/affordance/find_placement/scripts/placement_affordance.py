"""Deterministic GaP-style placement geometry helpers.

The public affordance ``position`` is always the executable TCP drop target.
``desired_object_center`` is retained as evidence, never as a motion target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def estimate_placement_affordance(
    points: np.ndarray,
    *,
    target_name: str,
    evidence: dict[str, Any] | None = None,
    mode: str | None = None,
    place_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
    held_object_frame: dict[str, Any] | None = None,
    grasp_ee_z: float | None = None,
    drop_clearance: float = 0.05,
    approach_height: float = 0.20,
) -> dict[str, Any]:
    """Compatibility entry point for :func:`compute_drop_affordance`."""

    return compute_drop_affordance(
        points,
        target_name=target_name,
        evidence=evidence,
        mode=mode,
        place_quat=place_quat,
        held_object_frame=held_object_frame,
        grasp_ee_z=grasp_ee_z,
        drop_clearance=drop_clearance,
        approach_height=approach_height,
    )


def estimate_container_zone(
    points: np.ndarray,
    mode: str,
    *,
    interior_lip: float = 0.02,
) -> dict[str, Any]:
    """Estimate a bounded placement zone from target points.

    For an open container, ``zone_floor`` models a shallow interior lip and
    ``zone_ceiling`` is the rim.  For a support surface both are anchored to
    the live top surface.
    """

    pts = _validate_points(points)
    if mode not in {"open_container", "support_surface"}:
        raise ValueError(f"unsupported placement mode: {mode!r}")

    z05 = float(np.percentile(pts[:, 2], 5))
    z90 = float(np.percentile(pts[:, 2], 90))
    z95 = float(np.percentile(pts[:, 2], 95))
    rim_cut = float(np.percentile(pts[:, 2], 80))
    rim_points = pts[pts[:, 2] >= rim_cut]

    if mode == "open_container":
        center_xy = np.median(rim_points[:, :2], axis=0) if len(rim_points) >= 30 else np.median(pts[:, :2], axis=0)
        refined = _free_space_center(pts, rim_points)
        strategy = "rim_center"
        if refined is not None:
            center_xy = refined
            strategy = "topdown_free_space_grid"
        zone_floor = min(z05 + max(0.0, float(interior_lip)), z95 - 0.001)
        zone_ceiling = z95
    else:
        core = _trimmed_core(pts)
        center_xy = np.median(core[:, :2], axis=0)
        zone_floor = z90
        zone_ceiling = z90
        strategy = "support_surface_center"

    return {
        "mode": mode,
        "center_xy": _list(center_xy),
        "zone_floor": float(zone_floor),
        "zone_ceiling": float(zone_ceiling),
        "rim_top": z95,
        "strategy": strategy,
        "point_count": int(len(pts)),
    }


def compute_drop_affordance(
    points: np.ndarray,
    *,
    target_name: str,
    evidence: dict[str, Any] | None = None,
    mode: str | None = None,
    place_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
    held_object_frame: dict[str, Any] | None = None,
    grasp_ee_z: float | None = None,
    drop_clearance: float = 0.05,
    approach_height: float = 0.20,
) -> dict[str, Any]:
    """Compute object-in-zone geometry and the corresponding TCP drop pose."""

    target_mode = mode or infer_place_mode(target_name)
    zone = estimate_container_zone(points, target_mode)
    held = held_object_frame or {}
    half_height = _held_half_height(held)
    center_xy = np.asarray(zone["center_xy"], dtype=float)

    if target_mode == "open_container":
        margin = max(0.03, float(drop_clearance))
        desired_z = float(zone["zone_floor"]) + margin + half_height
        desired_z = min(desired_z, float(zone["zone_ceiling"]) - 0.001)
        desired_z = max(desired_z, float(zone["zone_floor"]) + 0.001)
    else:
        margin = max(0.005, min(float(drop_clearance), 0.045))
        desired_z = float(zone["zone_floor"]) + margin + half_height

    desired_object_center = np.array([center_xy[0], center_xy[1], desired_z], dtype=float)
    tcp_position = _tcp_from_held_frame(
        desired_object_center,
        held,
        grasp_ee_z=grasp_ee_z,
        zone_floor=float(zone["zone_floor"]),
        bottom_margin=margin,
    )
    quat = _yaw_only_topdown(held.get("grasp_quaternion_wxyz"), place_quat)
    approach_position = tcp_position.copy()
    approach_position[2] += float(approach_height)

    affordance = {
        "position": _list(tcp_position),
        "quaternion_wxyz": _list(quat),
        "approach_dir_world": [0.0, 0.0, -1.0],
        "target": target_name,
        "mode": target_mode,
        "desired_object_center": _list(desired_object_center),
        "approach_position": _list(approach_position),
        "approach_height": float(approach_height),
        "zone_floor": float(zone["zone_floor"]),
        "zone_ceiling": float(zone["zone_ceiling"]),
        "rim_top": float(zone["rim_top"]),
        "object_half_height": half_height,
        "tcp_compensated": bool(held),
        "strategy": f"gap_style_{zone['strategy']}",
        "note": "position is the executable TCP drop target; desired_object_center is evidence only",
    }
    if evidence is not None:
        evidence["placement_affordance"] = affordance
    return affordance


def _held_half_height(held: dict[str, Any]) -> float:
    for height_key in ("height", "object_height"):
        value = held.get(height_key)
        if value is not None and float(value) > 0:
            return 0.5 * float(value)
    top = held.get("top_z")
    bottom = held.get("bottom_z")
    if top is not None and bottom is not None and float(top) > float(bottom):
        return 0.5 * (float(top) - float(bottom))
    return 0.03


def _tcp_from_held_frame(
    desired_object_center: np.ndarray,
    held: dict[str, Any],
    *,
    grasp_ee_z: float | None,
    zone_floor: float,
    bottom_margin: float,
) -> np.ndarray:
    tcp = desired_object_center.copy()
    offset = held.get("object_center_offset_from_grasp")
    if offset is not None:
        values = np.asarray(offset, dtype=float).reshape(3)
        if np.isfinite(values).all():
            tcp = desired_object_center - values

    object_at_grasp = held.get("object_center_at_grasp")
    if grasp_ee_z is not None and object_at_grasp is not None:
        object_z = float(np.asarray(object_at_grasp, dtype=float).reshape(3)[2])
        tcp[2] = desired_object_center[2] + float(grasp_ee_z) - object_z
    elif offset is None and held.get("tcp_to_bottom") is not None:
        tcp[2] = zone_floor + bottom_margin + float(held["tcp_to_bottom"])
    return tcp


def _yaw_only_topdown(
    grasp_quaternion_wxyz: Any,
    default: tuple[float, float, float, float],
) -> np.ndarray:
    if grasp_quaternion_wxyz is None:
        return np.asarray(default, dtype=float)
    q = np.asarray(grasp_quaternion_wxyz, dtype=float).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        return np.asarray(default, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return np.array([0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0], dtype=float)


def save_placement_overlay(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    desired_object_center: list[float] | np.ndarray,
    artifacts_dir: str,
    evidence: dict[str, Any] | None = None,
    bbox: list[float] | None = None,
    filename: str = "placement_point_overlay.png",
) -> str:
    """Save a compact 2D review overlay for the selected placement point."""

    import cv2
    from PIL import Image, ImageDraw

    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    out = str(art / filename)
    height, width = rgb.shape[:2]
    uv = project_world_to_pixel(desired_object_center, camera_intrinsics, camera_pose, (height, width))
    if uv is None:
        uv = _fallback_uv(bbox, width, height)

    vis = np.asarray(rgb).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    img = Image.fromarray(vis)
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 0, 0), width=2)
    u, v = int(round(uv[0])), int(round(uv[1]))
    r = 7
    draw.ellipse((u - r, v - r, u + r, v + r), outline=(0, 255, 255), width=3)
    draw.line((u - r, v, u + r, v), fill=(0, 255, 255), width=2)
    draw.line((u, v - r, u, v + r), fill=(0, 255, 255), width=2)
    img.save(out)
    if evidence is not None:
        evidence.setdefault("placement_affordance", {}).setdefault("artifacts", {})["overlay"] = out
    return out


def save_placement_3d_visualization(
    points: np.ndarray,
    placement_affordance: dict[str, Any],
    artifacts_dir: str,
    *,
    held_object_frame: dict[str, Any] | None = None,
    filename: str = "placement_3d.png",
) -> str:
    """Save a 3D review plot for the selected placement affordance."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = _validate_points(points)
    desired = np.asarray(placement_affordance["desired_object_center"], dtype=float)
    if placement_affordance.get("position") is not None:
        tcp = np.asarray(placement_affordance["position"], dtype=float)
    elif held_object_frame and held_object_frame.get("object_center_offset_from_grasp") is not None:
        tcp = desired - np.asarray(held_object_frame["object_center_offset_from_grasp"], dtype=float)
    else:
        tcp = desired.copy()
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    out = str(art / filename)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sample = pts
    if len(sample) > 2500:
        idx = np.linspace(0, len(sample) - 1, 2500).astype(int)
        sample = sample[idx]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=4, c="#888888", alpha=0.35, label="target points")
    ax.scatter([desired[0]], [desired[1]], [desired[2]], s=90, c="#00c8ff", marker="o", label="desired object center")
    ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], s=70, c="#ff3b30", marker="^", label="tcp drop pose")
    ax.plot([tcp[0], desired[0]], [tcp[1], desired[1]], [tcp[2], desired[2]], c="#ff3b30", linewidth=2)
    ax.quiver(
        tcp[0],
        tcp[1],
        tcp[2] + 0.08,
        0,
        0,
        -0.06,
        color="#ff8c00",
        linewidth=2,
        arrow_length_ratio=0.25,
        label="top-down release",
    )

    mode = placement_affordance.get("mode", "unknown")
    strategy = placement_affordance.get("strategy", "unknown")
    ax.set_title(f"placement affordance: {mode} / {strategy}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    _set_equal_3d(ax, np.vstack([sample, desired.reshape(1, 3), tcp.reshape(1, 3)]))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    placement_affordance.setdefault("artifacts", {})["3d_visualization"] = out
    return out


def infer_place_mode(target_name: str) -> str:
    text = target_name.lower()
    if any(word in text for word in ("basket", "bin", "container", "bowl", "cup")):
        return "open_container"
    return "support_surface"


def project_world_to_pixel(
    point: list[float] | np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    image_shape: tuple[int, int],
) -> list[float] | None:
    height, width = image_shape
    k = np.asarray(intrinsics, dtype=float)
    t_wc = np.asarray(camera_pose, dtype=float)
    p = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
    p_cam = np.linalg.inv(t_wc) @ p
    for z in (p_cam[2], -p_cam[2]):
        if abs(z) < 1e-6:
            continue
        u = k[0, 0] * (p_cam[0] / z) + k[0, 2]
        v = k[1, 1] * (p_cam[1] / z) + k[1, 2]
        if np.isfinite(u) and np.isfinite(v) and 0 <= u < width and 0 <= v < height:
            return [float(u), float(v)]
    return None


def _validate_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise ValueError(f"need at least 20 finite placement points, got {len(pts)}")
    return pts


def _trimmed_core(points: np.ndarray) -> np.ndarray:
    lo = np.percentile(points, 10, axis=0)
    hi = np.percentile(points, 90, axis=0)
    core = points[np.all((points >= lo) & (points <= hi), axis=1)]
    return core if len(core) >= 30 else points


def _free_space_center(points: np.ndarray, rim_points: np.ndarray) -> np.ndarray | None:
    grid_res = 0.015
    safe_margin = 0.035
    xy = points[:, :2]
    xy_min = np.percentile(xy, 5, axis=0)
    xy_max = np.percentile(xy, 95, axis=0)
    xs = np.arange(xy_min[0], xy_max[0] + grid_res, grid_res)
    ys = np.arange(xy_min[1], xy_max[1] + grid_res, grid_res)
    if len(xs) < 3 or len(ys) < 3:
        return None
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx, gy], axis=-1).reshape(-1, 2)
    unsafe = rim_points[:, :2] if len(rim_points) >= 30 else xy
    d2 = ((grid[:, None, :] - unsafe[None, :, :]) ** 2).sum(axis=2)
    min_dist = np.sqrt(d2.min(axis=1))
    best_i = int(np.argmax(min_dist))
    if min_dist[best_i] <= safe_margin:
        return None
    return grid[best_i]


def _fallback_uv(bbox: list[float] | None, width: int, height: int) -> list[float]:
    if bbox is None:
        return [float(width // 2), float(height // 2)]
    return [
        float(np.clip((bbox[0] + bbox[2]) / 2, 0, width - 1)),
        float(np.clip((bbox[1] + bbox[3]) / 2, 0, height - 1)),
    ]


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _list(arr: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(arr, dtype=float).reshape(-1)]
