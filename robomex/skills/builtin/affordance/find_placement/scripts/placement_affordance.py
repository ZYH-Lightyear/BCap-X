"""Placement affordance helpers for top-down place actions."""

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
) -> dict[str, Any]:
    """Estimate where the held object's center should go.

    Returns a compact affordance and writes it to ``evidence`` when provided.
    ``mode`` is either ``open_container`` or ``support_surface``.
    """

    pts = _validate_points(points)
    target_mode = mode or infer_place_mode(target_name)
    if target_mode not in {"open_container", "support_surface"}:
        raise ValueError(f"unsupported placement mode: {target_mode!r}")

    z80 = np.percentile(pts[:, 2], 80)
    rim_points = pts[pts[:, 2] >= z80]
    if target_mode == "open_container" and len(rim_points) >= 30:
        xy_source = rim_points[:, :2]
        top_z = float(np.percentile(rim_points[:, 2], 75))
        strategy = "rim_top_percentile_center"
    else:
        core = _trimmed_core(pts)
        xy_source = core[:, :2]
        top_z = float(np.percentile(pts[:, 2], 90))
        strategy = "support_surface_center" if target_mode == "support_surface" else "trimmed_point_cloud_center"

    center_xy = np.median(xy_source, axis=0)
    if target_mode == "open_container":
        refined = _free_space_center(pts, rim_points)
        if refined is not None:
            center_xy = refined
            strategy = "topdown_free_space_grid"

    clearance = 0.10 if target_mode == "open_container" else 0.045
    desired_object_center = np.array([center_xy[0], center_xy[1], top_z + clearance], dtype=float)
    affordance = {
        "target": target_name,
        "mode": target_mode,
        "desired_object_center": _list(desired_object_center),
        "place_quat": [float(v) for v in place_quat],
        "strategy": strategy,
    }
    if evidence is not None:
        evidence["placement_affordance"] = affordance
    return affordance


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
    ax.quiver(
        desired[0],
        desired[1],
        desired[2] + 0.08,
        0,
        0,
        -0.06,
        color="#ff8c00",
        linewidth=2,
        arrow_length_ratio=0.25,
        label="top-down release",
    )

    if held_object_frame and held_object_frame.get("object_center_offset_from_grasp") is not None:
        offset = np.asarray(held_object_frame["object_center_offset_from_grasp"], dtype=float)
        tcp = desired - offset
        ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], s=70, c="#ff3b30", marker="^", label="tcp release pos")
        ax.plot([tcp[0], desired[0]], [tcp[1], desired[1]], [tcp[2], desired[2]], c="#ff3b30", linewidth=2)

    mode = placement_affordance.get("mode", "unknown")
    strategy = placement_affordance.get("strategy", "unknown")
    ax.set_title(f"placement affordance: {mode} / {strategy}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="best")
    _set_equal_3d(ax, np.vstack([sample, desired.reshape(1, 3)]))
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
