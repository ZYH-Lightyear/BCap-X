"""Observed RGB-D surface reconstruction for policy-visible contact views.

Camera-aligned RGB-D looks dense because neighbouring sensor pixels remain
neighbours in the raster.  A novel orthographic view loses that property when
each depth sample is drawn as an isolated point.  This module preserves the
sensor's pixel adjacency as a small textured triangle mesh, while rejecting
triangles across depth discontinuities.  It fills only surfaces actually
observed by a camera; it never completes hidden geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RgbdSurface:
    """One camera's observed surface expressed in a caller-provided frame."""

    triangles_local: np.ndarray
    colors_rgb: np.ndarray


def reconstruct_rgbd_surface(
    camera: dict,
    *,
    frame_position_base: np.ndarray,
    frame_rotation_base: np.ndarray,
    half_extent_m: tuple[float, float, float],
    emphasis_mask: np.ndarray | None = None,
) -> RgbdSurface | None:
    """Build a deterministic textured mesh from one calibrated RGB-D image."""

    parsed = _camera_arrays(camera)
    if parsed is None:
        return None
    rgb, depth, intrinsics, base_from_camera = parsed
    height, width = depth.shape

    # An 832 px contact panel does not benefit from retaining substantially more
    # than ~50k source vertices.  Grid decimation keeps complete cells (and thus
    # continuous surfaces), unlike selecting an arbitrary subset of triangles.
    stride = max(1, int(np.ceil(np.sqrt((height * width) / 50_000.0))))
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    rows = rows[::stride, ::stride]
    cols = cols[::stride, ::stride]
    sampled_depth = depth[::stride, ::stride]
    sampled_rgb = rgb[::stride, ::stride]

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    valid = np.isfinite(sampled_depth) & (sampled_depth >= 0.015) & (sampled_depth <= 20.0)
    safe_depth = np.where(valid, sampled_depth, 1.0)
    points_camera = np.stack(
        (
            (cols - cx) * safe_depth / fx,
            (rows - cy) * safe_depth / fy,
            safe_depth,
            np.ones_like(safe_depth),
        ),
        axis=-1,
    )
    points_base = (points_camera @ base_from_camera.T)[..., :3]
    frame_position = np.asarray(frame_position_base, dtype=np.float64).reshape(3)
    frame_rotation = np.asarray(frame_rotation_base, dtype=np.float64).reshape(3, 3)
    points_local = (points_base - frame_position) @ frame_rotation
    valid &= np.isfinite(points_local).all(axis=-1)

    extent = np.asarray(half_extent_m, dtype=np.float64).reshape(3)
    # Keep a small guard band so triangles crossing the crop boundary do not
    # create a visibly clipped surface one pixel inside the panel.
    valid &= np.all(np.abs(points_local) <= extent * 1.08, axis=-1)
    sampled_emphasis = None
    if emphasis_mask is not None:
        mask = np.asarray(emphasis_mask, dtype=bool)
        if mask.shape == depth.shape:
            sampled_emphasis = mask[::stride, ::stride]
    colors = _policy_colors(sampled_rgb, sampled_emphasis)

    if points_local.shape[0] < 2 or points_local.shape[1] < 2:
        return None
    triangle_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    validity_parts: list[np.ndarray] = []
    for indices in (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 1), (0, 1)),
    ):
        vertices = np.stack(
            tuple(
                points_local[
                    row : points_local.shape[0] - 1 + row,
                    col : points_local.shape[1] - 1 + col,
                ]
                for row, col in indices
            ),
            axis=2,
        )
        vertex_depths = np.stack(
            tuple(
                sampled_depth[
                    row : sampled_depth.shape[0] - 1 + row,
                    col : sampled_depth.shape[1] - 1 + col,
                ]
                for row, col in indices
            ),
            axis=2,
        )
        vertex_valid = np.stack(
            tuple(
                valid[
                    row : valid.shape[0] - 1 + row,
                    col : valid.shape[1] - 1 + col,
                ]
                for row, col in indices
            ),
            axis=2,
        )
        vertex_colors = np.stack(
            tuple(
                colors[
                    row : colors.shape[0] - 1 + row,
                    col : colors.shape[1] - 1 + col,
                ]
                for row, col in indices
            ),
            axis=2,
        )
        triangle_parts.append(vertices.reshape(-1, 3, 3))
        depth_parts.append(vertex_depths.reshape(-1, 3))
        validity_parts.append(vertex_valid.reshape(-1, 3).all(axis=1))
        color_parts.append(
            np.asarray(
                np.round(vertex_colors.reshape(-1, 3, 3).mean(axis=1)),
                dtype=np.uint8,
            )
        )

    triangles = np.concatenate(triangle_parts, axis=0)
    triangle_depths = np.concatenate(depth_parts, axis=0)
    triangle_colors = np.concatenate(color_parts, axis=0)
    keep = np.concatenate(validity_parts, axis=0)

    # Reject discontinuity bridges using the local metric footprint of one
    # sampled source pixel.  The threshold scales with depth and grid stride,
    # so it remains meaningful for both wrist close-ups and distant agentview.
    mean_depth = triangle_depths.mean(axis=1)
    pixel_footprint = mean_depth * stride * max(1.0 / abs(fx), 1.0 / abs(fy))
    depth_limit = np.maximum(0.005, pixel_footprint * 4.0)
    edge_limit = np.maximum(0.007, pixel_footprint * 5.0)
    keep &= np.ptp(triangle_depths, axis=1) <= depth_limit
    edges = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    keep &= edges.max(axis=1) <= edge_limit
    keep &= (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        > 1e-10
    )
    if not np.any(keep):
        return None
    return RgbdSurface(
        triangles_local=np.ascontiguousarray(triangles[keep]),
        colors_rgb=np.ascontiguousarray(triangle_colors[keep]),
    )


def rasterize_rgbd_surfaces(
    image: np.ndarray,
    surfaces: tuple[RgbdSurface, ...],
    *,
    center_local: np.ndarray,
    forward_local: np.ndarray,
    right_local: np.ndarray,
    up_local: np.ndarray,
    half_width_m: float,
    half_height_m: float,
) -> None:
    """Paint observed surface triangles into an orthographic RGB raster."""

    if not surfaces:
        return
    triangles = np.concatenate([surface.triangles_local for surface in surfaces], axis=0)
    colors = np.concatenate([surface.colors_rgb for surface in surfaces], axis=0)
    height, width = image.shape[:2]
    relative = triangles - np.asarray(center_local, dtype=np.float64).reshape(1, 1, 3)
    horizontal = relative @ np.asarray(right_local, dtype=np.float64).reshape(3)
    vertical = relative @ np.asarray(up_local, dtype=np.float64).reshape(3)
    depth = 1.0 + relative @ np.asarray(forward_local, dtype=np.float64).reshape(3)
    projected = np.empty((len(triangles), 3, 2), dtype=np.float64)
    projected[..., 0] = width * 0.5 + horizontal * (width * 0.5 / half_width_m)
    projected[..., 1] = height * 0.5 - vertical * (height * 0.5 / half_height_m)

    finite = np.isfinite(projected).all(axis=(1, 2)) & np.isfinite(depth).all(axis=1)
    finite &= depth.min(axis=1) > 1e-6
    x_min = projected[..., 0].min(axis=1)
    x_max = projected[..., 0].max(axis=1)
    y_min = projected[..., 1].min(axis=1)
    y_max = projected[..., 1].max(axis=1)
    finite &= (x_max >= 0) & (y_max >= 0) & (x_min < width) & (y_min < height)
    area = np.abs(
        (projected[:, 1, 0] - projected[:, 0, 0]) * (projected[:, 2, 1] - projected[:, 0, 1])
        - (projected[:, 1, 1] - projected[:, 0, 1]) * (projected[:, 2, 0] - projected[:, 0, 0])
    )
    finite &= area >= 0.10
    indices = np.flatnonzero(finite)
    if len(indices) == 0:
        return

    # A stable far-to-near painter pass is sufficient for these locally smooth,
    # discontinuity-split camera surfaces and lets the nearer camera win where
    # agentview and wrist observe the same patch.
    order = indices[np.argsort(depth[indices].mean(axis=1), kind="stable")[::-1]]
    pixels = np.rint(projected).astype(np.int32)
    for index in order:
        cv2.fillConvexPoly(
            image,
            pixels[index],
            tuple(int(value) for value in colors[index]),
            lineType=cv2.LINE_8,
        )


def _camera_arrays(
    camera: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        images = camera["images"]
        rgb = np.asarray(images["rgb"], dtype=np.uint8)
        depth = np.asarray(images["depth"], dtype=np.float64)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return None
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if (
        depth.ndim != 2
        or rgb.shape != (*depth.shape, 3)
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(base_from_camera).all()
        or abs(float(intrinsics[0, 0])) <= 1e-12
        or abs(float(intrinsics[1, 1])) <= 1e-12
    ):
        return None
    return rgb, depth, intrinsics, base_from_camera


def _policy_colors(
    rgb: np.ndarray,
    emphasis_mask: np.ndarray | None,
) -> np.ndarray:
    colors = np.ascontiguousarray(rgb).copy()
    if emphasis_mask is None:
        return colors
    mask = np.asarray(emphasis_mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        return colors
    background = ~mask
    colors[background] = np.asarray(
        np.round(colors[background].astype(np.float64) * 0.50 + 92.0),
        dtype=np.uint8,
    )
    cyan = np.array([14.0, 165.0, 233.0], dtype=np.float64)
    colors[mask] = np.asarray(
        np.round(colors[mask].astype(np.float64) * 0.52 + cyan * 0.48),
        dtype=np.uint8,
    )
    return colors


__all__ = [
    "RgbdSurface",
    "rasterize_rgbd_surfaces",
    "reconstruct_rgbd_surface",
]
