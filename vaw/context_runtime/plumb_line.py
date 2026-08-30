"""Deterministic vertical plumb-line geometry for carried payloads.

The plumb line answers one recurring VLM failure: from a single oblique RGB
view, "is the payload horizontally above the target?" is a depth guess.  We
drop a geometric vertical from the payload bottom-centre (or the TCP when no
payload volume is attached), intersect it with the currently observed scene
surface, and render the landing footprint plus the XY offset to the active
target region as explicit pixels.  We intentionally do not print the vertical
height: small decimal labels are an unreliable visual-language interface and
can be read at the wrong scale when the canvas is resized.

Everything here is presentation-only geometry derived from privileged depth
and the attached-object OBB.  It is a geometric vertical and a surface
intersection, never a physics prediction: no tipping, bouncing or sliding is
implied, exactly like the preview gripper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from vaw.context_runtime.geometry import project_world_to_pixel

# Colour families follow the existing canvas language: cyan = current/real,
# violet = unexecuted preview, red = a correction the agent should read.
CURRENT_PLUMB_RGB = (8, 145, 178)
PREVIEW_PLUMB_RGB = (124, 58, 237)
OFFSET_ARROW_RGB = (220, 38, 38)

# A plumb line only exists when the anchor actually hangs above a surface.
_MIN_HEIGHT_M = 0.008
# Offsets below this are visual noise, not a usable correction signal.
_MIN_OFFSET_ARROW_M = 0.005
# On a zoomed contact panel a centimetre is only a few pixels; a sub-12 px
# arrow degenerates into two isolated wings, so we draw a target circle.
_MIN_DXY_ARROW_PX = 12.0
_DASH_LENGTH_M = 0.012
_SURFACE_RADIUS_M = 0.06
_SURFACE_PERCENTILE = 90.0
_DEPTH_STRIDE_TARGET = 200


@dataclass(frozen=True)
class PlumbLine:
    """One computed vertical drop from an anchor to the observed surface."""

    kind: Literal["current", "preview"]
    anchor_base_xyz: tuple[float, float, float]
    landing_base_xyz: tuple[float, float, float]
    height_m: float
    # Bottom-face corners of the payload OBB translated to the landing height,
    # or ``None`` when the anchor is a bare TCP without an attached volume.
    footprint_base: np.ndarray | None
    target_center_base_xyz: tuple[float, float, float] | None
    offset_xy_m: tuple[float, float] | None


def payload_bottom_from_triangles(
    triangles_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bottom-centre xyz, bottom-face corners (4,3)) of an OBB mesh."""

    vertices = np.asarray(triangles_base, dtype=np.float64).reshape(-1, 3)
    corners = np.unique(np.round(vertices, 9), axis=0)
    order = np.argsort(corners[:, 2], kind="stable")
    bottom = corners[order[: max(4, len(corners) // 2)][:4]]
    return bottom.mean(axis=0), bottom


def surface_height_below(
    camera: dict[str, Any],
    anchor_base_xyz: np.ndarray,
    *,
    radius_m: float = _SURFACE_RADIUS_M,
    clearance_m: float = _MIN_HEIGHT_M,
    exclude_mask: np.ndarray | None = None,
) -> float | None:
    """Height of the first observed support surface under an anchor's XY.

    The estimate uses the calibrated depth image only: unproject a strided
    sample, keep points inside an XY cylinder below the anchor, and take a
    high percentile of their heights.  A container rim inside the cylinder
    therefore counts as the support surface, which is the honest answer for
    "what does the payload meet first if it moves straight down".
    """

    anchor = np.asarray(anchor_base_xyz, dtype=np.float64).reshape(3)
    try:
        depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return None
    height, width = depth.shape
    stride = max(1, min(height, width) // _DEPTH_STRIDE_TARGET)
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    sampled = depth[rows, cols]
    valid = np.isfinite(sampled) & (sampled > 0.03) & (sampled < 4.0)
    if exclude_mask is not None:
        mask = np.asarray(exclude_mask, dtype=bool)
        if mask.shape == depth.shape:
            valid &= ~mask[rows, cols]
    if np.count_nonzero(valid) < 8:
        return None
    pixels = np.column_stack(
        (
            cols[valid].astype(np.float64),
            rows[valid].astype(np.float64),
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        )
    )
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    points_camera = rays * sampled[valid, None]
    homogeneous = np.column_stack((points_camera, np.ones(len(points_camera))))
    points_base = (base_from_camera @ homogeneous.T).T[:, :3]
    finite = points_base[np.isfinite(points_base).all(axis=1)]
    lateral = np.linalg.norm(finite[:, :2] - anchor[:2], axis=1)
    below = finite[(lateral <= radius_m) & (finite[:, 2] <= anchor[2] - clearance_m)]
    if len(below) < 4:
        return None
    return float(np.percentile(below[:, 2], _SURFACE_PERCENTILE))


def compute_plumb_line(
    kind: Literal["current", "preview"],
    anchor_base_xyz: np.ndarray,
    surface_z: float | None,
    *,
    footprint_base: np.ndarray | None = None,
    target_center_base_xyz: tuple[float, float, float] | None = None,
) -> PlumbLine | None:
    anchor = np.asarray(anchor_base_xyz, dtype=np.float64).reshape(3)
    if surface_z is None or not np.isfinite(anchor).all():
        return None
    height = float(anchor[2] - surface_z)
    if height < _MIN_HEIGHT_M:
        return None
    landing = np.array([anchor[0], anchor[1], surface_z], dtype=np.float64)
    footprint = None
    if footprint_base is not None:
        footprint = np.asarray(footprint_base, dtype=np.float64).reshape(-1, 3).copy()
        footprint[:, 2] = surface_z
    offset = None
    if target_center_base_xyz is not None:
        offset = (
            float(target_center_base_xyz[0] - landing[0]),
            float(target_center_base_xyz[1] - landing[1]),
        )
    return PlumbLine(
        kind=kind,
        anchor_base_xyz=tuple(float(value) for value in anchor),
        landing_base_xyz=tuple(float(value) for value in landing),
        height_m=height,
        footprint_base=footprint,
        target_center_base_xyz=target_center_base_xyz,
        offset_xy_m=offset,
    )


def draw_plumb_overlays(
    image_rgb: np.ndarray,
    camera: dict[str, Any],
    plumbs: Sequence[PlumbLine],
) -> None:
    """Draw plumb lines, footprints and offset arrows in place."""

    if not plumbs:
        return
    try:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        pose_mat = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return
    pil = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(pil, "RGBA")
    for plumb in plumbs:
        _draw_one(
            draw,
            plumb,
            intrinsics,
            pose_mat,
        )
    np.copyto(image_rgb, np.asarray(pil, dtype=np.uint8))


def _draw_one(
    draw: ImageDraw.ImageDraw,
    plumb: PlumbLine,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
) -> None:
    color = CURRENT_PLUMB_RGB if plumb.kind == "current" else PREVIEW_PLUMB_RGB
    anchor = np.asarray(plumb.anchor_base_xyz, dtype=np.float64)
    landing = np.asarray(plumb.landing_base_xyz, dtype=np.float64)

    # Dashed vertical: sample the 3D segment so dash lengths stay metric.
    segments = max(4, int(np.ceil(plumb.height_m / _DASH_LENGTH_M)))
    samples = np.linspace(anchor, landing, segments * 2 + 1)
    projected = _project(samples, intrinsics, pose_mat)
    if projected is None:
        return
    for index in range(0, len(projected) - 1, 2):
        start, end = projected[index], projected[index + 1]
        draw.line(
            (start[0], start[1], end[0], end[1]),
            fill=(*color, 235),
            width=2,
        )

    landing_px = projected[-1]
    _cross(draw, landing_px, color)

    if plumb.footprint_base is not None and len(plumb.footprint_base) >= 3:
        hull = _footprint_loop(plumb.footprint_base)
        loop = _project(hull, intrinsics, pose_mat)
        if loop is not None:
            draw.polygon(
                [(point[0], point[1]) for point in loop],
                outline=(*color, 235),
                width=2,
            )

    if plumb.offset_xy_m is not None and plumb.target_center_base_xyz is not None:
        offset_norm = float(np.hypot(*plumb.offset_xy_m))
        if offset_norm >= _MIN_OFFSET_ARROW_M:
            target_at_landing = np.array(
                [
                    plumb.target_center_base_xyz[0],
                    plumb.target_center_base_xyz[1],
                    plumb.landing_base_xyz[2],
                ],
                dtype=np.float64,
            )
            arrow = _project(
                np.stack([landing, target_at_landing]), intrinsics, pose_mat
            )
            if arrow is not None:
                span = float(np.linalg.norm(arrow[1] - arrow[0]))
                if span < _MIN_DXY_ARROW_PX:
                    _target_circle(draw, arrow[1], OFFSET_ARROW_RGB)
                else:
                    _arrow(draw, arrow[0], arrow[1], OFFSET_ARROW_RGB)


def _footprint_loop(footprint: np.ndarray) -> np.ndarray:
    """Order footprint corners counter-clockwise around their centroid."""

    corners = np.asarray(footprint, dtype=np.float64).reshape(-1, 3)
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    return corners[np.argsort(angles, kind="stable")]


def _project(
    points_base: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
) -> np.ndarray | None:
    try:
        projected = project_world_to_pixel(points_base, intrinsics, pose_mat)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 1e-6):
        return None
    return projected[:, :2]


def _target_circle(
    draw: ImageDraw.ImageDraw,
    pixel: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int = 8,
) -> None:
    x, y = float(pixel[0]), float(pixel[1])
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(*color, 255),
        width=2,
    )
    _cross(draw, pixel, color, radius=3)


def _cross(
    draw: ImageDraw.ImageDraw,
    pixel: np.ndarray,
    color: tuple[int, int, int],
    *,
    radius: int = 5,
) -> None:
    x, y = float(pixel[0]), float(pixel[1])
    draw.line((x - radius, y, x + radius, y), fill=(*color, 255), width=2)
    draw.line((x, y - radius, x, y + radius), fill=(*color, 255), width=2)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    draw.line(
        (float(start[0]), float(start[1]), float(end[0]), float(end[1])),
        fill=(*color, 255),
        width=3,
    )
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    unit = direction / norm
    normal = np.array([-unit[1], unit[0]])
    tip = np.asarray(end, dtype=np.float64)
    for side in (1.0, -1.0):
        wing = tip - unit * 9.0 + side * normal * 5.0
        draw.line(
            (float(tip[0]), float(tip[1]), float(wing[0]), float(wing[1])),
            fill=(*color, 255),
            width=3,
        )


__all__ = [
    "CURRENT_PLUMB_RGB",
    "OFFSET_ARROW_RGB",
    "PREVIEW_PLUMB_RGB",
    "PlumbLine",
    "compute_plumb_line",
    "draw_plumb_overlays",
    "payload_bottom_from_triangles",
    "surface_height_below",
]
