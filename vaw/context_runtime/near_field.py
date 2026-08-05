"""Deterministic, policy-visible gripper-near geometry raster.

The raster is reconstructed only from the current revision's calibrated RGB-D
observations and the robot's proprioceptive URDF geometry.  Raw depth, camera
matrices and point arrays remain inside the trusted presenter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from vaw.context_runtime.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    rasterize_silhouette,
)
from vaw.context_runtime.model import RobotState

NEAR_FIELD_WIDTH = 480
NEAR_FIELD_HEIGHT = 540

_PANEL_GAP = 8
_PANEL_HEIGHT = (NEAR_FIELD_HEIGHT - _PANEL_GAP) // 2
_BACKGROUND = np.array([232, 239, 247], dtype=np.uint8)
_GRIPPER = np.array([37, 99, 235], dtype=np.uint8)
_GRIPPER_OUTLINE = np.array([23, 55, 130], dtype=np.uint8)
_MAX_POINTS = 80_000


@dataclass(frozen=True)
class _View:
    label: str
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    center: np.ndarray
    half_width_m: float
    half_height_m: float


def render_near_field(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState | None,
) -> np.ndarray | None:
    """Render current fused RGB-D around the contact TCP from two fixed views."""

    if (
        robot is None
        or robot.tcp_pose is None
        or robot.joint_positions_rad is None
        or robot.gripper_opening is None
    ):
        return None

    tcp_position = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64)
    tcp_quaternion = np.asarray(robot.tcp_pose.quaternion_xyzw, dtype=np.float64)
    if not (
        np.isfinite(tcp_position).all()
        and np.isfinite(tcp_quaternion).all()
        and np.linalg.norm(tcp_quaternion) > 1e-12
    ):
        return None
    tcp_rotation = Rotation.from_quat(
        tcp_quaternion / np.linalg.norm(tcp_quaternion)
    ).as_matrix()

    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for camera in (agentview, wrist):
        if camera is None:
            continue
        sampled = _colored_points_base(camera)
        if sampled is None:
            continue
        points, colors = sampled
        local = (points - tcp_position) @ tcp_rotation
        keep = (
            (np.abs(local[:, 0]) <= 0.18)
            & (np.abs(local[:, 1]) <= 0.18)
            & (local[:, 2] >= -0.17)
            & (local[:, 2] <= 0.14)
        )
        if np.any(keep):
            point_parts.append(local[keep])
            color_parts.append(colors[keep])

    if point_parts:
        points_local = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        if len(points_local) > _MAX_POINTS:
            indices = np.linspace(
                0, len(points_local) - 1, num=_MAX_POINTS, dtype=np.int64
            )
            points_local = points_local[indices]
            colors = colors[indices]
    else:
        points_local = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)

    triangles_local = _gripper_triangles_local(robot, tcp_position, tcp_rotation)
    views = _near_field_views()
    panels = [
        _render_view(points_local, colors, triangles_local, view)
        for view in views
    ]
    canvas = np.full(
        (NEAR_FIELD_HEIGHT, NEAR_FIELD_WIDTH, 3), _BACKGROUND, dtype=np.uint8
    )
    canvas[:_PANEL_HEIGHT] = panels[0]
    second_top = _PANEL_HEIGHT + _PANEL_GAP
    canvas[second_top : second_top + _PANEL_HEIGHT] = panels[1]
    canvas[_PANEL_HEIGHT:second_top] = np.array([207, 217, 231], dtype=np.uint8)
    return canvas


def _colored_points_base(camera: dict) -> tuple[np.ndarray, np.ndarray] | None:
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
        or rgb.shape[:2] != depth.shape
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(base_from_camera).all()
    ):
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        return None

    rows, cols = np.indices(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth >= 0.015) & (depth <= 20.0)
    if not np.any(valid):
        return None
    z = depth[valid]
    points_camera = np.column_stack(
        (
            (cols[valid] - cx) * z / fx,
            (rows[valid] - cy) * z / fy,
            z,
            np.ones_like(z),
        )
    )
    points_base = (base_from_camera @ points_camera.T).T[:, :3]
    finite = np.isfinite(points_base).all(axis=1)
    return points_base[finite], np.ascontiguousarray(rgb[valid][finite])


def _gripper_triangles_local(
    robot: RobotState,
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        triangles = fk.triangles(
            np.asarray(robot.joint_positions_rad, dtype=np.float64),
            float(robot.gripper_opening),
        )
    except (RuntimeError, ValueError):
        return None
    return (triangles - tcp_position) @ tcp_rotation


def _near_field_views() -> tuple[_View, _View]:
    forward = _unit(np.array([0.72, -0.92, 0.62], dtype=np.float64))
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    # Franka local +Z runs from panda_hand toward the fingertip/contact TCP.
    # Put that contact side visually *down* in both panels, so the wrist/palm
    # stays above the fingers and support surfaces appear below the gripper.
    up = -_unit(np.cross(right, forward))
    return (
        _View(
            label="LOCAL 3/4 · CURRENT FUSED RGB-D",
            forward=forward,
            right=right,
            up=up,
            center=np.array([0.0, 0.0, -0.015], dtype=np.float64),
            half_width_m=0.155,
            half_height_m=0.09,
        ),
        _View(
            label="JAW PLANE · CURRENT FUSED RGB-D",
            forward=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            right=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            up=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            center=np.array([0.0, 0.0, -0.015], dtype=np.float64),
            half_width_m=0.13,
            half_height_m=0.135,
        ),
    )


def _render_view(
    points: np.ndarray,
    colors: np.ndarray,
    triangles: np.ndarray | None,
    view: _View,
) -> np.ndarray:
    height, width = _PANEL_HEIGHT, NEAR_FIELD_WIDTH
    image = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    _draw_metric_grid(image, view)
    if len(points):
        u, v, depth = _project(points, view, width, height)
        visible = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(depth)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        _splat_points(
            image,
            u[visible],
            v[visible],
            depth[visible],
            colors[visible],
            radius=2,
        )

    if triangles is not None and len(triangles):
        flat = triangles.reshape(-1, 3)
        u, v, depth = _project(flat, view, width, height)
        projected = np.column_stack((u, v)).reshape(-1, 3, 2)
        triangle_depths = depth.reshape(-1, 3)
        mask = rasterize_silhouette(
            projected,
            triangle_depths,
            width,
            height,
        )
        if np.any(mask):
            blended = (
                image[mask].astype(np.float64) * 0.25
                + _GRIPPER.astype(np.float64) * 0.75
            )
            image[mask] = np.asarray(np.round(blended), dtype=np.uint8)
            image[mask_outline(mask)] = _GRIPPER_OUTLINE

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _label_font()
    draw.rounded_rectangle(
        (8, 8, 286, 31),
        radius=4,
        fill=(255, 255, 255, 226),
        outline=(190, 204, 221, 255),
        width=1,
    )
    draw.text((15, 12), view.label, fill=(30, 41, 59, 255), font=font)
    _draw_contact_axis(draw, view, width)
    _draw_scale_bar(draw, view, width, height)
    return np.asarray(pil, dtype=np.uint8)


def _project(
    points: np.ndarray,
    view: _View,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = np.asarray(points, dtype=np.float64) - view.center
    horizontal = relative @ view.right
    vertical = relative @ view.up
    # The virtual camera is one metre behind the view centre.  The absolute
    # distance is arbitrary, but positive depth is required by the shared
    # triangle rasterizer.
    depth = 1.0 + relative @ view.forward
    u = width * 0.5 + horizontal * (width * 0.5 / view.half_width_m)
    v = height * 0.5 - vertical * (height * 0.5 / view.half_height_m)
    return u, v, depth


def _splat_points(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    colors: np.ndarray,
    *,
    radius: int,
) -> None:
    height, width = image.shape[:2]
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    center_u = np.rint(u).astype(np.int64)
    center_v = np.rint(v).astype(np.int64)
    offsets = (
        (du, dv)
        for dv in range(-radius, radius + 1)
        for du in range(-radius, radius + 1)
        if du * du + dv * dv <= radius * radius
    )
    flat_image = image.reshape(-1, 3)
    for du, dv in offsets:
        xs = center_u + du
        ys = center_v + dv
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not np.any(valid):
            continue
        indices = np.flatnonzero(valid)
        flat = ys[valid] * width + xs[valid]
        order = np.argsort(depth[valid], kind="stable")
        _, first = np.unique(flat[order], return_index=True)
        chosen = order[first]
        target = flat[chosen]
        source = indices[chosen]
        nearer = depth[source] < zbuffer[target]
        if np.any(nearer):
            target = target[nearer]
            source = source[nearer]
            zbuffer[target] = depth[source]
            flat_image[target] = colors[source]


def _draw_metric_grid(image: np.ndarray, view: _View) -> None:
    height, width = image.shape[:2]
    spacing_m = 0.05
    x_spacing = max(1, int(round(spacing_m * width * 0.5 / view.half_width_m)))
    y_spacing = max(1, int(round(spacing_m * height * 0.5 / view.half_height_m)))
    color = np.array([216, 226, 237], dtype=np.uint8)
    for x in range(width // 2 % x_spacing, width, x_spacing):
        image[:, x : x + 1] = color
    for y in range(height // 2 % y_spacing, height, y_spacing):
        image[y : y + 1, :] = color


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    view: _View,
    width: int,
    height: int,
) -> None:
    length_px = int(round(0.05 * width * 0.5 / view.half_width_m))
    right = width - 14
    left = right - length_px
    y = height - 17
    draw.line((left, y, right, y), fill=(30, 41, 59, 230), width=3)
    draw.line((left, y - 4, left, y + 4), fill=(30, 41, 59, 230), width=2)
    draw.line((right, y - 4, right, y + 4), fill=(30, 41, 59, 230), width=2)
    draw.text((left, y - 18), "5 cm", fill=(30, 41, 59, 255), font=_label_font())


def _draw_contact_axis(
    draw: ImageDraw.ImageDraw,
    view: _View,
    width: int,
) -> None:
    local_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = np.array(
        [
            float(local_z @ view.right),
            -float(local_z @ view.up),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return
    direction /= norm
    start = np.array([width - 34.0, 13.0], dtype=np.float64)
    end = start + direction * 24.0
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    wing_a = end - direction * 7.0 + normal * 3.5
    wing_b = end - direction * 7.0 - normal * 3.5
    draw.text(
        (width - 128, 11),
        "+Z CONTACT",
        fill=(30, 64, 175, 255),
        font=_label_font(),
    )
    draw.line((*start, *end), fill=(37, 99, 235, 255), width=3)
    draw.polygon(
        [tuple(end), tuple(wing_a), tuple(wing_b)],
        fill=(37, 99, 235, 255),
    )


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", 11)
    except OSError:
        return ImageFont.load_default()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("view axis must be non-zero")
    return vector / norm


__all__ = [
    "NEAR_FIELD_HEIGHT",
    "NEAR_FIELD_WIDTH",
    "render_near_field",
]
