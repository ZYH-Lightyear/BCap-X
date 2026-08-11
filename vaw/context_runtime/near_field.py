"""Deterministic, policy-visible gripper-near geometry raster.

The raster is reconstructed only from the current revision's calibrated RGB-D
observations and the robot's proprioceptive URDF geometry.  Raw depth, camera
matrices and point arrays remain inside the trusted presenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from vaw.context_runtime.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    rasterize_silhouette,
)
from vaw.context_runtime.model import Pose, RobotState
from vaw.context_runtime.private import VisualEdit

NEAR_FIELD_WIDTH = 640
NEAR_FIELD_HEIGHT = 720

_PANEL_GAP = 8
_PANEL_HEIGHT = (NEAR_FIELD_HEIGHT - _PANEL_GAP) // 2
CONTACT_FOCUS_WIDTH = NEAR_FIELD_WIDTH
CONTACT_FOCUS_HEIGHT = _PANEL_HEIGHT
_BACKGROUND = np.array([232, 239, 247], dtype=np.uint8)
_GRIPPER = np.array([37, 99, 235], dtype=np.uint8)
_GRIPPER_OUTLINE = np.array([23, 55, 130], dtype=np.uint8)
_PREVIEW = np.array([124, 58, 237], dtype=np.uint8)
_PREVIEW_OUTLINE = np.array([76, 29, 149], dtype=np.uint8)
_MOVE = (22, 163, 74, 255)
_AXIS_COLORS = (
    (220, 38, 38, 255),
    (22, 163, 74, 255),
    (37, 99, 235, 255),
)
_MAX_POINTS_OBSERVED = 80_000
_MAX_POINTS_IMAGINATION = 160_000


@dataclass(frozen=True)
class _View:
    label: str
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    center: np.ndarray
    half_width_m: float
    half_height_m: float


@dataclass(frozen=True)
class NearFieldPreview:
    """Geometry-only view of the active, unexecuted Waypoint."""

    target_pose: Pose | None
    joint_positions_rad: tuple[float, ...] | None
    gripper_opening: float | None
    visual_edit: VisualEdit | None = None


def render_near_field(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState | None,
    preview: NearFieldPreview | None = None,
) -> np.ndarray | None:
    """Render current fused RGB-D around the contact TCP from two fixed views."""

    if (
        robot is None
        or robot.tcp_pose is None
        or robot.joint_positions_rad is None
        or robot.gripper_opening is None
    ):
        return None

    return _render_geometry_pair(agentview, wrist, robot, preview)


def render_contact_focus(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState | None,
    preview: NearFieldPreview | None,
    *,
    source_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Render the orthogonal jaw-plane evidence for one virtual target.

    The camera-aligned imagination scene preserves global context, but it does
    not reveal whether an object is actually inside the two-finger channel.
    This panel uses the same current RGB-D and exact URDF geometry and never
    predicts object motion or task effects.
    """

    if preview is None:
        return None
    pair = _render_geometry_pair(
        agentview,
        wrist,
        robot,
        preview,
        agentview_emphasis_mask=source_mask,
    )
    if pair is None:
        return None
    top = _PANEL_HEIGHT + _PANEL_GAP
    return np.ascontiguousarray(pair[top : top + _PANEL_HEIGHT])


def _render_geometry_pair(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState,
    preview: NearFieldPreview | None,
    *,
    agentview_emphasis_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    if robot.tcp_pose is None or robot.joint_positions_rad is None or robot.gripper_opening is None:
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
    focus_center_local = np.array([0.0, 0.0, -0.015], dtype=np.float64)
    if preview is not None and preview.target_pose is not None:
        target_position_local, _ = _pose_in_current_tcp(
            preview.target_pose,
            tcp_position,
            tcp_rotation,
        )
        # Keep the observed and virtual TCP in the same local comparison view.
        focus_center_local = target_position_local * 0.5
        focus_center_local[2] -= 0.015

    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for camera_index, camera in enumerate((agentview, wrist)):
        if camera is None:
            continue
        sampled = _colored_points_base(
            camera,
            emphasis_mask=(
                agentview_emphasis_mask if camera_index == 0 else None
            ),
        )
        if sampled is None:
            continue
        points, colors = sampled
        local = (points - tcp_position) @ tcp_rotation
        relative_to_focus = local - focus_center_local
        keep = (
            (np.abs(relative_to_focus[:, 0]) <= 0.22)
            & (np.abs(relative_to_focus[:, 1]) <= 0.22)
            & (np.abs(relative_to_focus[:, 2]) <= 0.20)
        )
        if np.any(keep):
            point_parts.append(local[keep])
            color_parts.append(colors[keep])

    if point_parts:
        points_local = np.concatenate(point_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        point_limit = (
            _MAX_POINTS_IMAGINATION if preview is not None else _MAX_POINTS_OBSERVED
        )
        if len(points_local) > point_limit:
            indices = np.linspace(
                0, len(points_local) - 1, num=point_limit, dtype=np.int64
            )
            points_local = points_local[indices]
            colors = colors[indices]
    else:
        points_local = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)

    triangles_local = _gripper_triangles_local(
        robot.joint_positions_rad,
        robot.gripper_opening,
        tcp_position,
        tcp_rotation,
    )
    preview_triangles_local = None
    preview_is_target_ghost = False
    if (
        preview is not None
        and preview.joint_positions_rad is not None
        and preview.gripper_opening is not None
    ):
        preview_triangles_local = _gripper_triangles_local(
            preview.joint_positions_rad,
            preview.gripper_opening,
            tcp_position,
            tcp_rotation,
        )
    elif (
        preview is not None
        and preview.target_pose is not None
        and preview.gripper_opening is not None
    ):
        target_shape_local = _gripper_triangles_local(
            robot.joint_positions_rad,
            preview.gripper_opening,
            tcp_position,
            tcp_rotation,
        )
        if target_shape_local is not None:
            target_position_local, target_rotation_local = _pose_in_current_tcp(
                preview.target_pose,
                tcp_position,
                tcp_rotation,
            )
            preview_triangles_local = (
                target_shape_local @ target_rotation_local.T
                + target_position_local
            )
            preview_is_target_ghost = True
    views = _near_field_views(
        "CURRENT + PREVIEW" if preview is not None else "OBSERVED NOW",
        focus_center_local,
    )
    panels = [
        _render_view(
            points_local,
            colors,
            triangles_local,
            preview_triangles_local,
            view,
            current_tcp_position=tcp_position,
            current_tcp_rotation=tcp_rotation,
            preview=preview,
            preview_is_target_ghost=preview_is_target_ghost,
            axes_kind="base" if index == 0 else "tool",
        )
        for index, view in enumerate(views)
    ]
    canvas = np.full(
        (NEAR_FIELD_HEIGHT, NEAR_FIELD_WIDTH, 3), _BACKGROUND, dtype=np.uint8
    )
    canvas[:_PANEL_HEIGHT] = panels[0]
    second_top = _PANEL_HEIGHT + _PANEL_GAP
    canvas[second_top : second_top + _PANEL_HEIGHT] = panels[1]
    canvas[_PANEL_HEIGHT:second_top] = np.array([207, 217, 231], dtype=np.uint8)
    return canvas


def _colored_points_base(
    camera: dict,
    *,
    emphasis_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
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
    colors = np.ascontiguousarray(rgb[valid][finite]).copy()
    if emphasis_mask is not None:
        mask = np.asarray(emphasis_mask, dtype=bool)
        if mask.shape == depth.shape:
            emphasized = np.ascontiguousarray(mask[valid][finite])
            background = ~emphasized
            colors[background] = np.asarray(
                np.round(colors[background].astype(np.float64) * 0.50 + 92.0),
                dtype=np.uint8,
            )
            cyan = np.array([14.0, 165.0, 233.0], dtype=np.float64)
            colors[emphasized] = np.asarray(
                np.round(colors[emphasized].astype(np.float64) * 0.52 + cyan * 0.48),
                dtype=np.uint8,
            )
    return points_base[finite], colors


def _gripper_triangles_local(
    joints_rad: tuple[float, ...],
    opening: float,
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        triangles = fk.triangles(
            np.asarray(joints_rad, dtype=np.float64),
            float(opening),
        )
    except (RuntimeError, ValueError):
        return None
    return (triangles - tcp_position) @ tcp_rotation


def _near_field_views(
    label_suffix: str,
    center: np.ndarray | None = None,
) -> tuple[_View, _View]:
    view_center = (
        np.array([0.0, 0.0, -0.015], dtype=np.float64)
        if center is None
        else np.asarray(center, dtype=np.float64).reshape(3)
    )
    forward = _unit(np.array([0.72, -0.92, 0.62], dtype=np.float64))
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    # Franka local +Z runs from panda_hand toward the fingertip/contact TCP.
    # Put that contact side visually *down* in both panels, so the wrist/palm
    # stays above the fingers and support surfaces appear below the gripper.
    up = -_unit(np.cross(right, forward))
    return (
        _View(
            label=f"LOCAL 3/4 · {label_suffix}",
            forward=forward,
            right=right,
            up=up,
            center=view_center,
            half_width_m=0.155,
            half_height_m=0.115 if label_suffix == "CURRENT + PREVIEW" else 0.09,
        ),
        _View(
            label=f"JAW PLANE · {label_suffix}",
            forward=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            right=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            up=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            center=view_center,
            half_width_m=0.13,
            half_height_m=0.135,
        ),
    )


def _render_view(
    points: np.ndarray,
    colors: np.ndarray,
    current_triangles: np.ndarray | None,
    preview_triangles: np.ndarray | None,
    view: _View,
    *,
    current_tcp_position: np.ndarray,
    current_tcp_rotation: np.ndarray,
    preview: NearFieldPreview | None,
    preview_is_target_ghost: bool,
    axes_kind: Literal["base", "tool"],
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

    if current_triangles is not None and len(current_triangles):
        _overlay_triangles(
            image,
            current_triangles,
            view,
            _GRIPPER,
            _GRIPPER_OUTLINE,
            alpha=0.62,
        )
    if preview_triangles is not None and len(preview_triangles):
        _overlay_triangles(
            image,
            preview_triangles,
            view,
            _PREVIEW,
            _PREVIEW_OUTLINE,
            alpha=0.52 if preview_is_target_ghost else 0.76,
        )

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _label_font()
    draw.rounded_rectangle(
        (8, 8, 330, 31),
        radius=4,
        fill=(255, 255, 255, 226),
        outline=(190, 204, 221, 255),
        width=1,
    )
    draw.text((15, 12), view.label, fill=(30, 41, 59, 255), font=font)
    axes_rotation_local = current_tcp_rotation.T
    axes_label = "BASE / WORLD"
    if axes_kind == "tool":
        axes_rotation_local = np.eye(3, dtype=np.float64)
        axes_label = "CURRENT TOOL"
        if preview is not None and preview.target_pose is not None:
            _, axes_rotation_local = _pose_in_current_tcp(
                preview.target_pose,
                current_tcp_position,
                current_tcp_rotation,
            )
            axes_label = "TARGET TOOL"
    _draw_corner_axes(
        draw,
        view,
        axes_rotation_local,
        width,
        label=axes_label,
        origin=(width - 100.0, 112.0),
    )
    if preview is not None and preview.target_pose is not None:
        _draw_adjustment(
            draw,
            view,
            width,
            height,
            preview,
            current_tcp_position,
            current_tcp_rotation,
        )
    _draw_legend(draw, width, height, preview is not None)
    _draw_scale_bar(draw, view, width, height)
    return np.asarray(pil, dtype=np.uint8)


def _overlay_triangles(
    image: np.ndarray,
    triangles: np.ndarray,
    view: _View,
    color: np.ndarray,
    outline_color: np.ndarray,
    *,
    alpha: float,
) -> None:
    if triangles is not None and len(triangles):
        flat = triangles.reshape(-1, 3)
        height, width = image.shape[:2]
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
                image[mask].astype(np.float64) * (1.0 - alpha)
                + color.astype(np.float64) * alpha
            )
            image[mask] = np.asarray(np.round(blended), dtype=np.uint8)
            image[mask_outline(mask)] = outline_color


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


def _pose_in_current_tcp(
    pose: Pose,
    current_tcp_position: np.ndarray,
    current_tcp_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_position = np.asarray(pose.position_xyz, dtype=np.float64)
    target_rotation = Rotation.from_quat(
        np.asarray(pose.quaternion_xyzw, dtype=np.float64)
    ).as_matrix()
    position_local = (target_position - current_tcp_position) @ current_tcp_rotation
    rotation_local = current_tcp_rotation.T @ target_rotation
    return position_local, rotation_local


def _draw_corner_axes(
    draw: ImageDraw.ImageDraw,
    view: _View,
    rotation_local: np.ndarray,
    width: int,
    *,
    label: str,
    origin: tuple[float, float],
) -> None:
    del width
    accent = (30, 64, 175, 255)
    title = label
    title_font = _label_font()
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    plate_left = origin[0] - 86
    # Reserve a real title band above every projected +axis label.  Axis
    # labels may extend about 70 px from the origin; the earlier 62 px margin
    # let +X overlap TARGET TOOL for near-vertical poses.
    plate_top = origin[1] - 95
    plate_right = max(origin[0] + 86, plate_left + title_width + 14)
    plate_bottom = origin[1] + 52
    draw.rounded_rectangle(
        (plate_left, plate_top, plate_right, plate_bottom),
        radius=6,
        fill=(255, 255, 255, 228),
        outline=accent,
        width=2,
    )
    draw.text(
        (plate_left + 7, plate_top + 6),
        title,
        fill=accent,
        font=title_font,
    )
    draw.ellipse(
        (origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5),
        fill=(255, 255, 255, 255),
        outline=accent,
        width=3,
    )
    for index, (color, axis_name) in enumerate(
        zip(_AXIS_COLORS, "XYZ", strict=True)
    ):
        axis_label = f"+{axis_name}"
        axis_local = rotation_local[:, index]
        direction = np.array(
            [float(axis_local @ view.right), -float(axis_local @ view.up)],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        axis_color = (*color[:3], 255)
        if norm <= 0.08:
            draw.ellipse(
                (
                    origin[0] - 7,
                    origin[1] - 7,
                    origin[0] + 7,
                    origin[1] + 7,
                ),
                outline=axis_color,
                width=3,
            )
            draw.ellipse(
                (
                    origin[0] - 2,
                    origin[1] - 2,
                    origin[0] + 2,
                    origin[1] + 2,
                ),
                fill=axis_color,
            )
            _draw_axis_label_box(
                draw,
                (origin[0] + 20, origin[1]),
                axis_label,
                axis_color,
                title_font,
            )
            continue
        unit = direction / norm
        endpoint_array = np.asarray(origin) + unit * 44.0
        endpoint = (float(endpoint_array[0]), float(endpoint_array[1]))
        draw.line((*origin, *endpoint), fill=axis_color, width=6)
        _draw_arrow_head(draw, origin, endpoint, axis_color, size=11.0)
        label_anchor = endpoint_array + unit * 13.0
        _draw_axis_label_box(
            draw,
            (float(label_anchor[0]), float(label_anchor[1])),
            axis_label,
            axis_color,
            title_font,
        )


def _draw_axis_label_box(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[float, float],
    label: str,
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox((0, 0), label, font=font)
    width = bounds[2] - bounds[0] + 10
    height = bounds[3] - bounds[1] + 7
    left = anchor[0] - width * 0.5
    top = anchor[1] - height * 0.5
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=3,
        fill=(255, 255, 255, 232),
        outline=color,
        width=2,
    )
    draw.text(
        (left + 5, top + 3 - bounds[1]),
        label,
        fill=color,
        font=font,
    )


def _draw_adjustment(
    draw: ImageDraw.ImageDraw,
    view: _View,
    width: int,
    height: int,
    preview: NearFieldPreview,
    current_tcp_position: np.ndarray,
    current_tcp_rotation: np.ndarray,
) -> None:
    edit = preview.visual_edit
    target = preview.target_pose
    if (
        edit is None
        or target is None
        or edit.kind not in {"delta_move", "rotate"}
        or edit.reference_pose is None
    ):
        return
    reference_position_local, _ = _pose_in_current_tcp(
        edit.reference_pose,
        current_tcp_position,
        current_tcp_rotation,
    )
    target_position_local, _ = _pose_in_current_tcp(
        target,
        current_tcp_position,
        current_tcp_rotation,
    )
    if edit.kind == "delta_move" and edit.delta_xyz_m is not None:
        points = np.vstack([reference_position_local, target_position_local])
        u, v, _ = _project(points, view, width, height)
        if np.isfinite(np.column_stack((u, v))).all():
            start = (float(u[0]), float(v[0]))
            end = (float(u[1]), float(v[1]))
            draw.line((*start, *end), fill=_MOVE, width=6)
            _draw_arrow_head(draw, start, end, _MOVE, size=11.0)
        delta_cm = np.asarray(edit.delta_xyz_m, dtype=np.float64) * 100.0
        label = (
            f"MOVE {edit.frame.upper()}  "
            f"dX {delta_cm[0]:+.1f}  dY {delta_cm[1]:+.1f}  dZ {delta_cm[2]:+.1f} cm"
        )
        _draw_adjustment_label(draw, label, width, height, _MOVE)
        return
    if edit.kind != "rotate" or edit.axis is None or edit.angle_deg is None:
        return

    axis_index = "xyz".index(edit.axis)
    axis_unit = np.eye(3, dtype=np.float64)[axis_index]
    reference_rotation = Rotation.from_quat(
        np.asarray(edit.reference_pose.quaternion_xyzw, dtype=np.float64)
    ).as_matrix()
    axis_base = axis_unit if edit.frame == "base" else reference_rotation @ axis_unit
    radial_base = (
        np.eye(3, dtype=np.float64)[(axis_index + 1) % 3]
        if edit.frame == "base"
        else reference_rotation[:, (axis_index + 1) % 3]
    )
    center_base = np.asarray(edit.reference_pose.position_xyz, dtype=np.float64)
    radians = np.deg2rad(float(edit.angle_deg))
    samples = np.linspace(0.0, radians, num=25, dtype=np.float64)
    arc_base = np.vstack(
        [
            center_base
            + Rotation.from_rotvec(axis_base * angle).apply(radial_base * 0.045)
            for angle in samples
        ]
    )
    arc_local = (arc_base - current_tcp_position) @ current_tcp_rotation
    u, v, _ = _project(arc_local, view, width, height)
    finite = np.isfinite(u) & np.isfinite(v)
    arc_2d = [(float(x), float(y)) for x, y in zip(u[finite], v[finite], strict=True)]
    axis_color = _AXIS_COLORS[axis_index]
    if len(arc_2d) >= 2:
        draw.line(arc_2d, fill=axis_color, width=6, joint="curve")
        _draw_arrow_head(draw, arc_2d[-2], arc_2d[-1], axis_color, size=11.0)
    label = f"ROTATE {edit.frame.upper()}  {edit.axis.upper()} {edit.angle_deg:+.1f} deg"
    _draw_adjustment_label(draw, label, width, height, axis_color)


def _draw_adjustment_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> None:
    font = _label_font()
    box = draw.textbbox((0, 0), label, font=font)
    label_width = box[2] - box[0]
    top = height - 54
    draw.rounded_rectangle(
        (10, top, min(width - 10, label_width + 30), top + 31),
        radius=5,
        fill=(255, 255, 255, 235),
        outline=color,
        width=2,
    )
    draw.text((19, top + 7), label, fill=color, font=font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    has_preview: bool,
) -> None:
    labels = [((37, 99, 235, 235), "BLUE CURRENT")]
    if has_preview:
        labels.append(((124, 58, 237, 235), "PURPLE PREVIEW"))
    x = 10
    y = height - 20
    for color, label in labels:
        draw.rectangle((x, y - 1, x + 12, y + 11), fill=color)
        draw.text((x + 17, y - 3), label, fill=(30, 41, 59, 255), font=_label_font())
        x += 126


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    *,
    size: float,
) -> None:
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return
    unit = direction / norm
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    tip = np.asarray(end, dtype=np.float64)
    base = tip - unit * size
    draw.polygon(
        [tuple(tip), tuple(base + normal * size * 0.48), tuple(base - normal * size * 0.48)],
        fill=color,
    )


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("view axis must be non-zero")
    return vector / norm


__all__ = [
    "CONTACT_FOCUS_HEIGHT",
    "CONTACT_FOCUS_WIDTH",
    "NEAR_FIELD_HEIGHT",
    "NEAR_FIELD_WIDTH",
    "NearFieldPreview",
    "render_contact_focus",
    "render_near_field",
]
