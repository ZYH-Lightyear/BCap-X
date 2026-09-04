"""Deterministic, policy-visible gripper-near geometry raster.

Live LIBERO uses two direct MuJoCo Contact Cameras so the current scene has the
same dense raster quality and occlusion semantics as agentview.  Calibrated
RGB-D reprojection remains only as the offline fallback.  Raw depth, camera
matrices and geometry remain inside the trusted presenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_dilation
from scipy.spatial.transform import Rotation

from vaw.context_runtime.contact_camera import ContactCameraPair
from vaw.context_runtime.geometry import project_world_to_pixel
from vaw.context_runtime.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    overlay_projected_mesh_outline,
    rasterize_silhouette,
    semantic_parallel_jaw_triangles,
)
from vaw.context_runtime.model import Pose, RobotState
from vaw.context_runtime.plumb_line import PlumbLine, draw_plumb_overlays
from vaw.context_runtime.private import VisualEdit
from vaw.context_runtime.rgbd_surface import (
    RgbdSurface,
    rasterize_rgbd_surfaces,
    reconstruct_rgbd_surface,
)

NEAR_FIELD_WIDTH = 832
NEAR_FIELD_HEIGHT = 720

CONTACT_PANEL_GAP = 4
CONTACT_PANEL_HEIGHT = (NEAR_FIELD_HEIGHT - CONTACT_PANEL_GAP) // 2
CONTACT_FOCUS_WIDTH = NEAR_FIELD_WIDTH
CONTACT_FOCUS_HEIGHT = NEAR_FIELD_HEIGHT
_BACKGROUND = np.array([232, 239, 247], dtype=np.uint8)
_SOURCE_FILL = np.array([245, 158, 11], dtype=np.uint8)
_SOURCE_EDGE = np.array([253, 230, 138], dtype=np.uint8)
_CURRENT_FINGER_FILL = np.array([34, 211, 238], dtype=np.uint8)
_CURRENT_FINGER_EDGE = np.array([165, 243, 252], dtype=np.uint8)
_CURRENT_FINGER_ALPHA = 0.31
# Same cyan family as the fingers because both are current real robot surface,
# but darker so the palm floor reads as a separate part, not more finger.
_PALM_FLOOR_FILL = np.array([14, 116, 144], dtype=np.uint8)
_PALM_FLOOR_EDGE = np.array([8, 74, 92], dtype=np.uint8)
_PREVIEW_EDGE = (216, 203, 255)
_PREVIEW_OCCUPANCY = np.array([167, 139, 250], dtype=np.uint8)
_PREVIEW_OCCUPANCY_ALPHA = 0.25
_CARRIED = np.array([245, 158, 11], dtype=np.uint8)
_CARRIED_OUTLINE = np.array([253, 230, 138], dtype=np.uint8)
GRASP_SWEEP_RGB = (250, 204, 21)
_GRASP_SWEEP_OUTLINE = (15, 23, 42)
# 闭合夹爪的本体标记：青色表示当前真实机器人，深色描边保证它在
# 浅色机械臂和复杂背景上都清晰。该标记只表达实际 TCP 与 BASE-Z，
# 不推断是否抓住物体，也不表达落点或任务进度。
GRIPPER_Z_AXIS_RGB = (14, 165, 233)
_GRIPPER_Z_AXIS_OUTLINE = (15, 23, 42)
_MOVE = (22, 163, 74, 255)
_AXIS_COLORS = (
    (220, 38, 38, 255),
    (22, 163, 74, 255),
    (37, 99, 235, 255),
)
_SURFACE_HALF_EXTENT_M = (0.24, 0.24, 0.22)
# 20% larger than the original 244×168 plate so the projected BASE signs
# stay readable after Contact panels are packed into the canvas.
_TRANSLATION_CORNER_WIDTH = 293
_TRANSLATION_CORNER_HEIGHT = 202
# Below this the panel is level for reading purposes and the badge is noise.
_MIN_OBLIQUE_LABEL_DEG = 5.0

PreviewGripperStyle = Literal["fk-mesh", "semantic-wireframe"]


@dataclass(frozen=True)
class _View:
    label: str
    view_name: Literal["front", "side"]
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    center: np.ndarray
    half_width_m: float
    half_height_m: float
    horizontal_axis: Literal["x", "y", "z"]
    vertical_axis: Literal["x", "y", "z"]
    view_axis: Literal["x", "y", "z"]


@dataclass(frozen=True)
class NearFieldPreview:
    """Geometry-only view of the active, unexecuted Waypoint."""

    target_pose: Pose | None
    joint_positions_rad: tuple[float, ...] | None
    gripper_opening: float | None
    gripper_style: PreviewGripperStyle = "fk-mesh"
    realized_tcp_pose: Pose | None = None
    visual_edit: VisualEdit | None = None
    previous_target_pose: Pose | None = None
    previous_gripper_opening: float | None = None
    # Gravity-stable camera orientation locked when the Imagination session
    # starts.  It is presenter-private and deliberately independent of later
    # target rotations, so rotate edits move the virtual gripper rather than
    # counter-rotating the observed world.
    contact_frame_quaternion_xyzw: tuple[float, float, float, float] | None = None
    contact_frame_position_xyz: tuple[float, float, float] | None = None


def render_contact_focus(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState | None,
    preview: NearFieldPreview | None,
    *,
    source_mask: np.ndarray | None = None,
    source_points_base: np.ndarray | None = None,
    contact_cameras: ContactCameraPair | None = None,
    carried_volume_triangles_base: np.ndarray | None = None,
    grasp_sweep_segment_base: np.ndarray | None = None,
    gripper_z_axis_base: np.ndarray | None = None,
    plumb_lines: list[PlumbLine] | None = None,
    output_width: int = CONTACT_FOCUS_WIDTH,
    panel_height: int = CONTACT_PANEL_HEIGHT,
) -> np.ndarray | None:
    """Render two gravity-stable, session-locked views around one target.

    The crop centre follows target translation, but its orientation is locked
    at the start of the Imagination session with WORLD +Z pointing upward.
    Later rotate edits therefore rotate only the virtual gripper; they never
    counter-rotate the current RGB-D surfaces or tilt the support plane.
    """

    if preview is None:
        return None
    if contact_cameras is not None:
        return _render_direct_contact_pair(
            contact_cameras,
            robot,
            preview,
            source_points_base=source_points_base,
            carried_volume_triangles_base=carried_volume_triangles_base,
            grasp_sweep_segment_base=grasp_sweep_segment_base,
            gripper_z_axis_base=gripper_z_axis_base,
            plumb_lines=plumb_lines,
            output_width=output_width,
            panel_height=panel_height,
        )
    pair = _render_geometry_pair(
        agentview,
        wrist,
        robot,
        preview,
        agentview_emphasis_mask=source_mask,
        carried_volume_triangles_base=carried_volume_triangles_base,
        grasp_sweep_segment_base=grasp_sweep_segment_base,
        gripper_z_axis_base=gripper_z_axis_base,
    )
    if pair is None:
        return None
    return _resize_contact_pair(
        pair,
        output_width=output_width,
        panel_height=panel_height,
    )


def render_contact_auxiliary(
    camera: dict,
    robot: RobotState | None,
    preview: NearFieldPreview | None,
    *,
    source_points_base: np.ndarray | None = None,
    carried_volume_triangles_base: np.ndarray | None = None,
    grasp_sweep_segment_base: np.ndarray | None = None,
    gripper_z_axis_base: np.ndarray | None = None,
    plumb_lines: list[PlumbLine] | None = None,
    output_width: int = CONTACT_FOCUS_WIDTH,
    panel_height: int = CONTACT_PANEL_HEIGHT,
) -> np.ndarray | None:
    """Render the steep third view with a calibrated BASE-XY compass.

    FRONT/SIDE remain the calibrated action-reading pair.  This panel is only
    complementary visual evidence: its diagonal azimuth and steeper elevation
    make the fingers, payload and receptacle relation visible when the Panda
    hand occludes a pure side camera.
    """

    if preview is None or robot is None:
        return None
    return _render_direct_contact_view(
        camera,
        "auxiliary",
        robot,
        preview,
        source_points_base=source_points_base,
        carried_volume_triangles_base=carried_volume_triangles_base,
        grasp_sweep_segment_base=grasp_sweep_segment_base,
        gripper_z_axis_base=gripper_z_axis_base,
        plumb_lines=plumb_lines,
        output_width=output_width,
        panel_height=panel_height,
        show_translation_axes=False,
        show_base_xy_axes=True,
    )


def _render_geometry_pair(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState,
    preview: NearFieldPreview | None,
    *,
    agentview_emphasis_mask: np.ndarray | None = None,
    carried_volume_triangles_base: np.ndarray | None = None,
    grasp_sweep_segment_base: np.ndarray | None = None,
    gripper_z_axis_base: np.ndarray | None = None,
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
    tcp_rotation = Rotation.from_quat(tcp_quaternion / np.linalg.norm(tcp_quaternion)).as_matrix()
    frame_position, frame_rotation = _contact_render_frame(
        tcp_position,
        tcp_rotation,
        preview,
    )

    # The crop follows target translation but never target rotation.  A locked
    # gravity-stable frame keeps the observed world visually stationary across
    # successive rotate edits and makes the edit's causal direction explicit.
    focus_center_local = np.array([0.0, 0.0, 0.015], dtype=np.float64)
    surfaces: list[RgbdSurface] = []
    for camera_index, camera in enumerate((agentview, wrist)):
        if camera is None:
            continue
        surface = reconstruct_rgbd_surface(
            camera,
            frame_position_base=frame_position,
            frame_rotation_base=frame_rotation,
            half_extent_m=_SURFACE_HALF_EXTENT_M,
            emphasis_mask=(agentview_emphasis_mask if camera_index == 0 else None),
        )
        if surface is not None:
            surfaces.append(surface)

    triangles_local = _gripper_triangles_local(
        robot.joint_positions_rad,
        robot.gripper_opening,
        frame_position,
        frame_rotation,
    )
    preview_triangles_local = None
    preview_hand_triangles_local = None
    if preview is not None and preview.gripper_style == "semantic-wireframe":
        if (
            preview.joint_positions_rad is not None
            and preview.realized_tcp_pose is not None
            and preview.gripper_opening is not None
        ):
            semantic_base = semantic_parallel_jaw_triangles(
                preview.realized_tcp_pose.position_xyz,
                preview.realized_tcp_pose.quaternion_xyzw,
                preview.gripper_opening,
            )
            preview_triangles_local = np.ascontiguousarray(
                (semantic_base - frame_position) @ frame_rotation
            )
    elif (
        preview is not None
        and preview.joint_positions_rad is not None
        and preview.gripper_opening is not None
    ):
        preview_triangles_local = _gripper_triangles_local(
            preview.joint_positions_rad,
            preview.gripper_opening,
            frame_position,
            frame_rotation,
        )
        preview_hand_triangles_local = _gripper_part_triangles_local(
            preview.joint_positions_rad,
            preview.gripper_opening,
            frame_position,
            frame_rotation,
            method_name="hand_triangles",
        )
    elif (
        preview is not None
        and preview.target_pose is not None
        and preview.gripper_opening is not None
    ):
        target_position = np.asarray(
            preview.target_pose.position_xyz,
            dtype=np.float64,
        )
        target_rotation = Rotation.from_quat(
            np.asarray(preview.target_pose.quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
        target_shape_current_local = _gripper_triangles_local(
            robot.joint_positions_rad,
            preview.gripper_opening,
            tcp_position,
            tcp_rotation,
        )
        if target_shape_current_local is not None:
            target_shape_base = target_shape_current_local @ target_rotation.T + target_position
            preview_triangles_local = (target_shape_base - frame_position) @ frame_rotation
            preview_triangles_local = np.ascontiguousarray(preview_triangles_local)
        target_hand_current_local = _gripper_part_triangles_local(
            robot.joint_positions_rad,
            preview.gripper_opening,
            tcp_position,
            tcp_rotation,
            method_name="hand_triangles",
        )
        if target_hand_current_local is not None:
            target_hand_base = target_hand_current_local @ target_rotation.T + target_position
            preview_hand_triangles_local = (
                target_hand_base - frame_position
            ) @ frame_rotation
            preview_hand_triangles_local = np.ascontiguousarray(
                preview_hand_triangles_local
            )
    views = _contact_views(
        "CURRENT + PREVIEW" if preview is not None else "OBSERVED NOW",
        focus_center_local,
    )
    carried_triangles_local = None
    if carried_volume_triangles_base is not None:
        carried_triangles_local = (
            np.asarray(carried_volume_triangles_base, dtype=np.float64) - frame_position
        ) @ frame_rotation
    grasp_sweep_segment_local = None
    if grasp_sweep_segment_base is not None:
        segment = np.asarray(grasp_sweep_segment_base, dtype=np.float64).reshape(2, 3)
        if np.isfinite(segment).all():
            grasp_sweep_segment_local = np.ascontiguousarray(
                (segment - frame_position) @ frame_rotation
            )
    gripper_z_axis_local = None
    if gripper_z_axis_base is not None:
        axis = np.asarray(gripper_z_axis_base, dtype=np.float64).reshape(3, 3)
        if np.isfinite(axis).all():
            gripper_z_axis_local = np.ascontiguousarray(
                (axis - frame_position) @ frame_rotation
            )
    panels = [
        _render_view(
            tuple(surfaces),
            triangles_local,
            carried_triangles_local,
            preview_triangles_local,
            preview_hand_triangles_local,
            view,
            current_tcp_position=frame_position,
            current_tcp_rotation=frame_rotation,
            preview=preview,
            grasp_sweep_segment_local=grasp_sweep_segment_local,
            gripper_z_axis_local=gripper_z_axis_local,
        )
        for view in views
    ]
    canvas = np.full((NEAR_FIELD_HEIGHT, NEAR_FIELD_WIDTH, 3), _BACKGROUND, dtype=np.uint8)
    canvas[:CONTACT_PANEL_HEIGHT] = panels[0]
    second_top = CONTACT_PANEL_HEIGHT + CONTACT_PANEL_GAP
    canvas[second_top : second_top + CONTACT_PANEL_HEIGHT] = panels[1]
    canvas[CONTACT_PANEL_HEIGHT:second_top] = np.array([207, 217, 231], dtype=np.uint8)
    return canvas


def _render_direct_contact_pair(
    contact_cameras: ContactCameraPair,
    robot: RobotState | None,
    preview: NearFieldPreview,
    *,
    source_points_base: np.ndarray | None = None,
    carried_volume_triangles_base: np.ndarray | None = None,
    grasp_sweep_segment_base: np.ndarray | None = None,
    gripper_z_axis_base: np.ndarray | None = None,
    plumb_lines: list[PlumbLine] | None = None,
    output_width: int = CONTACT_FOCUS_WIDTH,
    panel_height: int = CONTACT_PANEL_HEIGHT,
) -> np.ndarray | None:
    """Compose MuJoCo-rendered Contact Cameras with deterministic overlays."""

    if robot is None or robot.joint_positions_rad is None or robot.gripper_opening is None:
        return None
    panels = [
        _render_direct_contact_view(
            camera,
            view_name,
            robot,
            preview,
            source_points_base=source_points_base,
            carried_volume_triangles_base=carried_volume_triangles_base,
            grasp_sweep_segment_base=grasp_sweep_segment_base,
            gripper_z_axis_base=gripper_z_axis_base,
            plumb_lines=plumb_lines,
            output_width=output_width,
            panel_height=panel_height,
        )
        for view_name, camera in (
            ("front", contact_cameras.front),
            ("side", contact_cameras.side),
        )
    ]
    canvas_height = 2 * panel_height + CONTACT_PANEL_GAP
    canvas = np.full((canvas_height, output_width, 3), _BACKGROUND, dtype=np.uint8)
    canvas[:panel_height] = panels[0]
    second_top = panel_height + CONTACT_PANEL_GAP
    canvas[second_top : second_top + panel_height] = panels[1]
    canvas[panel_height:second_top] = np.array([207, 217, 231], dtype=np.uint8)
    return np.ascontiguousarray(canvas)


def _resize_contact_pair(
    pair: np.ndarray,
    *,
    output_width: int,
    panel_height: int,
) -> np.ndarray:
    """Resize fallback panels independently instead of stretching their gap."""

    source = np.asarray(pair, dtype=np.uint8)
    source_front = source[:CONTACT_PANEL_HEIGHT]
    source_side_top = CONTACT_PANEL_HEIGHT + CONTACT_PANEL_GAP
    source_side = source[source_side_top : source_side_top + CONTACT_PANEL_HEIGHT]
    panels = [
        np.asarray(
            Image.fromarray(panel).resize(
                (output_width, panel_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        for panel in (source_front, source_side)
    ]
    canvas = np.full(
        (2 * panel_height + CONTACT_PANEL_GAP, output_width, 3),
        _BACKGROUND,
        dtype=np.uint8,
    )
    canvas[:panel_height] = panels[0]
    side_top = panel_height + CONTACT_PANEL_GAP
    canvas[side_top : side_top + panel_height] = panels[1]
    canvas[panel_height:side_top] = np.array([207, 217, 231], dtype=np.uint8)
    return np.ascontiguousarray(canvas)


def _render_direct_contact_view(
    camera: dict,
    view_name: Literal["front", "side", "auxiliary"],
    robot: RobotState,
    preview: NearFieldPreview,
    *,
    source_points_base: np.ndarray | None = None,
    carried_volume_triangles_base: np.ndarray | None = None,
    grasp_sweep_segment_base: np.ndarray | None = None,
    gripper_z_axis_base: np.ndarray | None = None,
    plumb_lines: list[PlumbLine] | None = None,
    output_width: int = CONTACT_FOCUS_WIDTH,
    panel_height: int = CONTACT_PANEL_HEIGHT,
    show_translation_axes: bool = True,
    show_base_xy_axes: bool = False,
) -> np.ndarray:
    image = np.asarray(camera["images"]["rgb"], dtype=np.uint8).copy()
    height, width = image.shape[:2]
    if image.shape != (panel_height, output_width, 3):
        image = np.asarray(
            Image.fromarray(image).resize(
                (output_width, panel_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        ).copy()
        height, width = image.shape[:2]
    render_camera = _scaled_camera(camera, width, height)

    # Both cues are presenter overlays on the current MuJoCo RGB.  The amber
    # surface comes only from current-revision sensor geometry; when that
    # evidence expires no object mask is invented.  The cyan mask is the
    # element segmentation rendered by the exact same MuJoCo camera, not a
    # second robot model projected onto the image.
    if source_points_base is not None:
        _overlay_camera_source_mask(image, source_points_base, render_camera)
    current_finger_mask = camera.get("finger_mask")
    if current_finger_mask is not None:
        _blend_mask(
            image,
            _fit_mask(current_finger_mask, width, height),
            _CURRENT_FINGER_FILL,
            _CURRENT_FINGER_EDGE,
            alpha=_CURRENT_FINGER_ALPHA,
        )
    # Marked after the fingers and opaquely: on a top-down approach the palm
    # underside runs out of clearance first, and a translucent band would be
    # lost against the pale gripper body it sits on.
    palm_floor_mask = camera.get("palm_floor_mask")
    if palm_floor_mask is not None:
        _blend_mask(
            image,
            _fit_mask(palm_floor_mask, width, height),
            _PALM_FLOOR_FILL,
            _PALM_FLOOR_EDGE,
            alpha=0.85,
        )

    preview_triangles = _preview_gripper_triangles(
        robot,
        preview,
    )
    preview_hand_triangles = _preview_hand_triangles(robot, preview)
    if preview_hand_triangles is not None:
        _overlay_camera_mesh_occupancy(
            image,
            preview_hand_triangles,
            render_camera,
        )
    if preview_triangles is not None:
        _overlay_camera_mesh_outline(image, preview_triangles, render_camera)

    carried_mask = None
    if carried_volume_triangles_base is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            carried_mask = _projected_silhouette(
                carried_volume_triangles_base,
                render_camera,
                width,
                height,
            )
        _blend_mask(
            image,
            carried_mask,
            _CARRIED,
            _CARRIED_OUTLINE,
            alpha=0.42,
        )

    # Draw metric overlays in the same calibrated camera as the RGB scene.
    if plumb_lines:
        draw_plumb_overlays(image, render_camera, plumb_lines)
    if grasp_sweep_segment_base is not None:
        draw_grasp_sweep_overlay(image, render_camera, grasp_sweep_segment_base)
    if gripper_z_axis_base is not None:
        draw_gripper_z_axis_overlay(image, render_camera, gripper_z_axis_base)

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _label_font()
    scene_width = width
    display_camera = _scaled_camera(camera, width, height)
    if show_translation_axes:
        _draw_direct_translation_axes(draw, display_camera, scene_width)
    elif show_base_xy_axes:
        _draw_direct_base_xy_axes(draw, display_camera, scene_width)

    # A tilted panel must say so. Read as level, an oblique image makes the
    # payload look laterally centred whenever it is merely nearer the camera.
    tilt_deg = _camera_tilt_deg(camera)
    label = (
        f"CONTACT {view_name.upper()} · "
        + (f"OBLIQUE {tilt_deg:.0f}° DOWN · " if tilt_deg >= _MIN_OBLIQUE_LABEL_DEG else "")
        + ("CURRENT + PREVIEW" if preview.target_pose is not None else "CURRENT")
    )
    bounds = draw.textbbox((0, 0), label, font=font)
    label_left = 8
    label_right = min(
        scene_width - _TRANSLATION_CORNER_WIDTH - 18,
        label_left + bounds[2] - bounds[0] + 28,
    )
    draw.rounded_rectangle(
        (label_left, 8, label_right, 32),
        radius=4,
        fill=(255, 255, 255, 226),
        outline=(190, 204, 221, 255),
        width=1,
    )
    draw.text((label_left + 7, 12), label, fill=(30, 41, 59, 255), font=font)
    if preview.target_pose is not None:
        _draw_direct_adjustment(
            draw,
            scene_width,
            height,
            preview,
        )
        _draw_target_depth_scale(
            draw,
            display_camera,
            scene_width,
            height,
            preview.target_pose,
        )
    return np.asarray(pil, dtype=np.uint8)


def _overlay_camera_source_mask(
    image: np.ndarray,
    points_base: np.ndarray,
    camera: dict,
) -> None:
    """Highlight current sensor-derived object surfaces without filling holes."""

    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return
    if len(points) > 1024:
        indices = np.linspace(0, len(points) - 1, 1024, dtype=np.int64)
        points = points[indices]
    projected = project_world_to_pixel(
        points,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    )
    height, width = image.shape[:2]
    valid = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 2] > 0.01)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    pixels = projected[valid, :2]
    if len(pixels) < 3:
        return
    lower = np.quantile(pixels, 0.02, axis=0)
    upper = np.quantile(pixels, 0.98, axis=0)
    pixels = pixels[
        (pixels[:, 0] >= lower[0])
        & (pixels[:, 0] <= upper[0])
        & (pixels[:, 1] >= lower[1])
        & (pixels[:, 1] <= upper[1])
    ]
    if len(pixels) < 3 or np.any(np.ptp(pixels, axis=0) < 2.0):
        return
    pixel_indices = np.rint(pixels).astype(np.int64)
    pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, width - 1)
    pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=bool)
    mask[pixel_indices[:, 1], pixel_indices[:, 0]] = True
    radius = max(2, int(round(min(width, height) / 220.0)))
    mask = binary_dilation(mask, iterations=radius)
    mask = binary_closing(mask, iterations=max(1, radius // 2))
    _blend_mask(
        image,
        mask,
        _SOURCE_FILL,
        _SOURCE_EDGE,
        alpha=0.24,
    )


def _base_axis_screen_directions(camera: dict) -> np.ndarray:
    """Return calibrated image-plane directions for BASE +X/+Y/+Z.

    ``pose_mat`` is BASE-from-image-camera, whose image-camera +X points
    right and +Y points down.  The first two camera components of each BASE
    unit vector therefore give the local screen direction without depending
    on a potentially occluded 3-D anchor in the scene.
    """

    base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64)
    camera_from_base_rotation = base_from_camera[:3, :3].T
    return np.asarray(
        [camera_from_base_rotation[:2, index] for index in range(3)],
        dtype=np.float64,
    )


def _base_axis_camera_directions(camera: dict) -> np.ndarray:
    """Return BASE +X/+Y/+Z expressed in the image-camera frame."""

    base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64)
    return np.asarray(base_from_camera[:3, :3].T, dtype=np.float64).T


def _draw_direct_translation_axes(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    scene_width: int,
) -> None:
    """Draw only the two BASE axes observable in this Contact image plane.

    The view-normal axis cannot be judged directly from one 2-D RGB panel.
    Showing it as an ``IN/OUT`` control mixed image evidence with an inferred
    depth command and encouraged models to edit the hidden dimension.  The
    orthogonal Contact panel exposes that remaining BASE axis in its own image
    plane, so each panel deliberately presents only its two visible controls.
    """

    left = scene_width - _TRANSLATION_CORNER_WIDTH - 8
    top = 8
    right = scene_width - 8
    bottom = top + _TRANSLATION_CORNER_HEIGHT
    _draw_gizmo_plate(
        draw,
        (left, top, right, bottom),
        "MOVE BASE",
        title_font=_translation_title_font(),
    )

    origin = np.array(
        [
            left + 0.5 * _TRANSLATION_CORNER_WIDTH,
            top + 0.57 * _TRANSLATION_CORNER_HEIGHT,
        ],
        dtype=np.float64,
    )
    camera_directions = _base_axis_camera_directions(camera)
    depth_axis_index = int(np.argmax(np.abs(camera_directions[:, 2])))
    arm_px = 64.0
    for axis_index in range(3):
        if axis_index == depth_axis_index:
            continue
        axis = "XYZ"[axis_index]
        camera_direction = camera_directions[axis_index]
        direction = camera_direction[:2]
        color = _AXIS_COLORS[axis_index]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        unit_direction = direction / norm
        endpoint = origin + unit_direction * arm_px
        reverse_endpoint = origin - unit_direction * arm_px
        start = tuple(float(value) for value in origin)
        end = tuple(float(value) for value in endpoint)
        reverse_end = tuple(float(value) for value in reverse_endpoint)
        draw.line((*reverse_end, *end), fill=(2, 6, 23, 238), width=17)
        draw.line((*reverse_end, *end), fill=color, width=11)
        _draw_arrow_head(draw, start, end, color, size=22.0)
        _draw_arrow_head(draw, start, reverse_end, color, size=22.0)
        _draw_centered_dark_axis_label(
            draw,
            tuple(float(value) for value in endpoint + unit_direction * 20.0),
            f"+{axis}",
            color,
            bounds=(left + 8, top + 42, right - 8, bottom - 8),
            font=_translation_axis_font(),
        )
        _draw_centered_dark_axis_label(
            draw,
            tuple(float(value) for value in reverse_endpoint - unit_direction * 20.0),
            f"−{axis}",
            color,
            bounds=(left + 8, top + 42, right - 8, bottom - 8),
            font=_translation_axis_font(),
        )

    draw.ellipse(
        (origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8),
        fill=(248, 250, 252, 255),
        outline=(15, 23, 42, 255),
        width=3,
    )


def _draw_direct_base_xy_axes(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    scene_width: int,
) -> None:
    """Draw calibrated ``±X/±Y`` directions in the 68° auxiliary view."""

    left = scene_width - _TRANSLATION_CORNER_WIDTH - 8
    top = 8
    right = scene_width - 8
    bottom = top + _TRANSLATION_CORNER_HEIGHT
    _draw_gizmo_plate(
        draw,
        (left, top, right, bottom),
        "BASE XY",
        title_font=_translation_title_font(),
    )
    origin = np.array(
        [
            left + 0.5 * _TRANSLATION_CORNER_WIDTH,
            top + 0.57 * _TRANSLATION_CORNER_HEIGHT,
        ],
        dtype=np.float64,
    )
    camera_directions = _base_axis_camera_directions(camera)
    arm_px = 64.0
    for axis_index in (0, 1):
        axis = "XY"[axis_index]
        direction = camera_directions[axis_index, :2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        unit_direction = direction / norm
        endpoint = origin + unit_direction * arm_px
        reverse_endpoint = origin - unit_direction * arm_px
        color = _AXIS_COLORS[axis_index]
        start = tuple(float(value) for value in origin)
        end = tuple(float(value) for value in endpoint)
        reverse_end = tuple(float(value) for value in reverse_endpoint)
        draw.line((*reverse_end, *end), fill=(2, 6, 23, 238), width=17)
        draw.line((*reverse_end, *end), fill=color, width=11)
        _draw_arrow_head(draw, start, end, color, size=22.0)
        _draw_arrow_head(draw, start, reverse_end, color, size=22.0)
        _draw_centered_dark_axis_label(
            draw,
            tuple(float(value) for value in endpoint + unit_direction * 20.0),
            f"+{axis}",
            color,
            bounds=(left + 8, top + 42, right - 8, bottom - 8),
            font=_translation_axis_font(),
        )
        _draw_centered_dark_axis_label(
            draw,
            tuple(float(value) for value in reverse_endpoint - unit_direction * 20.0),
            f"−{axis}",
            color,
            bounds=(left + 8, top + 42, right - 8, bottom - 8),
            font=_translation_axis_font(),
        )
    draw.ellipse(
        (origin[0] - 8, origin[1] - 8, origin[0] + 8, origin[1] + 8),
        fill=(248, 250, 252, 255),
        outline=(15, 23, 42, 255),
        width=3,
    )


def _draw_gizmo_plate(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    *,
    title_font: ImageFont.ImageFont | None = None,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=6,
        fill=(3, 7, 18, 205),
        outline=(148, 163, 184, 235),
        width=2,
    )
    draw.text(
        (box[0] + 8, box[1] + 6),
        title,
        fill=(248, 250, 252, 255),
        font=title_font or _label_font(),
    )


def _draw_centered_dark_axis_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    color: tuple[int, int, int, int],
    *,
    bounds: tuple[float, float, float, float],
    font: ImageFont.ImageFont | None = None,
) -> None:
    """Draw a centered axis label clamped to a reserved non-overlap area."""

    font = font or _label_font()
    text_bounds = draw.textbbox((0, 0), label, font=font)
    width = text_bounds[2] - text_bounds[0] + 10
    height = text_bounds[3] - text_bounds[1] + 6
    left = min(max(center[0] - width * 0.5, bounds[0]), bounds[2] - width)
    top = min(max(center[1] - height * 0.5, bounds[1]), bounds[3] - height)
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=3,
        fill=(3, 7, 18, 232),
        outline=color,
        width=2,
    )
    draw.text(
        (left + 5, top + 3 - text_bounds[1]),
        label,
        fill=color,
        font=font,
    )


def _scaled_camera(camera: dict, width: int, height: int) -> dict:
    """Scale private calibration with a raster resize."""

    source = np.asarray(camera["images"]["rgb"])
    source_height, source_width = source.shape[:2]
    scale_x = float(width) / float(source_width)
    scale_y = float(height) / float(source_height)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).copy()
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    return {
        "images": {"rgb": np.empty((height, width, 3), dtype=np.uint8)},
        "intrinsics": intrinsics,
        "pose_mat": camera["pose_mat"],
    }


def _camera_gripper_triangles(
    joints: tuple[float, ...],
    opening: float,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        return fk.triangles(np.asarray(joints, dtype=np.float64), float(opening))
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _preview_gripper_triangles(
    robot: RobotState,
    preview: NearFieldPreview,
) -> np.ndarray | None:
    if preview.gripper_style == "semantic-wireframe":
        if (
            preview.joint_positions_rad is None
            or preview.realized_tcp_pose is None
            or preview.gripper_opening is None
        ):
            return None
        return semantic_parallel_jaw_triangles(
            preview.realized_tcp_pose.position_xyz,
            preview.realized_tcp_pose.quaternion_xyzw,
            preview.gripper_opening,
        )
    if preview.joint_positions_rad is not None and preview.gripper_opening is not None:
        return _camera_gripper_triangles(
            preview.joint_positions_rad,
            preview.gripper_opening,
        )
    return _direct_target_gripper_triangles(
        robot,
        preview.target_pose,
        preview.gripper_opening,
    )


def _preview_hand_triangles(
    robot: RobotState,
    preview: NearFieldPreview,
) -> np.ndarray | None:
    if preview.gripper_style == "semantic-wireframe":
        return None
    if preview.joint_positions_rad is not None and preview.gripper_opening is not None:
        return _camera_hand_triangles(
            preview.joint_positions_rad,
            preview.gripper_opening,
        )
    return _direct_target_hand_triangles(
        robot,
        preview.target_pose,
        preview.gripper_opening,
    )


def _camera_hand_triangles(
    joints: tuple[float, ...],
    opening: float,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    hand_triangles = getattr(fk, "hand_triangles", None) if fk is not None else None
    if not callable(hand_triangles):
        return None
    try:
        return hand_triangles(np.asarray(joints, dtype=np.float64), float(opening))
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _direct_target_gripper_triangles(
    robot: RobotState,
    pose: Pose | None,
    opening: float | None,
) -> np.ndarray | None:
    return _direct_target_part_triangles(robot, pose, opening, "triangles")


def _direct_target_hand_triangles(
    robot: RobotState,
    pose: Pose | None,
    opening: float | None,
) -> np.ndarray | None:
    return _direct_target_part_triangles(robot, pose, opening, "hand_triangles")


def _direct_target_part_triangles(
    robot: RobotState,
    pose: Pose | None,
    opening: float | None,
    method_name: str,
) -> np.ndarray | None:
    if (
        pose is None
        or opening is None
        or robot.tcp_pose is None
        or robot.joint_positions_rad is None
    ):
        return None
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    triangles_for_part = getattr(fk, method_name, None)
    if not callable(triangles_for_part):
        return None
    try:
        current_tcp_position = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64)
        current_tcp_rotation = Rotation.from_quat(
            np.asarray(robot.tcp_pose.quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
        local = (
            triangles_for_part(np.asarray(robot.joint_positions_rad), float(opening))
            - current_tcp_position
        ) @ current_tcp_rotation
        target_position = np.asarray(pose.position_xyz, dtype=np.float64)
        target_rotation = Rotation.from_quat(
            np.asarray(pose.quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
        return local @ target_rotation.T + target_position
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _direct_target_gripper_mask(
    robot: RobotState,
    pose: Pose | None,
    opening: float | None,
    camera: dict,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Compatibility helper for target cropping; rendering uses 3-D lines."""

    triangles = _direct_target_gripper_triangles(robot, pose, opening)
    return (
        None
        if triangles is None
        else _projected_silhouette(triangles, camera, width, height)
    )


def _projected_silhouette(
    triangles: np.ndarray,
    camera: dict,
    width: int,
    height: int,
) -> np.ndarray:
    projected = project_world_to_pixel(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 3),
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    ).reshape(-1, 3, 3)
    return rasterize_silhouette(
        projected[..., :2],
        projected[..., 2],
        width,
        height,
    )


def _overlay_camera_mesh_outline(
    image: np.ndarray,
    triangles: np.ndarray,
    camera: dict,
) -> None:
    """Overlay transparent lavender 3-D line art in one camera raster."""

    projected = project_world_to_pixel(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 3),
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    ).reshape(-1, 3, 3)
    overlay_projected_mesh_outline(
        image,
        triangles,
        projected[..., :2],
        projected[..., 2],
        color=_PREVIEW_EDGE,
        camera_position_base=np.asarray(camera["pose_mat"], dtype=np.float64)[:3, 3],
    )


def _overlay_camera_mesh_occupancy(
    image: np.ndarray,
    triangles: np.ndarray,
    camera: dict,
) -> None:
    """Faintly fill only the rigid palm body; fingers remain line art."""

    mask = _projected_silhouette(
        triangles,
        camera,
        image.shape[1],
        image.shape[0],
    )
    if not np.any(mask):
        return
    image[mask] = np.asarray(
        np.round(
            image[mask].astype(np.float64) * (1.0 - _PREVIEW_OCCUPANCY_ALPHA)
            + _PREVIEW_OCCUPANCY.astype(np.float64) * _PREVIEW_OCCUPANCY_ALPHA
        ),
        dtype=np.uint8,
    )


def _overlay_camera_mesh_mask(
    image: np.ndarray,
    triangles: np.ndarray,
    camera: dict,
    *,
    color: np.ndarray,
    outline: np.ndarray,
    alpha: float,
) -> None:
    """Blend one projected FK part while preserving the underlying RGB."""

    mask = _projected_silhouette(
        triangles,
        camera,
        image.shape[1],
        image.shape[0],
    )
    _blend_mask(image, mask, color, outline, alpha=alpha)


def _fit_mask(mask: Any, width: int, height: int) -> np.ndarray:
    """Resample a segmentation mask onto the panel it is composited into."""

    resolved = np.asarray(mask, dtype=bool)
    if resolved.shape == (height, width):
        return resolved
    resized = Image.fromarray(resolved.astype(np.uint8) * 255).resize(
        (width, height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8) > 0


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    outline: np.ndarray,
    *,
    alpha: float,
) -> None:
    if not np.any(mask):
        return
    image[mask] = np.asarray(
        np.round(image[mask].astype(np.float64) * (1.0 - alpha) + color * alpha),
        dtype=np.uint8,
    )
    image[mask_outline(mask)] = outline


def _draw_direct_adjustment(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    preview: NearFieldPreview,
) -> None:
    edit = preview.visual_edit
    target = preview.target_pose
    if edit is None or target is None:
        return
    color = _MOVE
    if edit.kind == "rotate" and edit.axis is not None and edit.axis in "xyz":
        color = _AXIS_COLORS["xyz".index(edit.axis)]
        label = (
            f"ROTATE {str(edit.frame).upper()} {edit.axis.upper()} {float(edit.angle_deg):+.1f} deg"
        )
        _draw_adjustment_label(draw, label, width, height, color)
        return
    if edit.kind != "delta_move" or edit.delta_xyz_m is None:
        return
    delta_cm = np.asarray(edit.delta_xyz_m, dtype=np.float64) * 100.0
    label = (
        f"MOVE {str(edit.frame).upper()} "
        f"dX {delta_cm[0]:+.1f} dY {delta_cm[1]:+.1f} dZ {delta_cm[2]:+.1f} cm"
    )
    _draw_adjustment_label(draw, label, width, height, color)


def _draw_target_depth_scale(
    draw: ImageDraw.ImageDraw,
    camera: dict,
    width: int,
    height: int,
    target_pose: Pose,
) -> None:
    """Draw one uncluttered, perspective-correct 5 cm reference bar."""

    projected = project_world_to_pixel(
        np.asarray(target_pose.position_xyz, dtype=np.float64).reshape(1, 3),
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    )[0]
    depth = float(projected[2])
    focal_px = float(np.asarray(camera["intrinsics"], dtype=np.float64)[0, 0])
    if not np.isfinite((depth, focal_px)).all() or depth <= 0.01 or focal_px <= 0.0:
        return
    five_cm_px = focal_px * 0.05 / depth
    if not 20.0 <= five_cm_px <= 280.0:
        return
    right = float(width - 10)
    left = right - five_cm_px
    bottom = float(height - 27)
    top = bottom - 24.0
    draw.text(
        (left, top - 1),
        "5 cm",
        fill=(30, 64, 175, 255),
        font=_label_font(),
        stroke_width=2,
        stroke_fill=(255, 255, 255, 245),
    )
    x0 = left
    y = bottom
    draw.line((x0, y, right, y), fill=(255, 255, 255, 245), width=7)
    draw.line((x0, y, right, y), fill=(30, 64, 175, 255), width=3)
    for x in (x0, right):
        draw.line((x, y - 7, x, y + 5), fill=(255, 255, 255, 245), width=6)
        draw.line((x, y - 6, x, y + 4), fill=(30, 64, 175, 255), width=2)


def _contact_render_frame(
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
    preview: NearFieldPreview | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a translating focus centre and a non-rotating camera frame."""

    frame_position = np.asarray(tcp_position, dtype=np.float64)
    if preview is not None and preview.contact_frame_position_xyz is not None:
        frame_position = np.asarray(
            preview.contact_frame_position_xyz,
            dtype=np.float64,
        )
    elif preview is not None and preview.target_pose is not None:
        frame_position = np.asarray(preview.target_pose.position_xyz, dtype=np.float64)
    frame_rotation = _gravity_stable_contact_rotation(tcp_rotation)
    if preview is not None and preview.contact_frame_quaternion_xyzw is not None:
        frame_rotation = Rotation.from_quat(
            np.asarray(preview.contact_frame_quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
    return frame_position, frame_rotation


def _gripper_triangles_local(
    joints_rad: tuple[float, ...],
    opening: float,
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
) -> np.ndarray | None:
    return _gripper_part_triangles_local(
        joints_rad,
        opening,
        tcp_position,
        tcp_rotation,
        method_name="triangles",
    )


def _gripper_part_triangles_local(
    joints_rad: tuple[float, ...],
    opening: float,
    tcp_position: np.ndarray,
    tcp_rotation: np.ndarray,
    *,
    method_name: str,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    triangles_for_part = getattr(fk, method_name, None)
    if not callable(triangles_for_part):
        return None
    try:
        triangles = triangles_for_part(
            np.asarray(joints_rad, dtype=np.float64),
            float(opening),
        )
    except (RuntimeError, ValueError):
        return None
    return (triangles - tcp_position) @ tcp_rotation


def _camera_tilt_deg(camera: dict[str, Any]) -> float:
    """Degrees the optical axis points below the robot-base horizon."""

    try:
        pose_mat = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return 0.0
    forward = pose_mat[:3, 2]
    norm = float(np.linalg.norm(forward))
    if not np.isfinite(norm) or norm < 1e-9:
        return 0.0
    return float(np.degrees(np.arcsin(np.clip(-forward[2] / norm, -1.0, 1.0))))


def _contact_views(
    label_suffix: str,
    center: np.ndarray | None = None,
) -> tuple[_View, _View]:
    view_center = (
        np.array([0.0, 0.0, -0.015], dtype=np.float64)
        if center is None
        else np.asarray(center, dtype=np.float64).reshape(3)
    )
    # These are axes of the session-locked horizontal frame, not the current
    # target-tool frame.  Both panels keep WORLD +Z visually upward.  FRONT and
    # SIDE remain orthogonal so one exposes depth hidden by the other.
    return (
        _View(
            label=f"CONTACT FRONT · WORLD-Z UP · {label_suffix}",
            view_name="front",
            forward=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            right=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            center=view_center,
            half_width_m=0.13,
            half_height_m=0.12,
            horizontal_axis="y",
            vertical_axis="z",
            view_axis="x",
        ),
        _View(
            label=f"CONTACT SIDE · WORLD-Z UP · {label_suffix}",
            view_name="side",
            forward=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            right=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            center=view_center,
            half_width_m=0.13,
            half_height_m=0.12,
            horizontal_axis="x",
            vertical_axis="z",
            view_axis="y",
        ),
    )


def _render_view(
    surfaces: tuple[RgbdSurface, ...],
    current_triangles: np.ndarray | None,
    carried_triangles: np.ndarray | None,
    preview_triangles: np.ndarray | None,
    preview_hand_triangles: np.ndarray | None,
    view: _View,
    *,
    current_tcp_position: np.ndarray,
    current_tcp_rotation: np.ndarray,
    preview: NearFieldPreview | None,
    grasp_sweep_segment_local: np.ndarray | None = None,
    gripper_z_axis_local: np.ndarray | None = None,
) -> np.ndarray:
    height, width = CONTACT_PANEL_HEIGHT, NEAR_FIELD_WIDTH
    image = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    _draw_metric_grid(image, view)
    rasterize_rgbd_surfaces(
        image,
        surfaces,
        center_local=view.center,
        forward_local=view.forward,
        right_local=view.right,
        up_local=view.up,
        half_width_m=view.half_width_m,
        half_height_m=view.half_height_m,
    )

    if preview_triangles is not None and len(preview_triangles):
        if preview_hand_triangles is not None and len(preview_hand_triangles):
            _overlay_triangles(
                image,
                preview_hand_triangles,
                view,
                _PREVIEW_OCCUPANCY,
                np.asarray(_PREVIEW_EDGE, dtype=np.uint8),
                alpha=_PREVIEW_OCCUPANCY_ALPHA,
            )
        _overlay_preview_outline(
            image,
            preview_triangles,
            view,
        )
    if carried_triangles is not None and len(carried_triangles):
        _overlay_triangles(
            image,
            carried_triangles,
            view,
            _CARRIED,
            _CARRIED_OUTLINE,
            alpha=0.42,
        )

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = _label_font()
    if grasp_sweep_segment_local is not None:
        _draw_local_grasp_sweep(
            draw,
            grasp_sweep_segment_local,
            view,
            width,
            height,
        )
    if gripper_z_axis_local is not None:
        _draw_local_gripper_z_axis(
            draw,
            gripper_z_axis_local,
            view,
            width,
            height,
        )
    draw.rounded_rectangle(
        (8, 8, 330, 31),
        radius=4,
        fill=(255, 255, 255, 226),
        outline=(190, 204, 221, 255),
        width=1,
    )
    draw.text((15, 12), view.label, fill=(30, 41, 59, 255), font=font)
    _draw_contact_axis_key(draw, view, width)
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
    _draw_scale_bar(draw, view, width, height)
    return np.asarray(pil, dtype=np.uint8)


def draw_grasp_sweep_overlay(
    image: np.ndarray,
    camera: dict[str, Any],
    segment_base: np.ndarray,
) -> bool:
    """Draw the observed parallel-jaw closing channel into a calibrated RGB view."""

    segment = np.asarray(segment_base, dtype=np.float64).reshape(2, 3)
    if not np.isfinite(segment).all():
        return False
    projected = project_world_to_pixel(
        segment,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    )
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0.01):
        return False
    height, width = image.shape[:2]
    points = [tuple(np.rint(point[:2]).astype(int)) for point in projected]
    if all(
        x < -12 or x >= width + 12 or y < -12 or y >= height + 12
        for x, y in points
    ):
        return False
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(pil, "RGBA")
    _draw_grasp_sweep_line(draw, points)
    image[...] = np.asarray(pil, dtype=np.uint8)
    return True


def draw_gripper_z_axis_overlay(
    image: np.ndarray,
    camera: dict[str, Any],
    axis_base: np.ndarray,
) -> bool:
    """绘制闭合夹爪 TCP 小球及穿过它的 BASE-Z 轴。

    ``axis_base`` 按 ``[下端, TCP, 上端]`` 排列。几何完全来自当前机器人
    状态；函数不读取 attachment、目标点或 Preview，因此不会把认知假设
    伪装成本体感知。
    """

    axis = np.asarray(axis_base, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(axis).all():
        return False
    projected = project_world_to_pixel(
        axis,
        np.asarray(camera["intrinsics"], dtype=np.float64),
        np.asarray(camera["pose_mat"], dtype=np.float64),
    )
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0.01):
        return False
    pixels = [tuple(np.rint(point[:2]).astype(int)) for point in projected]
    height, width = image.shape[:2]
    if all(
        x < -12 or x >= width + 12 or y < -12 or y >= height + 12
        for x, y in pixels
    ):
        return False
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(pil, "RGBA")
    _draw_gripper_z_axis(draw, pixels)
    image[...] = np.asarray(pil, dtype=np.uint8)
    return True


def _draw_local_grasp_sweep(
    draw: ImageDraw.ImageDraw,
    segment_local: np.ndarray,
    view: _View,
    width: int,
    height: int,
) -> None:
    segment = np.asarray(segment_local, dtype=np.float64).reshape(2, 3)
    u, v, _depth = _project(segment, view, width, height)
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        return
    _draw_grasp_sweep_line(
        draw,
        [(int(round(float(u[0]))), int(round(float(v[0])))),
         (int(round(float(u[1]))), int(round(float(v[1]))))],
    )


def _draw_local_gripper_z_axis(
    draw: ImageDraw.ImageDraw,
    axis_local: np.ndarray,
    view: _View,
    width: int,
    height: int,
) -> None:
    axis = np.asarray(axis_local, dtype=np.float64).reshape(3, 3)
    u, v, _depth = _project(axis, view, width, height)
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        return
    _draw_gripper_z_axis(
        draw,
        [
            (int(round(float(x))), int(round(float(y))))
            for x, y in zip(u, v, strict=True)
        ],
    )


def _draw_grasp_sweep_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
) -> None:
    """Use an outlined, non-translucent cue that survives image rescaling."""

    draw.line(points, fill=(*_GRASP_SWEEP_OUTLINE, 230), width=8)
    draw.line(points, fill=(*GRASP_SWEEP_RGB, 255), width=4)
    for x, y in points:
        draw.ellipse(
            (x - 5, y - 5, x + 5, y + 5),
            fill=(*GRASP_SWEEP_RGB, 255),
            outline=(*_GRASP_SWEEP_OUTLINE, 255),
            width=2,
        )


def _draw_gripper_z_axis(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
) -> None:
    """画一根细的 BASE-Z 轴，并用有高光的小球标出真实 TCP。"""

    lower, anchor, upper = points
    draw.line(
        (lower, upper),
        fill=(*_GRIPPER_Z_AXIS_OUTLINE, 225),
        width=6,
    )
    draw.line(
        (lower, upper),
        fill=(*GRIPPER_Z_AXIS_RGB, 255),
        width=3,
    )
    x, y = anchor
    radius = 8
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(*GRIPPER_Z_AXIS_RGB, 255),
        outline=(*_GRIPPER_Z_AXIS_OUTLINE, 255),
        width=2,
    )
    draw.ellipse(
        (x - 3, y - 4, x + 1, y),
        fill=(255, 255, 255, 210),
    )


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
                image[mask].astype(np.float64) * (1.0 - alpha) + color.astype(np.float64) * alpha
            )
            image[mask] = np.asarray(np.round(blended), dtype=np.uint8)
            image[mask_outline(mask)] = outline_color


def _overlay_preview_outline(
    image: np.ndarray,
    triangles: np.ndarray,
    view: _View,
) -> None:
    """Draw the orthographic Preview without covering current RGB-D points."""

    values = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    flat = values.reshape(-1, 3)
    height, width = image.shape[:2]
    u, v, depth = _project(flat, view, width, height)
    overlay_projected_mesh_outline(
        image,
        values,
        np.column_stack((u, v)).reshape(-1, 3, 2),
        depth.reshape(-1, 3),
        color=_PREVIEW_EDGE,
        view_forward_base=view.forward,
    )


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
    one_cm_px = 0.01 * width * 0.5 / view.half_width_m
    length_px = int(round(3.0 * one_cm_px))
    right = width - 14
    left = right - length_px
    y = height - 17
    draw.line((left, y, right, y), fill=(30, 41, 59, 230), width=3)
    for centimetre in range(4):
        x = left + centimetre * one_cm_px
        draw.line((x, y - 4, x, y + 4), fill=(30, 41, 59, 230), width=2)
        draw.text(
            (x - 3, y - 18),
            str(centimetre),
            fill=(30, 41, 59, 255),
            font=_label_font(),
        )
    draw.text((left - 25, y - 18), "cm", fill=(30, 41, 59, 255), font=_label_font())


def _draw_contact_axis_key(
    draw: ImageDraw.ImageDraw,
    view: _View,
    width: int,
) -> None:
    """Draw the two measurable axes and state the view-normal separately."""

    font = _label_font()
    accent = (30, 64, 175, 255)
    plate_right = width - 10
    plate_left = plate_right - 176
    plate_top = 8
    plate_bottom = 104
    draw.rounded_rectangle(
        (plate_left, plate_top, plate_right, plate_bottom),
        radius=6,
        fill=(255, 255, 255, 220),
        outline=accent,
        width=2,
    )
    draw.text(
        (plate_left + 8, plate_top + 6),
        f"LOCKED {view.view_name.upper()}",
        fill=(30, 64, 175, 255),
        font=font,
    )
    origin = (plate_left + 38.0, plate_top + 69.0)
    draw.ellipse(
        (origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4),
        fill=(255, 255, 255, 255),
        outline=accent,
        width=2,
    )
    axis_specs = (
        (
            "SCREEN RIGHT",
            np.array([1.0, 0.0]),
            40.0,
            (71, 85, 105, 255),
            np.array([56.0, 0.0]),
        ),
        (
            "WORLD +Z",
            np.array([0.0, -1.0]),
            31.0,
            (*_AXIS_COLORS[2][:3], 255),
            np.array([53.0, 0.0]),
        ),
    )
    for axis_name, direction, length, axis_color, label_offset in axis_specs:
        endpoint_array = np.asarray(origin) + direction * length
        endpoint = (float(endpoint_array[0]), float(endpoint_array[1]))
        draw.line((*origin, *endpoint), fill=axis_color, width=5)
        _draw_arrow_head(draw, origin, endpoint, axis_color, size=9.0)
        label_anchor = endpoint_array + label_offset
        _draw_axis_label_box(
            draw,
            (float(label_anchor[0]), float(label_anchor[1])),
            axis_name,
            axis_color,
            font,
        )


def gravity_stable_contact_frame_quaternion(
    pose: Pose,
    *,
    reference_quaternion_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return a yaw-only contact frame with local +Z aligned to WORLD +Z.

    使用 tool-X/tool-Y 中水平投影更稳定的一根轴建立方位，避免水平抓取
    时把接近竖直的 tool-Y 投影归一化，进而将微小 IK 数值噪声放大为
    Contact Camera 的大幅偏航。若给出上一 Contact frame，则只消除 180°
    符号翻转；frame 的生命周期和冻结由调用方管理。
    """

    rotation = Rotation.from_quat(np.asarray(pose.quaternion_xyzw, dtype=np.float64)).as_matrix()
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    projected_x = rotation[:, 0].copy()
    projected_y = rotation[:, 1].copy()
    projected_x[2] = 0.0
    projected_y[2] = 0.0
    norm_x = float(np.linalg.norm(projected_x))
    norm_y = float(np.linalg.norm(projected_y))

    # 对合法旋转矩阵，tool-X/tool-Y 不可能同时平行于 WORLD-Z，因此较大
    # 的水平投影总是良定义；这里不需要经验阈值或世界轴硬编码兜底。
    # 两根轴同样稳定时沿用原来的 tool-Y 语义，避免改变常见竖直抓取的
    # FRONT/SIDE 定义；只有 tool-X 明显拥有更大的水平投影时才切换。
    if norm_y >= norm_x:
        locked_y = projected_y / norm_y
        locked_x = np.cross(locked_y, world_z)
    else:
        locked_x = projected_x / norm_x
        locked_y = np.cross(world_z, locked_x)
    locked_x /= np.linalg.norm(locked_x)
    locked_y = np.cross(world_z, locked_x)
    locked_y /= np.linalg.norm(locked_y)

    if reference_quaternion_xyzw is not None:
        reference_x = Rotation.from_quat(
            np.asarray(reference_quaternion_xyzw, dtype=np.float64)
        ).as_matrix()[:, 0]
        if float(np.dot(locked_x, reference_x)) < 0.0:
            locked_x = -locked_x
            locked_y = -locked_y

    matrix = np.column_stack((locked_x, locked_y, world_z))
    quaternion = Rotation.from_matrix(matrix).as_quat()
    return tuple(float(value) for value in quaternion)


def _gravity_stable_contact_rotation(rotation: np.ndarray) -> np.ndarray:
    quaternion = gravity_stable_contact_frame_quaternion(
        Pose(
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in Rotation.from_matrix(rotation).as_quat()),
        )
    )
    return Rotation.from_quat(quaternion).as_matrix()


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
    del view, current_tcp_position, current_tcp_rotation
    edit = preview.visual_edit
    target = preview.target_pose
    if edit is None or target is None or edit.kind not in {"delta_move", "rotate"}:
        return
    if edit.kind == "delta_move" and edit.delta_xyz_m is not None:
        delta_cm = np.asarray(edit.delta_xyz_m, dtype=np.float64) * 100.0
        label = (
            f"MOVE {edit.frame.upper()}  "
            f"dX {delta_cm[0]:+.1f}  dY {delta_cm[1]:+.1f}  dZ {delta_cm[2]:+.1f} cm"
        )
        _draw_adjustment_label(draw, label, width, height, _MOVE)
        return
    if edit.kind != "rotate" or edit.axis is None or edit.angle_deg is None:
        return
    axis_color = _AXIS_COLORS["xyz".index(edit.axis)]
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


def _translation_title_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", 23)
    except OSError:
        return ImageFont.load_default()


def _translation_axis_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", 26)
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
    "GRASP_SWEEP_RGB",
    "GRIPPER_Z_AXIS_RGB",
    "NEAR_FIELD_HEIGHT",
    "NEAR_FIELD_WIDTH",
    "NearFieldPreview",
    "draw_grasp_sweep_overlay",
    "draw_gripper_z_axis_overlay",
    "gravity_stable_contact_frame_quaternion",
    "render_contact_auxiliary",
    "render_contact_focus",
]
