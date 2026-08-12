"""Dense camera-aligned RGB-D scene views for the VIA-style Context Canvas.

The previous implementation projected one depth image into a substantially new
orthographic viewpoint.  Every surface hidden from the source camera then became
a conspicuous hole.  Increasing the point radius only blurred object boundaries.

This presenter instead keeps the calibrated agentview perspective (the same
choice that makes VIA's scene surface visually dense), performs a target-centred
crop for imagination, and overlays exact URDF silhouettes in that camera.  Raw
depth and calibration remain private to the compiler.
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
    thick_mask_outline,
)
from vaw.context_runtime.model import Pose, RobotState
from vaw.context_runtime.near_field import NearFieldPreview

OBSERVED_SCENE_WIDTH = 960
OBSERVED_SCENE_HEIGHT = 570
IMAGINATION_SCENE_WIDTH = 1200
IMAGINATION_SCENE_HEIGHT = 720
CONTACT_FOCUS_WIDTH_M = 0.32

_CURRENT_OUTLINE = np.array([255, 255, 255], dtype=np.uint8)
_VIOLET = np.array([124, 58, 237], dtype=np.uint8)
_VIOLET_EDGE = np.array([196, 181, 253], dtype=np.uint8)
_AXIS_COLORS = ((239, 68, 68), (34, 197, 94), (59, 130, 246))


@dataclass(frozen=True)
class _CameraData:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    base_from_camera: np.ndarray
    points_base: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class _RasterMap:
    left: float
    top: float
    crop_width: float
    crop_height: float
    output_width: int
    output_height: int

    def pixels(self, uv: np.ndarray) -> np.ndarray:
        values = np.asarray(uv, dtype=np.float64)
        out = values.copy()
        out[..., 0] = (values[..., 0] - self.left) * (self.output_width / self.crop_width)
        out[..., 1] = (values[..., 1] - self.top) * (self.output_height / self.crop_height)
        return out


@dataclass(frozen=True)
class _FocusSpec:
    center_uv: np.ndarray
    crop_width_px: float


def render_scene_view(
    agentview: dict,
    wrist: dict | None,
    robot: RobotState | None,
    preview: NearFieldPreview | None = None,
    *,
    dark: bool,
    source_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render current dense RGB-D with an optional unexecuted target gripper."""

    del wrist  # Wrist RGB-D remains in the contact-local renderer and raw trace.
    camera = _camera_data(agentview)
    if dark:
        width, height = IMAGINATION_SCENE_WIDTH, IMAGINATION_SCENE_HEIGHT
    else:
        width, height = OBSERVED_SCENE_WIDTH, OBSERVED_SCENE_HEIGHT
    if camera is None:
        color = (3, 7, 14) if dark else (247, 250, 253)
        return np.full((height, width, 3), color, dtype=np.uint8)

    focus = _focus_spec(camera, robot, preview, source_mask) if dark else None
    mapping = _crop_mapping(
        camera.rgb.shape[1],
        camera.rgb.shape[0],
        width,
        height,
        focus=focus.center_uv if focus is not None else None,
        crop_width_hint=focus.crop_width_px if focus is not None else None,
        target_focused=dark and focus is not None,
    )
    native = _dense_surface(camera, dark=dark)

    crop = Image.fromarray(native).crop(
        (
            int(round(mapping.left)),
            int(round(mapping.top)),
            int(round(mapping.left + mapping.crop_width)),
            int(round(mapping.top + mapping.crop_height)),
        )
    )
    output_array = np.asarray(
        crop.resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ).copy()
    if dark and source_mask is not None:
        _overlay_source_surface(output_array, source_mask, mapping)
    # Rasterize geometry at final resolution.  Compositing at the source depth
    # resolution and then enlarging it made diagonal fingers look soft and
    # mask-like; final-resolution silhouettes stay crisp and hole-free.
    if robot is not None:
        _overlay_current(output_array, camera, mapping, robot)
        if preview is not None:
            _overlay_reference_target(output_array, camera, mapping, robot, preview)
            _overlay_preview(output_array, camera, mapping, robot, preview)
    output = Image.fromarray(output_array)
    draw = ImageDraw.Draw(output, "RGBA")
    if dark:
        _draw_world_axes(draw, camera, mapping)
    return np.asarray(output, dtype=np.uint8)


def _camera_data(camera: dict) -> _CameraData | None:
    try:
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return None
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        return None
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        return None
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    valid_depth = np.isfinite(depth) & (depth >= 0.015) & (depth <= 20.0)
    safe_depth = np.where(valid_depth, depth, 1.0)
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
    finite = np.isfinite(points_base).all(axis=-1)
    # Remove the distant walls and floor outside the manipulation workspace.
    workspace = (
        (points_base[..., 0] >= -0.04)
        & (points_base[..., 0] <= 0.82)
        & (np.abs(points_base[..., 1]) <= 0.40)
        & (points_base[..., 2] >= -0.04)
        & (points_base[..., 2] <= 0.72)
    )
    return _CameraData(
        rgb=np.ascontiguousarray(rgb),
        depth=np.ascontiguousarray(depth),
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        points_base=np.ascontiguousarray(points_base),
        valid=np.ascontiguousarray(valid_depth & finite & workspace),
    )


def _dense_surface(camera: _CameraData, *, dark: bool) -> np.ndarray:
    background = np.array([3, 7, 14] if dark else [247, 250, 253], dtype=np.uint8)
    image = np.empty_like(camera.rgb)
    image[:] = background
    colors = camera.rgb
    if dark:
        colors = np.asarray(
            np.clip(colors.astype(np.float64) * 0.82 + 24.0, 0, 255),
            dtype=np.uint8,
        )
    image[camera.valid] = colors[camera.valid]
    # Close only one-pixel sampling holes supported on at least three sides.
    # Large disocclusions and true object boundaries remain untouched.
    return _fill_small_holes(image, camera.valid, iterations=2)


def _fill_small_holes(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    result = image.copy()
    mask = valid.copy()
    for _ in range(iterations):
        padded_mask = np.pad(mask, 1, mode="constant")
        padded_rgb = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        neighbour_masks = (
            padded_mask[:-2, 1:-1],
            padded_mask[2:, 1:-1],
            padded_mask[1:-1, :-2],
            padded_mask[1:-1, 2:],
        )
        count = sum(item.astype(np.uint8) for item in neighbour_masks)
        fill = ~mask & (count >= 3)
        if not np.any(fill):
            break
        total = np.zeros_like(result, dtype=np.float64)
        for neighbour_mask, neighbour_rgb in zip(
            neighbour_masks,
            (
                padded_rgb[:-2, 1:-1],
                padded_rgb[2:, 1:-1],
                padded_rgb[1:-1, :-2],
                padded_rgb[1:-1, 2:],
            ),
            strict=True,
        ):
            total += neighbour_rgb * neighbour_mask[..., None]
        result[fill] = np.asarray(
            np.round(total[fill] / count[fill, None]),
            dtype=np.uint8,
        )
        mask[fill] = True
    return result


def _crop_mapping(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    *,
    focus: np.ndarray | None,
    crop_width_hint: float | None,
    target_focused: bool,
) -> _RasterMap:
    aspect = output_width / output_height
    if target_focused and crop_width_hint is not None:
        crop_width = float(np.clip(crop_width_hint, 1.0, source_width))
    else:
        crop_width = float(source_width)
    crop_height = crop_width / aspect
    if crop_height > source_height:
        crop_height = float(source_height)
        crop_width = crop_height * aspect
    center_x = float(focus[0]) if focus is not None else source_width * 0.5
    center_y = float(focus[1]) if focus is not None else source_height * 0.52
    left = float(np.clip(center_x - crop_width * 0.5, 0.0, source_width - crop_width))
    top = float(np.clip(center_y - crop_height * 0.5, 0.0, source_height - crop_height))
    return _RasterMap(left, top, crop_width, crop_height, output_width, output_height)


def _focus_spec(
    camera: _CameraData,
    robot: RobotState | None,
    preview: NearFieldPreview | None,
    source_mask: np.ndarray | None,
) -> _FocusSpec | None:
    pose = None
    if preview is not None:
        pose = preview.target_pose
    if pose is None and robot is not None:
        pose = robot.tcp_pose
    if pose is None:
        return None
    uv, depth = _project_base(
        np.asarray(pose.position_xyz, dtype=np.float64)[None, :],
        camera,
    )
    if depth[0] <= 0 or not np.isfinite(uv[0]).all():
        return None
    # TCP alone is not the visible target: reserve calibrated room for the
    # palm and fingers before unioning it with the source object evidence.
    target_half_width = float(camera.intrinsics[0, 0]) * 0.08 / float(depth[0])
    target_half_height = float(camera.intrinsics[1, 1]) * 0.10 / float(depth[0])
    lower = uv[0] - np.array([target_half_width, target_half_height])
    upper = uv[0] + np.array([target_half_width, target_half_height])
    if source_mask is not None and source_mask.shape == camera.depth.shape:
        ys, xs = np.nonzero(source_mask)
        if len(xs):
            lower = np.minimum(lower, np.array([xs.min(), ys.min()], dtype=np.float64))
            upper = np.maximum(upper, np.array([xs.max(), ys.max()], dtype=np.float64))
    center = (lower + upper) * 0.5
    aspect = IMAGINATION_SCENE_WIDTH / IMAGINATION_SCENE_HEIGHT
    metric_width = float(camera.intrinsics[0, 0]) * CONTACT_FOCUS_WIDTH_M / float(depth[0])
    union = np.maximum(upper - lower, 1.0)
    required_width = max(
        metric_width,
        float(union[0]) * 1.45,
        float(union[1]) * aspect * 1.45,
    )
    return _FocusSpec(center_uv=center, crop_width_px=required_width)


def _overlay_source_surface(
    image: np.ndarray,
    source_mask: np.ndarray,
    mapping: _RasterMap,
) -> None:
    mask = np.asarray(source_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return
    left = int(round(mapping.left))
    top = int(round(mapping.top))
    right = int(round(mapping.left + mapping.crop_width))
    bottom = int(round(mapping.top + mapping.crop_height))
    cropped = Image.fromarray(mask.astype(np.uint8) * 255).crop((left, top, right, bottom))
    visible = (
        np.asarray(
            cropped.resize(
                (mapping.output_width, mapping.output_height),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )
    if not np.any(visible):
        return
    emphasis = np.array([14, 165, 233], dtype=np.uint8)
    image[visible] = np.asarray(
        np.round(image[visible] * 0.84 + emphasis * 0.16),
        dtype=np.uint8,
    )
    image[mask_outline(visible)] = np.array([125, 211, 252], dtype=np.uint8)


def _project_base(
    points_base: np.ndarray,
    camera: _CameraData,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    camera_from_base = np.linalg.inv(camera.base_from_camera)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    points_camera = (homogeneous @ camera_from_base.T)[:, :3]
    depth = points_camera[:, 2]
    uvw = points_camera @ camera.intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / depth[:, None]
    return uv, depth


def _overlay_current(
    image: np.ndarray,
    camera: _CameraData,
    mapping: _RasterMap,
    robot: RobotState,
) -> None:
    if robot.joint_positions_rad is None or robot.gripper_opening is None:
        return
    triangles = _robot_triangles(robot.joint_positions_rad, robot.gripper_opening)
    if triangles is None:
        return
    mask = _triangle_mask(image, camera, mapping, triangles)
    if np.any(mask):
        image[thick_mask_outline(mask, radius=2)] = _CURRENT_OUTLINE


def _overlay_preview(
    image: np.ndarray,
    camera: _CameraData,
    mapping: _RasterMap,
    robot: RobotState,
    preview: NearFieldPreview,
) -> None:
    triangles = None
    if preview.joint_positions_rad is not None and preview.gripper_opening is not None:
        triangles = _robot_triangles(
            preview.joint_positions_rad,
            preview.gripper_opening,
        )
    elif preview.target_pose is not None and preview.gripper_opening is not None:
        triangles = _target_gripper_triangles(robot, preview)
    if triangles is None:
        return
    _overlay_triangles(
        image,
        camera,
        mapping,
        triangles,
        _VIOLET,
        _VIOLET_EDGE,
        alpha=0.90,
    )


def _overlay_reference_target(
    image: np.ndarray,
    camera: _CameraData,
    mapping: _RasterMap,
    robot: RobotState,
    preview: NearFieldPreview,
) -> None:
    edit = preview.visual_edit
    if edit is None or edit.reference_pose is None or preview.gripper_opening is None:
        return
    reference = NearFieldPreview(
        target_pose=edit.reference_pose,
        joint_positions_rad=None,
        gripper_opening=preview.gripper_opening,
    )
    triangles = _target_gripper_triangles(robot, reference)
    if triangles is None:
        return
    mask = _triangle_mask(image, camera, mapping, triangles)
    if not np.any(mask):
        return
    outline = mask_outline(mask)
    for _ in range(2):
        padded = np.pad(outline, 1, mode="constant")
        outline = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    image[outline] = np.array([226, 232, 240], dtype=np.uint8)


def _robot_triangles(
    joints: tuple[float, ...],
    opening: float,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        return fk.triangles(np.asarray(joints, dtype=np.float64), float(opening))
    except (RuntimeError, ValueError):
        return None


def _target_gripper_triangles(
    robot: RobotState,
    preview: NearFieldPreview,
) -> np.ndarray | None:
    if (
        robot.joint_positions_rad is None
        or robot.tcp_pose is None
        or preview.target_pose is None
        or preview.gripper_opening is None
    ):
        return None
    triangles = _robot_triangles(robot.joint_positions_rad, preview.gripper_opening)
    if triangles is None:
        return None
    current_position = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64)
    current_rotation = Rotation.from_quat(robot.tcp_pose.quaternion_xyzw).as_matrix()
    target_position = np.asarray(preview.target_pose.position_xyz, dtype=np.float64)
    target_rotation = Rotation.from_quat(preview.target_pose.quaternion_xyzw).as_matrix()
    local = (triangles - current_position) @ current_rotation
    return local @ target_rotation.T + target_position


def _overlay_triangles(
    image: np.ndarray,
    camera: _CameraData,
    mapping: _RasterMap,
    triangles: np.ndarray,
    color: np.ndarray,
    edge: np.ndarray,
    *,
    alpha: float,
) -> None:
    mask = _triangle_mask(image, camera, mapping, triangles)
    if not np.any(mask):
        return
    image[mask] = np.asarray(
        np.round(image[mask] * (1.0 - alpha) + color * alpha),
        dtype=np.uint8,
    )
    image[mask_outline(mask)] = edge


def _triangle_mask(
    image: np.ndarray,
    camera: _CameraData,
    mapping: _RasterMap,
    triangles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    uv, depth = _project_base(values.reshape(-1, 3), camera)
    uv = mapping.pixels(uv)
    return rasterize_silhouette(
        uv.reshape(-1, 3, 2),
        depth.reshape(-1, 3),
        image.shape[1],
        image.shape[0],
    )


def _draw_world_axes(
    draw: ImageDraw.ImageDraw,
    camera: _CameraData,
    mapping: _RasterMap,
) -> None:
    origin_base = np.array([0.42, 0.0, 0.12], dtype=np.float64)
    points = np.vstack((origin_base, origin_base + np.eye(3) * 0.12))
    uv, depth = _project_base(points, camera)
    uv = mapping.pixels(uv)
    if not np.isfinite(uv).all() or np.any(depth <= 0):
        return
    # Use the calibrated directions but place the readable compass in a fixed
    # corner, exactly as VIA does.
    origin = np.array([mapping.output_width - 145.0, 82.0])
    draw.text(
        (origin[0] - 116, origin[1] - 58),
        "BASE / WORLD",
        fill=(226, 232, 240, 255),
        font=_font(15),
    )
    for index, (name, color) in enumerate(zip(("+X", "+Y", "+Z"), _AXIS_COLORS, strict=True)):
        direction = uv[index + 1] - uv[0]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        endpoint = origin + direction / norm * 48.0
        _arrow(draw, tuple(origin), tuple(endpoint), color, width=8)
        _boxed_label(
            draw,
            tuple(endpoint + direction / norm * 16.0),
            name,
            color,
            _font(18),
        )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    vector = np.asarray(end) - np.asarray(start)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0:
        return
    unit = vector / norm
    normal = np.array([-unit[1], unit[0]])
    tip = np.asarray(end)
    draw.line((*start, *end), fill=(*color, 255), width=width)
    draw.polygon(
        [tuple(tip), tuple(tip - unit * 17 + normal * 9), tuple(tip - unit * 17 - normal * 9)],
        fill=(*color, 255),
    )


def _boxed_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0] + 22, box[3] - box[1] + 14
    left, top = center[0] - width / 2, center[1] - height / 2
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=5,
        fill=(3, 7, 14, 235),
        outline=(*color, 255),
        width=3,
    )
    draw.text((left + 11, top + 7 - box[1]), text, fill=(255, 255, 255, 255), font=font)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


__all__ = [
    "IMAGINATION_SCENE_HEIGHT",
    "IMAGINATION_SCENE_WIDTH",
    "CONTACT_FOCUS_WIDTH_M",
    "OBSERVED_SCENE_HEIGHT",
    "OBSERVED_SCENE_WIDTH",
    "render_scene_view",
]
