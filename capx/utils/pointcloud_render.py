"""Offscreen point cloud rendering and gripper pose overlay (verifier evidence, M1).

Pure numpy + OpenCV implementation (z-buffer splatting). No GL / EGL offscreen
context is required, so it works on any headless server.

Conventions (matching the rest of the codebase, see
``FrankaLiberoApiReducedSkillLibrary.mask_to_world_points``):

- ``intrinsics``: (3, 3) pinhole camera matrix K.
- ``pose_mat``:   (4, 4) camera-to-world transform. Camera frame is OpenCV
  style: +z forward, +x right, +y down.
- Grasp / gripper poses: (4, 4) world-frame transform of the ``panda_hand``
  link. +z is the approach direction, fingers slide along +y.
- Quaternions are wxyz unless stated otherwise.

Main entry points:

- ``render_point_cloud``       (V1.1) multi-view offscreen rendering of a scene
  cloud with optional highlighted subsets and gripper wireframes.
- ``draw_gripper_on_image``    (V1.2) project a gripper wireframe onto a real
  camera image (e.g. agentview RGB).
- ``auto_virtual_cameras``     deterministic virtual viewpoints framed around
  the scene (does NOT move the robot - fixed-viewpoint principle holds).

Usage:
    from capx.utils.pointcloud_render import (
        render_point_cloud,
        draw_gripper_on_image,
        pose_from_position_wxyz,
    )
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Gripper wireframe model (Franka panda hand, simplified)
# ---------------------------------------------------------------------------

# panda_hand origin -> finger base ~= 0.058 m, finger length ~= 0.045 m
# (origin + 0.103 ~= TCP between fingertips)
HAND_DEPTH = 0.058
FINGER_LEN = 0.045
DEFAULT_GRIPPER_WIDTH = 0.08  # max opening of the panda gripper

# Highlight palette (RGB), order: orange, green, purple, blue, red
HIGHLIGHT_PALETTE = [
    (255, 140, 0),
    (60, 200, 60),
    (170, 60, 220),
    (40, 120, 255),
    (230, 40, 40),
]


def gripper_segments(
    width: float = DEFAULT_GRIPPER_WIDTH,
    hand_depth: float = HAND_DEPTH,
    finger_len: float = FINGER_LEN,
) -> np.ndarray:
    """Line segments of a simplified gripper wireframe in the gripper frame.

    The gripper frame follows the panda_hand convention: +z is the approach
    direction, fingers slide along +y.

    Returns:
        (5, 2, 3) array of line segments (start, end):
        tail, palm bar, two fingers, and is closed by the approach arrow drawn
        separately in :func:`draw_gripper_on_image`.
    """
    w = width / 2.0
    return np.array(
        [
            # tail: wrist towards palm
            [[0.0, 0.0, -0.04], [0.0, 0.0, hand_depth]],
            # palm bar (fingers slide along +y / -y)
            [[0.0, -w, hand_depth], [0.0, w, hand_depth]],
            # left finger
            [[0.0, -w, hand_depth], [0.0, -w, hand_depth + finger_len]],
            # right finger
            [[0.0, w, hand_depth], [0.0, w, hand_depth + finger_len]],
            # approach arrow shaft (drawn with an arrow head when rendered)
            [[0.0, 0.0, hand_depth], [0.0, 0.0, hand_depth + finger_len]],
        ]
    )


def pose_from_position_wxyz(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Build a (4, 4) pose matrix from a position and a wxyz quaternion."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return pose


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Camera-to-world pose looking from ``eye`` to ``target`` (OpenCV frame).

    +z points from eye to target, +x right, +y down.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    z_axis = target - eye
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        raise ValueError("eye and target coincide")
    z_axis = z_axis / z_norm
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:  # looking straight along up: pick fallback
        x_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    f = 0.5 * height / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]
    )


def auto_virtual_cameras(
    points: np.ndarray,
    image_size: tuple[int, int] = (480, 640),
    views: tuple[str, ...] = ("front", "side", "top"),
    fov_deg: float = 60.0,
    distance_scale: float = 1.25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Deterministic virtual cameras framed around the scene bounding box.

    These are *virtual* viewpoints rendered from an already-captured cloud;
    the robot is never moved (fixed-viewpoint principle).

    Args:
        points: (N, 3) world-frame points used to frame the cameras.
        image_size: (H, W) of the rendered images.
        views: subset of {"front", "side", "back", "top"}.
        fov_deg: vertical field of view.
        distance_scale: camera distance in units of scene extent.

    Returns:
        dict view name -> (intrinsics (3,3), pose_mat (4,4) cam-to-world).
    """
    h, w = image_size
    # robust framing: ignore outliers (walls, floor, stray depth noise)
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    center = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo))
    extent = max(extent, 0.3)
    dist = distance_scale * extent

    # azimuth (deg, around world z, 0 = +x axis), elevation (deg)
    angles = {
        "front": (0.0, 35.0),
        "side": (90.0, 30.0),
        "back": (180.0, 35.0),
        "top": (0.0, 80.0),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    K = default_intrinsics(w, h, fov_deg)
    for name in views:
        az, el = angles[name]
        az_r, el_r = np.deg2rad(az), np.deg2rad(el)
        offset = dist * np.array(
            [np.cos(el_r) * np.cos(az_r), np.cos(el_r) * np.sin(az_r), np.sin(el_r)]
        )
        out[name] = (K.copy(), look_at_pose(center + offset, center))
    return out


def project_points(
    points: np.ndarray, intrinsics: np.ndarray, pose_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a camera.

    Args:
        points: (N, 3) world-frame points.
        intrinsics: (3, 3) K.
        pose_mat: (4, 4) camera-to-world.

    Returns:
        (uv (N, 2) float pixel coords, z (N,) camera depth, valid (N,) bool).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_to_cam = np.linalg.inv(pose_mat)
    pts_cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(points), 2))
    zz = np.where(valid, z, 1.0)
    uv[:, 0] = intrinsics[0, 0] * pts_cam[:, 0] / zz + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * pts_cam[:, 1] / zz + intrinsics[1, 2]
    return uv, z, valid


# ---------------------------------------------------------------------------
# Point splatting (z-buffer)
# ---------------------------------------------------------------------------


def _splat(
    canvas: np.ndarray,
    zbuf: np.ndarray,
    uv: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
    point_size: int,
) -> None:
    """Vectorized z-buffered point splatting into ``canvas`` (in place)."""
    h, w = canvas.shape[:2]
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    r = max(int(point_size) // 2, 0)
    offsets = [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1)]
    for du, dv in offsets:
        uu = u + du
        vv = v + dv
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        if not ok.any():
            continue
        flat = vv[ok] * w + uu[ok]
        zo = z[ok]
        co = colors[ok]
        # nearest point wins: sort so the closest is written last
        order = np.argsort(-zo)
        flat, zo, co = flat[order], zo[order], co[order]
        zflat = zbuf.reshape(-1)
        cflat = canvas.reshape(-1, 3)
        closer = zo < zflat[flat]
        # later duplicates in `flat` overwrite earlier ones, and since we
        # sorted far -> near, the nearest point ends up in the buffer
        zflat[flat[closer]] = zo[closer]
        cflat[flat[closer]] = co[closer]


def _depth_shade(z: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Slightly darken far points to convey depth."""
    if len(z) == 0:
        return base
    z0, z1 = np.percentile(z, 5), np.percentile(z, 95)
    t = np.clip((z - z0) / max(z1 - z0, 1e-6), 0.0, 1.0)
    scale = (1.0 - 0.45 * t)[:, None]
    return np.clip(base * scale, 0, 255)


# ---------------------------------------------------------------------------
# V1.2: gripper wireframe drawing
# ---------------------------------------------------------------------------


def draw_gripper_on_image(
    image: np.ndarray,
    grasp_pose: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    width: float = DEFAULT_GRIPPER_WIDTH,
    color: tuple[int, int, int] = (0, 220, 0),
    thickness: int = 2,
    label: str | None = None,
    draw_approach_arrow: bool = True,
) -> np.ndarray:
    """Project a gripper wireframe onto a camera image (V1.2).

    Args:
        image: (H, W, 3) uint8 RGB image (real camera frame or rendered view).
        grasp_pose: (4, 4) world-frame pose of the panda_hand link
            (+z approach, fingers along +y).
        intrinsics: (3, 3) K of the camera.
        pose_mat: (4, 4) camera-to-world of the camera.
        width: gripper opening width in meters (palm bar / finger spacing).
        color: RGB wireframe color.
        thickness: line thickness in pixels.
        label: optional text label drawn near the gripper origin.
        draw_approach_arrow: draw an arrowhead along +z (approach direction).

    Returns:
        A copy of ``image`` with the wireframe drawn.
    """
    out = np.ascontiguousarray(image.copy())
    segs = gripper_segments(width=width)
    rot, trans = grasp_pose[:3, :3], grasp_pose[:3, 3]
    pts = segs.reshape(-1, 3) @ rot.T + trans  # (10, 3) world
    uv, z, valid = project_points(pts, intrinsics, pose_mat)
    uv = uv.reshape(-1, 2, 2)
    z = z.reshape(-1, 2)
    valid = valid.reshape(-1, 2)

    n_segs = len(segs)
    for i in range(n_segs):
        if not valid[i].all():
            continue
        p0 = tuple(np.round(uv[i, 0]).astype(int))
        p1 = tuple(np.round(uv[i, 1]).astype(int))
        if i == n_segs - 1:  # approach arrow shaft
            if draw_approach_arrow:
                cv2.arrowedLine(out, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    if label is not None and valid[0].all():
        org = (int(uv[0, 0, 0]) + 6, int(uv[0, 0, 1]) - 6)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# V1.1: multi-view point cloud rendering
# ---------------------------------------------------------------------------


def render_point_cloud(
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None = None,
    highlights: list[dict] | None = None,
    grippers: list[dict] | None = None,
    views: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    image_size: tuple[int, int] = (480, 640),
    point_size: int = 3,
    background: tuple[int, int, int] = (255, 255, 255),
    max_points: int = 250_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Render a point cloud from virtual viewpoints with z-buffering (V1.1).

    Args:
        scene_points: (N, 3) world-frame scene cloud.
        scene_colors: optional (N, 3) uint8 RGB per point. Defaults to gray
            with depth shading.
        highlights: optional list of dicts, each with keys:
            ``points`` (M, 3) required; ``color`` (3,) RGB optional (palette
            fallback); ``label`` str optional (drawn at the subset centroid).
        grippers: optional list of dicts, each with keys:
            ``pose`` (4, 4) required; ``width`` float optional;
            ``color`` (3,) optional; ``label`` str optional.
        views: dict name -> (intrinsics, pose_mat). Defaults to
            :func:`auto_virtual_cameras` framed on the scene.
        image_size: (H, W) of rendered images.
        point_size: splat diameter in pixels.
        background: RGB background color.
        max_points: random subsample threshold for speed.
        seed: subsample RNG seed (deterministic output).

    Returns:
        dict view name -> (H, W, 3) uint8 RGB image.
    """
    scene_points = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    h, w = image_size

    rng = np.random.default_rng(seed)
    if len(scene_points) > max_points:
        idx = rng.choice(len(scene_points), max_points, replace=False)
        scene_points = scene_points[idx]
        if scene_colors is not None:
            scene_colors = np.asarray(scene_colors)[idx]

    if scene_colors is None:
        base_colors = np.full((len(scene_points), 3), 160.0)
        shade_scene = True
    else:
        base_colors = np.asarray(scene_colors, dtype=np.float64).reshape(-1, 3)
        shade_scene = False

    highlights = highlights or []
    grippers = grippers or []

    frame_pts = [scene_points] + [
        np.asarray(hl["points"], dtype=np.float64).reshape(-1, 3) for hl in highlights
    ]
    all_frame = np.concatenate([p for p in frame_pts if len(p)], axis=0)
    if views is None:
        views = auto_virtual_cameras(all_frame, image_size=image_size)

    out: dict[str, np.ndarray] = {}
    for name, (K, cam_pose) in views.items():
        canvas = np.full((h, w, 3), background, dtype=np.float64)
        zbuf = np.full((h, w), np.inf)

        # scene
        uv, z, valid = project_points(scene_points, K, cam_pose)
        colors = _depth_shade(z[valid], base_colors[valid]) if shade_scene else base_colors[valid]
        _splat(canvas, zbuf, uv[valid], z[valid], colors, point_size)

        # highlighted subsets (slightly larger splats, on top by z-test)
        label_anchors: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
        for j, hl in enumerate(highlights):
            pts = np.asarray(hl["points"], dtype=np.float64).reshape(-1, 3)
            if len(pts) == 0:
                continue
            color = tuple(hl.get("color", HIGHLIGHT_PALETTE[j % len(HIGHLIGHT_PALETTE)]))
            uv_h, z_h, valid_h = project_points(pts, K, cam_pose)
            # small epsilon so highlights win z-fights against the same points
            _splat(
                canvas,
                zbuf,
                uv_h[valid_h],
                z_h[valid_h] - 1e-4,
                np.tile(np.array(color, dtype=np.float64), (int(valid_h.sum()), 1)),
                point_size + 2,
            )
            if hl.get("label") and valid_h.any():
                centroid = pts[valid_h].mean(axis=0, keepdims=True)
                uv_c, _, ok = project_points(centroid, K, cam_pose)
                if ok[0]:
                    label_anchors.append(
                        (str(hl["label"]), (int(uv_c[0, 0]), int(uv_c[0, 1])), color)
                    )

        img = np.clip(canvas, 0, 255).astype(np.uint8)

        # gripper wireframes are drawn on top (vector graphics, no z-test)
        for g in grippers:
            img = draw_gripper_on_image(
                img,
                np.asarray(g["pose"], dtype=np.float64),
                K,
                cam_pose,
                width=float(g.get("width", DEFAULT_GRIPPER_WIDTH)),
                color=tuple(g.get("color", (0, 220, 0))),
                label=g.get("label"),
            )

        for text, (u0, v0), color in label_anchors:
            org = (u0 + 6, v0 - 6)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # view tag
        cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out[name] = img

    return out


def render_ortho_section(
    points: np.ndarray,
    colors: np.ndarray | None,
    center: np.ndarray,
    view_dir: np.ndarray = (0.0, 1.0, 0.0),
    slab: float = 0.04,
    px_per_m: float = 2000.0,
    image_size: tuple[int, int] = (480, 640),
    grid_step: float = 0.01,
    z_lines: list[tuple[float, str, tuple[int, int, int]]] | None = None,
    markers: list[dict] | None = None,
    point_size: int = 3,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Orthographic horizontal cross-section with a metric grid.

    Unlike perspective views, this projection maps vertical distances to
    pixels LINEARLY (no foreshortening), so heights are readable: 1 cm is
    always ``grid_step * px_per_m`` pixels. Only points inside a thin slab
    around ``center`` are drawn, keeping the section clean.

    Args:
        points: (N, 3) world points.
        colors: optional (N, 3) uint8 RGB per point.
        center: (3,) section center (slab passes through it).
        view_dir: horizontal viewing direction (z component is ignored).
        slab: slab thickness in meters along ``view_dir``.
        px_per_m: scale (2000 -> 1 cm = 20 px).
        image_size: (H, W) output size.
        grid_step: horizontal grid line spacing in meters (default 1 cm).
        z_lines: optional [(z_value, label, color)] horizontal reference
            lines, e.g. the support surface height.
        markers: optional [{"point": (3,), "label": str, "color": (3,)}].
        point_size: splat size in pixels.
        background: RGB background.

    Returns:
        (H, W, 3) uint8 image. Image x = horizontal in-plane axis,
        image y = world z (up). Grid labels are world z in centimeters.
    """
    h, w = image_size
    center = np.asarray(center, dtype=np.float64).reshape(3)
    d = np.asarray(view_dir, dtype=np.float64).copy()
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("view_dir must have a horizontal component")
    d = d / n
    r_axis = np.array([-d[1], d[0], 0.0])  # screen right, horizontal

    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - center
    depth = rel @ d
    keep = np.abs(depth) < slab / 2.0
    rel = rel[keep]
    depth = depth[keep]
    cols = (
        np.asarray(colors)[keep].astype(np.float64)
        if colors is not None
        else np.full((len(rel), 3), 150.0)
    )

    sx = rel @ r_axis * px_per_m + w / 2.0
    sy = -rel[:, 2] * px_per_m + h / 2.0
    uv = np.stack([sx, sy], axis=-1)

    canvas = np.full((h, w, 3), background, dtype=np.float64)
    zbuf = np.full((h, w), np.inf)
    # z-buffer wants smaller = closer; shift depth to positive
    _splat(canvas, zbuf, uv, depth - depth.min() + 1.0, cols, point_size)
    img = np.clip(canvas, 0, 255).astype(np.uint8)

    # metric grid: horizontal lines every grid_step (world z)
    z_center = center[2]
    half_span = h / 2.0 / px_per_m
    k0 = int(np.ceil((z_center - half_span) / grid_step))
    k1 = int(np.floor((z_center + half_span) / grid_step))
    for k in range(k0, k1 + 1):
        z_val = k * grid_step
        y = int(round(-(z_val - z_center) * px_per_m + h / 2.0))
        if not (0 <= y < h):
            continue
        major = (k * grid_step * 100) % 5 < 1e-6  # every 5 cm
        color = (170, 170, 170) if major else (220, 220, 220)
        cv2.line(img, (0, y), (w, y), color, 1)
        if major:
            cv2.putText(img, f"z={z_val * 100:.0f}cm", (4, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    for z_val, label, color in z_lines or []:
        y = int(round(-(z_val - z_center) * px_per_m + h / 2.0))
        if 0 <= y < h:
            cv2.line(img, (0, y), (w, y), color, 2)
            cv2.putText(img, label, (w - 230, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (w - 230, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    for m in markers or []:
        p = np.asarray(m["point"], dtype=np.float64) - center
        x = int(round(p @ r_axis * px_per_m + w / 2.0))
        y = int(round(-p[2] * px_per_m + h / 2.0))
        color = tuple(m.get("color", (0, 220, 0)))
        if 0 <= x < w and 0 <= y < h:
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            if m.get("label"):
                cv2.putText(img, str(m["label"]), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, str(m["label"]), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img


def draw_grasp_point_on_image(
    image: np.ndarray,
    grasp_point_pose: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 0),
    label: str | None = None,
    arrow_len: float = 0.09,
    point_radius: int = 6,
    thickness: int = 3,
) -> np.ndarray:
    """Minimal grasp visualization: a dot at the grasp point + approach arrow.

    The dot marks where the fingers close (TCP); the arrow points along the
    approach direction (+z of the pose), ending at the dot.

    Args:
        image: (H, W, 3) uint8 RGB.
        grasp_point_pose: (4, 4) world pose whose ORIGIN is the grasp point
            (TCP) and +z is the approach direction.
        intrinsics / pose_mat: camera K and cam-to-world.
        color: marker RGB color.
        label: optional text next to the dot.
        arrow_len: arrow length in meters (drawn behind the point, along -z).
        point_radius: dot radius in pixels.
        thickness: arrow thickness in pixels.

    Returns:
        A copy of ``image`` with the marker drawn.
    """
    out = np.ascontiguousarray(image.copy())
    origin = grasp_point_pose[:3, 3]
    approach = grasp_point_pose[:3, 2]
    tail = origin - approach * arrow_len
    uv, _, valid = project_points(np.stack([origin, tail]), intrinsics, pose_mat)
    if not valid.all():
        return out
    p_pt = tuple(np.round(uv[0]).astype(int))
    p_tail = tuple(np.round(uv[1]).astype(int))
    cv2.arrowedLine(out, p_tail, p_pt, color, thickness, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(out, p_pt, point_radius, color, -1, cv2.LINE_AA)
    cv2.circle(out, p_pt, point_radius, (0, 0, 0), 1, cv2.LINE_AA)
    if label:
        org = (p_pt[0] + 10, p_pt[1] - 8)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def grasp_marker_sample_points(
    grasp_point_pose: np.ndarray,
    arrow_len: float = 0.09,
    step: float = 0.002,
    point_radius: float = 0.006,
) -> np.ndarray:
    """Dense points tracing a grasp point marker (dot + approach arrow) for PLY export.

    Args:
        grasp_point_pose: (4, 4) world pose, origin = grasp point, +z = approach.
        arrow_len: arrow shaft length in meters.
        step: sampling distance.
        point_radius: radius of the dot (sampled as a small sphere).

    Returns:
        (N, 3) world points.
    """
    origin = grasp_point_pose[:3, 3]
    approach = grasp_point_pose[:3, 2]
    n = max(int(arrow_len / step), 2)
    t = np.linspace(0.0, 1.0, n)[:, None]
    shaft = (origin - approach * arrow_len)[None, :] * (1 - t) + origin[None, :] * t
    # small sphere at the grasp point
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(300, 3))
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True) * point_radius
    return np.concatenate([shaft, origin[None, :] + sphere], axis=0)


def wireframe_sample_points(
    pose: np.ndarray,
    width: float = DEFAULT_GRIPPER_WIDTH,
    step: float = 0.002,
) -> np.ndarray:
    """Sample dense points along the gripper wireframe (for PLY export).

    Args:
        pose: (4, 4) world-frame gripper pose.
        width: gripper opening width.
        step: sampling distance along each segment in meters.

    Returns:
        (N, 3) world-frame points tracing the wireframe.
    """
    segs = gripper_segments(width=width)
    pts = []
    for p0, p1 in segs:
        n = max(int(np.linalg.norm(p1 - p0) / step), 2)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(p0[None, :] * (1 - t) + p1[None, :] * t)
    local = np.concatenate(pts, axis=0)
    return local @ pose[:3, :3].T + pose[:3, 3]


def save_ply(
    path: str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Save a colored point cloud as a binary PLY (viewable in MeshLab/CloudCompare/Open3D).

    Args:
        path: output .ply path.
        points: (N, 3) float points.
        colors: optional (N, 3) uint8 RGB. Defaults to mid-gray.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 160, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    assert len(colors) == len(points)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    rec = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
    rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    rgb: np.ndarray | None = None,
    stride: int = 1,
    z_near: float = 0.01,
    z_far: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Deproject a full depth image to world-frame points (+ optional colors).

    Args:
        depth: (H, W) metric depth.
        intrinsics: (3, 3) K.
        pose_mat: (4, 4) camera-to-world.
        rgb: optional (H, W, 3) image to sample point colors from.
        stride: pixel subsampling stride.
        z_near / z_far: depth validity range in meters.

    Returns:
        (points (N, 3), colors (N, 3) uint8 or None).
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    d = depth[::stride, ::stride]
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs * stride
    ys = ys * stride
    z = d.reshape(-1)
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    valid = (z > z_near) & (z < z_far)
    z, xs, ys = z[valid], xs[valid], ys[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    pts_world = pts_cam @ pose_mat[:3, :3].T + pose_mat[:3, 3]
    colors = None
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.uint8)
    return pts_world, colors
