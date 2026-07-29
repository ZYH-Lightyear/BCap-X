"""Scene point cloud: construction from RGB-D and rendering from any viewpoint.

This is what makes the main view's camera virtual. Every RGB-D camera in the
observation is deprojected into one world-frame coloured cloud, and rendering
is a z-buffered splat through a :class:`~vaw.camera.VirtualCamera`. Pure numpy;
no GPU, no browser, no scene graph — the canvas has to be reproducible
byte-for-byte from a trace, which rules out anything with its own frame loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image

from vaw.camera import VirtualCamera
from vaw.geometry import project_world_to_pixel

#: Keep only points inside a generous box around the table. LIBERO renders room
#: walls and floor that carry no task information but would otherwise dominate
#: the splat once the view tilts.
WORKSPACE_BOUNDS = ((-0.4, 1.6), (-1.0, 1.0), (-0.05, 1.5))

#: Depth shading range: nearest points keep their colour, far ones fade toward
#: the background. A monocular splat has no lighting, so this is the only cue
#: that survives a viewpoint change.
FOG_FLOOR = 0.55


@dataclass
class SceneCloud:
    """World-frame coloured points, fused from all cameras of one observation.

    ``footprints`` is the world-space size each point stands for (its source
    pixel's extent at its depth). Carrying it is what lets cameras of different
    resolutions and distances be fused without the splat betraying which camera
    a point came from: every point is drawn at its true size instead of a size
    picked for one sensor.
    """

    points: np.ndarray  # (N, 3) float64
    colors: np.ndarray  # (N, 3) uint8
    footprints: np.ndarray | None = None  # (N,) metres
    revision: int = 0

    def __len__(self) -> int:
        return len(self.points)

    @property
    def center(self) -> np.ndarray:
        if len(self.points) == 0:
            return np.array([0.5, 0.0, 0.1])
        return np.median(self.points, axis=0)


def deproject_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_mat: np.ndarray,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deproject a strided RGB-D frame to (points, colors, footprints)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = depth[::stride, ::stride]
    rgb = rgb[::stride, ::stride]

    K = np.asarray(intrinsics, dtype=np.float64)
    # On-surface spacing between neighbouring samples, not just the ray-normal
    # footprint: a surface seen at a grazing angle spreads consecutive samples
    # much further apart, and sizing splats by depth alone renders those
    # surfaces as stripes. The depth gradient recovers the stretch without
    # needing normals; the cap keeps silhouette discontinuities (where the
    # gradient is a cliff, not a slope) from smearing across the background.
    base = depth * stride / K[0, 0]
    slope = np.maximum(
        np.abs(np.gradient(depth, axis=1)), np.abs(np.gradient(depth, axis=0))
    )
    spacing = np.minimum(np.hypot(base, slope), base * 4.0)

    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    us = us.ravel() * stride
    vs = vs.ravel() * stride
    zs = depth.ravel()
    colors = rgb.reshape(-1, 3)
    footprints = spacing.ravel()

    valid = np.isfinite(zs) & (zs > 1e-6) & np.isfinite(footprints)
    us, vs, zs = us[valid], vs[valid], zs[valid]
    colors, footprints = colors[valid], footprints[valid]
    if len(zs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros(0)

    x = (us - K[0, 2]) / K[0, 0] * zs
    y = (vs - K[1, 2]) / K[1, 1] * zs
    cam_pts = np.stack([x, y, zs, np.ones_like(zs)], axis=1)
    world = (np.asarray(pose_mat, dtype=np.float64) @ cam_pts.T).T[:, :3]
    return world, colors, footprints


def build_scene_cloud(
    obs: dict[str, Any] | None,
    camera_names: Iterable[str],
    *,
    stride: int = 3,
    revision: int = 0,
    bounds: tuple[tuple[float, float], ...] | None = WORKSPACE_BOUNDS,
) -> SceneCloud:
    """Fuse every camera that reports depth + intrinsics + pose into one cloud.

    Fusing the wrist camera matters more than its resolution suggests: it sees
    the scene from a completely different angle, which is exactly the geometry
    the agentview cloud is missing when the agent orbits away from the mount.
    """
    empty = SceneCloud(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.zeros(0), revision
    )
    if obs is None:
        return empty

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_foot: list[np.ndarray] = []
    for name in camera_names:
        cam = obs.get(name)
        if not isinstance(cam, dict):
            continue
        images = cam.get("images", {})
        if "depth" not in images or "rgb" not in images:
            continue
        if cam.get("intrinsics") is None or cam.get("pose_mat") is None:
            continue
        pts, cols, foot = deproject_rgbd(
            images["rgb"], images["depth"], cam["intrinsics"], cam["pose_mat"], stride=stride
        )
        all_points.append(pts)
        all_colors.append(cols)
        all_foot.append(foot)

    if not all_points:
        return empty

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    footprints = np.concatenate(all_foot, axis=0)
    if bounds is not None and len(points):
        keep = np.ones(len(points), dtype=bool)
        for axis, (lo, hi) in enumerate(bounds):
            keep &= (points[:, axis] >= lo) & (points[:, axis] <= hi)
        points, colors, footprints = points[keep], colors[keep], footprints[keep]
    return SceneCloud(points, colors, footprints, revision)


def splat_cloud(
    cloud: SceneCloud,
    cam: VirtualCamera,
    *,
    background: tuple[int, int, int] = (24, 24, 28),
    tints: list[tuple[np.ndarray, tuple[int, int, int]]] | None = None,
    tint_weight: float = 0.4,
    max_radius: int = 3,
    scale: float = 0.5,
) -> np.ndarray:
    """Render ``cloud`` from ``cam``. Returns (H, W, 3) uint8.

    Z-buffering is done by painter's order: every splat pixel is emitted with
    its depth, the whole batch is sorted far-to-near, and the nearest write
    lands last. One sort replaces a per-pixel depth test and stays exact.

    :param tints: ``[(point_indices, rgb), ...]`` — colour blends marking which
        points belong to which grounded object. This is how object identity
        survives a viewpoint change: an agentview mask is pixels, but a tinted
        subset of the cloud is geometry and reprojects from any angle.
    :param tint_weight: how much of the marker colour to mix in. Kept low (and
        matched to the mask overlay alpha of the physical view) because a tint
        strong enough to hide the object's real colour trades one recognition
        cue for another instead of adding one.
    :param scale: resolution the splat is computed at before being scaled back
        up. A strided depth map puts its points several pixels apart at full
        resolution, so rasterising there spends four times the work drawing the
        same information with larger discs; half resolution costs ~6x less and
        the difference is not visible.
    """
    out_h, out_w = cam.height, cam.width
    w, h = max(int(out_w * scale), 1), max(int(out_h * scale), 1)
    intrinsics = np.asarray(cam.intrinsics, dtype=np.float64).copy()
    intrinsics[:2, :] *= scale

    img = np.empty((h * w, 3), dtype=np.uint8)
    img[:] = np.asarray(background, dtype=np.uint8)
    if len(cloud) == 0:
        return _to_output(img.reshape(h, w, 3), out_w, out_h)

    colors_all = cloud.colors.astype(np.float64)
    if tints:
        for indices, rgb in tints:
            if len(indices) == 0:
                continue
            tint = np.asarray(rgb, dtype=np.float64)
            colors_all[indices] = (1.0 - tint_weight) * colors_all[indices] + tint_weight * tint

    uvz = project_world_to_pixel(cloud.points, intrinsics, cam.pose_mat)
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    visible = (z > 0.05) & np.isfinite(u) & np.isfinite(v)
    visible &= (u > -8) & (u < w + 8) & (v > -8) & (v < h + 8)
    if not visible.any():
        return _to_output(img.reshape(h, w, 3), out_w, out_h)

    ui = np.rint(u[visible]).astype(np.int64)
    vi = np.rint(v[visible]).astype(np.int64)
    zi = z[visible]
    colors = colors_all[visible]

    # Depth fog: fade with distance so shape reads even without lighting.
    lo, hi = np.percentile(zi, [2, 98])
    span = max(hi - lo, 1e-6)
    fade = np.clip((zi - lo) / span, 0.0, 1.0)
    factor = (1.0 - (1.0 - FOG_FLOOR) * fade)[:, None]
    bg = np.asarray(background, dtype=np.float64)
    colors = colors * factor + bg * (1.0 - factor)
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    # Splat radius from each point's own world footprint, projected into this
    # view: a point covers the area its source pixel covered. Sizing by depth
    # alone made surfaces seen at a grazing angle (the wrist camera's view of
    # vertical faces) render as stripes, because their true footprint is much
    # larger than their distance suggests.
    if cloud.footprints is not None and len(cloud.footprints) == len(cloud.points):
        foot = cloud.footprints[visible]
    else:
        foot = np.full(len(zi), 0.004)
    # Radius is half the projected spacing (plus a little): just enough for
    # neighbouring points to meet, since anything more is pure overdraw.
    radii_px = foot * float(intrinsics[0, 0]) / np.maximum(zi, 1e-6) * 0.6
    radii = np.clip(np.rint(radii_px), 1, max_radius).astype(np.int64)

    flat_idx: list[np.ndarray] = []
    flat_z: list[np.ndarray] = []
    flat_c: list[np.ndarray] = []
    for r in range(1, max_radius + 1):
        sel = radii == r
        if not sel.any():
            continue
        su, sv, sz, sc = ui[sel], vi[sel], zi[sel], colors[sel]
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                pu, pv = su + du, sv + dv
                inside = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
                if not inside.any():
                    continue
                flat_idx.append(pv[inside] * w + pu[inside])
                flat_z.append(sz[inside])
                flat_c.append(sc[inside])

    if not flat_idx:
        return _to_output(img.reshape(h, w, 3), out_w, out_h)

    indices = np.concatenate(flat_idx)
    depths = np.concatenate(flat_z).astype(np.float32)
    cols = np.concatenate(flat_c, axis=0)
    order = np.argsort(-depths, kind="stable")
    img[indices[order]] = cols[order]
    return _to_output(img.reshape(h, w, 3), out_w, out_h)


def _to_output(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if img.shape[1] == out_w and img.shape[0] == out_h:
        return img
    return np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.BILINEAR))
