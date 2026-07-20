"""证据渲染器:把原始数组变成对 VLM 友好的图像。

渲染产物是 judge 的临时一等输入,而非技能资产。Phase 1 提供 before/after 渲染器
(gate 3);gate 1 的 mask/bbox/grasp 叠加图以后可复用 CapX 的 debug 叠加绘制。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def image_content_part(path: str | Path) -> dict:
    """Build an OpenAI-compatible image_url content part from a local PNG/JPG file."""

    data = base64.b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def save_rgb(path: str | Path, rgb: np.ndarray) -> str:
    """把一个 (H, W, 3) 的 uint8 数组存成 PNG;返回路径字符串。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return str(path)


def save_mask_overlay(
    path: str | Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    bbox: list[float] | tuple[float, float, float, float] | None = None,
) -> str:
    """Save a grounding review image with a translucent mask and bounding box."""

    rgb_array = np.asarray(rgb).astype(np.uint8)
    mask_array = np.asarray(mask).astype(bool)
    if rgb_array.ndim != 3 or rgb_array.shape[2] < 3:
        raise ValueError("rgb must have shape (H, W, 3).")
    if mask_array.shape != rgb_array.shape[:2]:
        raise ValueError("mask shape must match the RGB image.")

    image = Image.fromarray(rgb_array[..., :3]).convert("RGBA")
    _draw_mask_overlay(image, mask_array, color=(0, 255, 0, 80))
    if bbox is None and np.any(mask_array):
        ys, xs = np.nonzero(mask_array)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    if bbox is not None and len(bbox) == 4:
        ImageDraw.Draw(image).rectangle(
            [float(value) for value in bbox],
            outline=(255, 64, 64, 255),
            width=3,
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output)
    return str(output)


def project_world_to_pixel(
    point_world: np.ndarray | list[float] | tuple[float, ...],
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[float, float] | None:
    """Project a world-frame 3D point into image pixels.

    LIBERO camera poses are treated as camera-to-world transforms. Some backends expose
    camera z with the opposite sign, so this mirrors the robust local projection used in
    existing skills and accepts whichever sign lands inside the image.
    """

    height, width = int(image_shape[0]), int(image_shape[1])
    k = np.asarray(intrinsics, dtype=float)
    t_wc = np.asarray(camera_pose, dtype=float)
    p = np.asarray(point_world, dtype=float).reshape(-1)
    if p.shape[0] < 3:
        return None
    p_h = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    try:
        p_cam = np.linalg.inv(t_wc) @ p_h
    except np.linalg.LinAlgError:
        return None

    candidates: list[tuple[float, float, bool]] = []
    for z in (float(p_cam[2]), float(-p_cam[2])):
        if abs(z) < 1e-6:
            continue
        u = float(k[0, 0] * (p_cam[0] / z) + k[0, 2])
        v = float(k[1, 1] * (p_cam[1] / z) + k[1, 2])
        if np.isfinite(u) and np.isfinite(v):
            candidates.append((u, v, 0 <= u < width and 0 <= v < height))
    for u, v, inside in candidates:
        if inside:
            return u, v
    return (candidates[0][0], candidates[0][1]) if candidates else None


def save_grasp_affordance_overlay(
    path: str | Path,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    mask: np.ndarray | None = None,
    bbox: list[float] | tuple[float, float, float, float] | None = None,
    max_candidates: int = 8,
    axis_scale_m: float = 0.055,
) -> str:
    """Save a visual grasp-affordance overlay.

    Each candidate should contain ``pos`` and ``quat`` in world frame, with quaternion in
    wxyz order. The overlay marks:
    - grasp point: yellow dot
    - approach direction / gripper direction: orange arrow along local +z
    - gripper jaw axis: cyan line along local +x
    - IK-feasible candidates: green label; infeasible candidates: red label
    """

    image = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    height, width = np.asarray(rgb).shape[:2]

    if mask is not None:
        _draw_mask_overlay(image, np.asarray(mask).astype(bool), color=(0, 255, 0, 60))
    if bbox is not None and len(bbox) == 4:
        draw.rectangle([float(v) for v in bbox], outline=(255, 0, 0, 255), width=2)

    for idx, cand in enumerate(candidates[:max_candidates]):
        pos = np.asarray(cand.get("pos", cand.get("position", [])), dtype=float).reshape(-1)
        quat = np.asarray(cand.get("quat", cand.get("quaternion", [])), dtype=float).reshape(-1)
        if pos.shape[0] < 3 or quat.shape[0] < 4:
            continue
        center = project_world_to_pixel(pos[:3], intrinsics, camera_pose, (height, width))
        if center is None:
            continue
        rot = _quat_wxyz_to_matrix(quat[:4])
        approach_end = project_world_to_pixel(pos[:3] + rot[:, 2] * axis_scale_m, intrinsics, camera_pose, (height, width))
        jaw_a = project_world_to_pixel(pos[:3] - rot[:, 0] * axis_scale_m * 0.5, intrinsics, camera_pose, (height, width))
        jaw_b = project_world_to_pixel(pos[:3] + rot[:, 0] * axis_scale_m * 0.5, intrinsics, camera_pose, (height, width))

        u, v = center
        ik_ok = bool(cand.get("ik_ok", cand.get("feasible", True)))
        label_color = (0, 180, 0, 255) if ik_ok else (220, 0, 0, 255)
        radius = 5 if idx == 0 else 4
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 230, 0, 255), outline=(0, 0, 0, 255))
        if approach_end is not None:
            _draw_arrow(draw, center, approach_end, fill=(255, 140, 0, 255), width=3)
        if jaw_a is not None and jaw_b is not None:
            draw.line((*jaw_a, *jaw_b), fill=(0, 220, 255, 255), width=3)

        score = cand.get("score")
        score_txt = "" if score is None else f" {float(score):.2f}"
        label = f"{idx}{score_txt}"
        draw.text((u + 7, v - 12), label, fill=label_color)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out)
    return str(out)


def save_grasp_affordance_3d(
    path: str | Path,
    points: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int = 8,
    axis_scale_m: float = 0.055,
) -> str:
    """Save a 3D review plot for grasp affordance candidates.

    This complements the camera overlay when projection is ambiguous or candidates look
    detached from the object. Candidate dictionaries use the same compact convention as
    :func:`save_grasp_affordance_overlay`: ``pos`` and ``quat`` in world frame.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    pts = pts.reshape((-1, 3)) if pts.size else np.zeros((0, 3), dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        sample = pts
        if len(sample) > 2500:
            idx = np.linspace(0, len(sample) - 1, 2500).astype(int)
            sample = sample[idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=4, c="#8a8f98", alpha=0.35, label="target points")
    else:
        sample = np.zeros((0, 3), dtype=float)

    plotted: list[np.ndarray] = []
    for idx, cand in enumerate(candidates[:max_candidates]):
        pos = np.asarray(cand.get("pos", cand.get("position", [])), dtype=float).reshape(-1)
        quat = np.asarray(cand.get("quat", cand.get("quaternion", [])), dtype=float).reshape(-1)
        if pos.shape[0] < 3 or quat.shape[0] < 4:
            continue
        rot = _quat_wxyz_to_matrix(quat[:4])
        approach = rot[:, 2] * axis_scale_m
        jaw = rot[:, 0] * axis_scale_m * 0.5
        color = "#1f9d55" if bool(cand.get("ik_ok", cand.get("feasible", True))) else "#d92d20"
        ax.scatter([pos[0]], [pos[1]], [pos[2]], s=70 if idx == 0 else 42, c=color, marker="o")
        ax.quiver(pos[0], pos[1], pos[2], approach[0], approach[1], approach[2], color="#ff8c00", linewidth=2)
        ax.plot(
            [pos[0] - jaw[0], pos[0] + jaw[0]],
            [pos[1] - jaw[1], pos[1] + jaw[1]],
            [pos[2] - jaw[2], pos[2] + jaw[2]],
            color="#00c8ff",
            linewidth=2,
        )
        score = cand.get("score")
        label = f"{idx}" if score is None else f"{idx}:{float(score):.2f}"
        ax.text(pos[0], pos[1], pos[2], label, color=color)
        plotted.append(pos[:3])

    ax.set_title("grasp affordance candidates")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    all_points = sample
    if plotted:
        all_points = np.vstack([all_points, np.vstack(plotted)]) if len(all_points) else np.vstack(plotted)
    _set_equal_3d(ax, all_points if len(all_points) else np.zeros((1, 3)))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def _draw_mask_overlay(image: Image.Image, mask: np.ndarray, *, color: tuple[int, int, int, int]) -> None:
    if mask.shape[:2] != (image.height, image.width):
        return
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_arr = np.asarray(overlay).copy()
    overlay_arr[mask] = np.array(color, dtype=np.uint8)
    image.alpha_composite(Image.fromarray(overlay_arr, mode="RGBA"))


def _set_equal_3d(ax: Any, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    vec = np.array([ex - sx, ey - sy], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return
    direction = vec / norm
    normal = np.array([-direction[1], direction[0]])
    head = 8.0
    p1 = np.array([ex, ey]) - direction * head + normal * head * 0.45
    p2 = np.array([ex, ey]) - direction * head - normal * head * 0.45
    draw.polygon([(ex, ey), tuple(p1), tuple(p2)], fill=fill)


def save_video(path: str | Path, frames: list[np.ndarray], fps: int = 30) -> str | None:
    """把一串 RGB 帧写成 MP4;空帧序列直接跳过(返回 ``None``)。

    用 ``imageio`` 的 FFMPEG 后端写,与 :func:`clip_frames` 的读取后端对齐;逐 code block
    落盘动作视频时调用。
    """

    if not frames:
        return None
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, format="FFMPEG", codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(np.asarray(frame).astype(np.uint8)))
    return str(path)


def render_before_after(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """生成带标注的左右并排对比图,用于 VLM 评判。"""

    before = Image.fromarray(before_rgb.astype(np.uint8))
    after = Image.fromarray(after_rgb.astype(np.uint8))
    if after.size != before.size:
        after = after.resize(before.size)

    width, height = before.size
    label_h = 28
    canvas = Image.new("RGB", (width * 2 + 8, height + label_h), color=(255, 255, 255))
    canvas.paste(before, (0, label_h))
    canvas.paste(after, (width + 8, label_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "BEFORE", fill=(200, 0, 0))
    draw.text((width + 16, 6), "AFTER", fill=(0, 140, 0))
    return np.asarray(canvas)
