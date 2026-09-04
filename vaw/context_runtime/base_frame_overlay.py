"""Project the robot base coordinate frame into the current Agentview RGB."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vaw.context_runtime.geometry import project_world_to_pixel

BASE_FRAME_AXIS_COLORS: dict[str, tuple[int, int, int]] = {
    "x": (239, 68, 68),
    "y": (22, 163, 74),
    "z": (37, 99, 235),
}

# The marker is anchored slightly above the mathematical base origin so it is
# visible on the Panda pedestal instead of being hidden by the table surface.
_BASE_FRAME_ORIGIN_XYZ = (0.0, 0.0, 0.08)
_BASE_FRAME_AXIS_LENGTH_M = 0.14


def draw_agentview_base_frame(
    image: np.ndarray,
    camera: dict[str, Any],
    *,
    origin_base_xyz: tuple[float, float, float] | np.ndarray = _BASE_FRAME_ORIGIN_XYZ,
    axis_length_m: float = _BASE_FRAME_AXIS_LENGTH_M,
) -> bool:
    """Draw calibrated ``+X/+Y/+Z`` arrows at the robot base.

    The directions are projected from metric base-frame points through the
    current Agentview calibration.  No fixed screen-space compass is used, so
    the cue remains correct if the camera pose changes.
    """

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    origin = np.asarray(origin_base_xyz, dtype=np.float64).reshape(3)
    length = float(axis_length_m)
    if not np.isfinite(origin).all() or not np.isfinite(length) or length <= 0:
        raise ValueError("base-frame origin and axis length must be finite")

    points = np.vstack(
        (
            origin,
            origin + np.array([length, 0.0, 0.0]),
            origin + np.array([0.0, length, 0.0]),
            origin + np.array([0.0, 0.0, length]),
        )
    )
    try:
        projected = project_world_to_pixel(
            points,
            camera["intrinsics"],
            camera["pose_mat"],
        )
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return False
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return False

    height, width = rgb.shape[:2]
    origin_px = projected[0, :2]
    if not _inside(origin_px, width, height, margin=8.0):
        return False

    canvas = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, int(round(min(width, height) / 150.0)))
    font = _font(max(13, int(round(min(width, height) / 32.0))))
    drawn_axes = 0
    for index, axis_name in enumerate(("x", "y", "z"), start=1):
        endpoint = _clip_endpoint(
            origin_px,
            projected[index, :2],
            width=width,
            height=height,
            margin=18.0,
        )
        if endpoint is None or float(np.linalg.norm(endpoint - origin_px)) < 10.0:
            continue
        color = BASE_FRAME_AXIS_COLORS[axis_name]
        _draw_arrow(
            draw,
            origin_px,
            endpoint,
            color,
            line_width=line_width,
        )
        _draw_label(
            draw,
            endpoint,
            f"+{axis_name.upper()}",
            color,
            font,
            width=width,
            height=height,
        )
        drawn_axes += 1

    if drawn_axes == 0:
        return False
    radius = line_width + 3
    x, y = (float(origin_px[0]), float(origin_px[1]))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 255, 255, 245),
        outline=(12, 18, 28, 245),
        width=2,
    )
    _draw_label(
        draw,
        origin_px + np.array([0.0, -(radius + 14.0)]),
        "BASE",
        (30, 41, 59),
        font,
        width=width,
        height=height,
    )
    composed = Image.alpha_composite(canvas, overlay).convert("RGB")
    image[...] = np.asarray(composed, dtype=np.uint8)
    return True


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: np.ndarray,
    endpoint: np.ndarray,
    color: tuple[int, int, int],
    *,
    line_width: int,
) -> None:
    start = (float(origin[0]), float(origin[1]))
    end = (float(endpoint[0]), float(endpoint[1]))
    draw.line((*start, *end), fill=(7, 12, 20, 235), width=line_width + 4)
    draw.line((*start, *end), fill=(*color, 255), width=line_width)
    direction = endpoint - origin
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    head_length = float(line_width * 3.5)
    head_width = float(line_width * 2.4)
    base = endpoint - direction * head_length
    triangle = [
        (float(endpoint[0]), float(endpoint[1])),
        (float(base[0] + normal[0] * head_width), float(base[1] + normal[1] * head_width)),
        (float(base[0] - normal[0] * head_width), float(base[1] - normal[1] * head_width)),
    ]
    draw.polygon(triangle, fill=(7, 12, 20, 235))
    inset = endpoint - direction * 1.5
    inner_base = endpoint - direction * (head_length - 2.0)
    inner_width = max(1.0, head_width - 2.0)
    draw.polygon(
        [
            (float(inset[0]), float(inset[1])),
            (
                float(inner_base[0] + normal[0] * inner_width),
                float(inner_base[1] + normal[1] * inner_width),
            ),
            (
                float(inner_base[0] - normal[0] * inner_width),
                float(inner_base[1] - normal[1] * inner_width),
            ),
        ],
        fill=(*color, 255),
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    anchor: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    *,
    width: int,
    height: int,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    box_width = bounds[2] - bounds[0] + 10
    box_height = bounds[3] - bounds[1] + 7
    left = min(max(3.0, float(anchor[0]) - box_width / 2), width - box_width - 3.0)
    top = min(max(3.0, float(anchor[1]) - box_height / 2), height - box_height - 3.0)
    draw.rounded_rectangle(
        (left, top, left + box_width, top + box_height),
        radius=4,
        fill=(8, 14, 24, 220),
        outline=(*color, 255),
        width=2,
    )
    draw.text(
        (left + 5, top + 3 - bounds[1]),
        text,
        font=font,
        fill=(*color, 255),
    )


def _clip_endpoint(
    origin: np.ndarray,
    endpoint: np.ndarray,
    *,
    width: int,
    height: int,
    margin: float,
) -> np.ndarray | None:
    direction = np.asarray(endpoint, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    if not np.isfinite(direction).all() or float(np.linalg.norm(direction)) < 1e-9:
        return None
    scale = 1.0
    for coordinate, delta, lower, upper in (
        (origin[0], direction[0], margin, width - margin),
        (origin[1], direction[1], margin, height - margin),
    ):
        if abs(float(delta)) < 1e-9:
            continue
        boundary = upper if delta > 0 else lower
        scale = min(scale, float((boundary - coordinate) / delta))
    if not math.isfinite(scale) or scale <= 0:
        return None
    return np.asarray(origin, dtype=np.float64) + direction * min(1.0, scale)


def _inside(pixel: np.ndarray, width: int, height: int, *, margin: float) -> bool:
    x, y = (float(pixel[0]), float(pixel[1]))
    return margin <= x < width - margin and margin <= y < height - margin


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


__all__ = ["BASE_FRAME_AXIS_COLORS", "draw_agentview_base_frame"]
