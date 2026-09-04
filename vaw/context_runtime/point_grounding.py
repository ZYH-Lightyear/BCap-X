"""Model-adapter boundary for VLM point grounding."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def resolve_point_coord_space(model: str, requested: str = "auto") -> str:
    """Return the native point coordinate protocol for a VLM family."""

    supported = {"pixel_xy", "norm1000_xy", "norm1000_yx"}
    if requested in supported:
        return requested
    if requested != "auto":
        raise ValueError(
            "point coordinate space must be auto, pixel_xy, norm1000_xy "
            "or norm1000_yx"
        )
    normalized_model = str(model).casefold()
    if "gemini" in normalized_model:
        return "norm1000_yx"
    if "qwen" in normalized_model:
        return "norm1000_xy"
    return "pixel_xy"


def point_prompt(
    query: str,
    *,
    width: int,
    height: int,
    coord_space: str,
) -> str:
    """Build a compact, model-protocol-aware point request."""

    target = " ".join(str(query).split())
    if coord_space == "norm1000_yx":
        schema = "y,x"
        convention = "坐标归一化到 0–1000，顺序严格为 [y,x]"
    elif coord_space == "norm1000_xy":
        schema = "x,y"
        convention = "坐标归一化到 0–1000，顺序严格为 [x,y]"
    elif coord_space == "pixel_xy":
        schema = "x,y"
        convention = f"使用真实像素坐标，图像宽={width}、高={height}，顺序严格为 [x,y]"
    else:
        raise ValueError(f"unsupported point coordinate space: {coord_space}")
    return (
        f"请在图像中定位“{target}”的精确操作点。点必须落在目标可见像素内部，"
        "不要返回目标外的近似位置。"
        f"{convention}。只返回 JSON：{{\"point\":[{schema}]}}；目标不可见时返回 "
        '{"point":null}。'
    )


def parse_point_reply(
    reply: str,
    *,
    width: int,
    height: int,
    coord_space: str,
) -> tuple[float, float] | None:
    """Parse one model-native point into absolute ``(x, y)`` pixels."""

    payload = _json_payload(reply)
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return None
    values = payload.get("point", payload.get("point_2d"))
    if values is None:
        return None
    try:
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if raw.size < 2 or not np.isfinite(raw[:2]).all():
        return None
    if coord_space == "norm1000_yx":
        y, x = float(raw[0]), float(raw[1])
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    elif coord_space == "norm1000_xy":
        x, y = float(raw[0]), float(raw[1])
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    elif coord_space == "pixel_xy":
        x, y = float(raw[0]), float(raw[1])
    else:
        raise ValueError(f"unsupported point coordinate space: {coord_space}")
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None
    return x, y


def _json_payload(reply: str) -> Any:
    text = str(reply or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    for start in sorted(index for index, char in enumerate(text) if char in "[{"):
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("point response contains no valid JSON value")


__all__ = [
    "parse_point_reply",
    "point_prompt",
    "resolve_point_coord_space",
]
