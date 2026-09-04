"""Private semantic disambiguation for VAW region grounding.

The CaP-X reduced detector is intentionally a single-box API.  VAW scenes can
contain multiple visually similar objects, so forcing one answer directly can
turn an uncertain category match into authoritative-looking evidence.  This
module keeps that uncertainty inside the perception boundary: generate a small
candidate set, enlarge every crop, and require an independent visual choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GroundingCandidate:
    box_xyxy_px: tuple[float, float, float, float]
    evidence: str


def resolve_grounding_coord_space(model: str, requested: str = "auto") -> str:
    """Resolve the coordinate protocol used by one grounding VLM adapter.

    Gemini and Qwen vision models conventionally emit boxes in a 0--1000
    coordinate space even when the image has a different pixel resolution;
    Gemini additionally uses ``[y1,x1,y2,x2]`` ordering. Keeping those adapter
    facts explicit prevents a correct box from being transposed or interpreted
    as an upper-left pixel box.
    """

    if requested in {"pixel", "norm1000", "norm1000_yxyx"}:
        return requested
    if requested != "auto":
        raise ValueError(
            "grounding coordinate space must be auto, pixel, norm1000 "
            "or norm1000_yxyx"
        )
    normalized_model = str(model).casefold()
    if "gemini" in normalized_model:
        return "norm1000_yxyx"
    if "qwen" in normalized_model:
        return "norm1000"
    return "pixel"


def candidate_prompt(
    query: str,
    *,
    width: int,
    height: int,
    coord_space: str,
) -> str:
    target = " ".join(str(query).split())
    if coord_space == "norm1000_yxyx":
        coordinate_instruction = "坐标使用 0 到 1000 的归一化数值，box 顺序为 [y1,x1,y2,x2]"
        box_schema = "y1,x1,y2,x2"
    elif coord_space == "norm1000":
        coordinate_instruction = "坐标使用 0 到 1000 的归一化数值，box 顺序为 [x1,y1,x2,y2]"
        box_schema = "x1,y1,x2,y2"
    else:
        coordinate_instruction = (
            f"坐标使用图像真实像素，图像宽度={width}、高度={height}，"
            "box 顺序为 [x1,y1,x2,y2]"
        )
        box_schema = "x1,y1,x2,y2"
    return (
        f"请为精确语义目标“{target}”找出最多三个彼此不同的合理候选。比较可见属性与场景关系；"
        "如果存在同类别竞争物体，也要保留为候选，不能直接选择最近或最大的物体。"
        "只返回 JSON 数组："
        f'[{{"box":[{box_schema}],"evidence":"可见依据"}}]。'
        f"{coordinate_instruction}；最可信候选排在最前面。没有可见候选时返回 []。"
    )


def review_prompt(query: str) -> str:
    target = " ".join(str(query).split())
    return (
        f"编号卡片是精确语义目标“{target}”的候选区域。图像同时包含完整场景与放大的候选裁剪图。"
        "请检查每张裁剪图的真实像素，并比较可见语义属性与场景关系。只选择一个完全匹配的候选，"
        "不能接受仅类别相同的普通物体。若目标无法辨认或候选仍有歧义，使用 null。"
        '只返回 JSON：{"candidate":1} 或 {"candidate":null}。'
    )


def parse_candidates(
    reply: str,
    *,
    width: int,
    height: int,
    coord_space: str,
    limit: int = 3,
) -> list[GroundingCandidate]:
    """Parse model-specific coordinates into clipped pixel candidates."""

    if coord_space not in {"pixel", "norm1000", "norm1000_yxyx"}:
        raise ValueError(
            "candidate coordinate space must be pixel, norm1000 or norm1000_yxyx"
        )

    payload = _json_payload(reply)
    if isinstance(payload, dict):
        payload = payload.get("candidates", [payload] if "box" in payload else [])
    if not isinstance(payload, list):
        raise ValueError("candidate response is not a JSON list")

    candidates: list[GroundingCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        values = item.get("box")
        try:
            box = np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if box.size != 4 or not np.isfinite(box).all():
            continue
        if coord_space == "norm1000_yxyx":
            y1, x1, y2, x2 = (float(value) for value in box)
        else:
            x1, y1, x2, y2 = (float(value) for value in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if coord_space in {"norm1000", "norm1000_yxyx"}:
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
        x1 = float(np.clip(x1, 0, width - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        pixel_box = (x1, y1, x2, y2)
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        if any(_box_iou(pixel_box, existing.box_xyxy_px) >= 0.9 for existing in candidates):
            continue
        candidates.append(
            GroundingCandidate(
                box_xyxy_px=pixel_box,
                evidence=" ".join(str(item.get("evidence", "")).split()),
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def render_candidate_review(
    rgb: np.ndarray,
    candidates: list[GroundingCandidate],
) -> np.ndarray:
    """Build one fixed full-scene plus enlarged-crop review raster."""

    source = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    scene = source.resize((800, 512), Image.Resampling.BILINEAR)
    scene_draw = ImageDraw.Draw(scene)
    colors = ("#ef3340", "#00a86b", "#1473e6")
    scale_x = 800.0 / source.width
    scale_y = 512.0 / source.height
    font = _font(22)
    label_font = _font(28)
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.box_xyxy_px
        shown = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        scene_draw.rectangle(shown, outline=colors[index], width=5)
        scene_draw.rectangle(
            (shown[0], shown[1], shown[0] + 42, shown[1] + 34),
            fill=colors[index],
        )
        scene_draw.text(
            (shown[0] + 12, shown[1] + 2),
            str(index + 1),
            fill="white",
            font=label_font,
        )

    review = Image.new("RGB", (1200, 720), "white")
    review.paste(scene, (0, 104))
    draw = ImageDraw.Draw(review)
    draw.text((20, 64), "完整场景", fill="#111827", font=font)
    card_height = 720 // max(1, len(candidates))
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.box_xyxy_px
        pad_x = max(8.0, (x2 - x1) * 0.35)
        pad_y = max(8.0, (y2 - y1) * 0.25)
        crop = source.crop(
            (
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(source.width), x2 + pad_x),
                min(float(source.height), y2 + pad_y),
            )
        )
        crop.thumbnail((380, card_height - 56), Image.Resampling.NEAREST)
        top = index * card_height
        draw.rectangle((800, top, 1199, top + card_height - 1), outline=colors[index], width=5)
        draw.text(
            (816, top + 10),
            f"候选 {index + 1}",
            fill=colors[index],
            font=font,
        )
        review.paste(crop, (800 + (400 - crop.width) // 2, top + 48))
    return np.asarray(review, dtype=np.uint8)


def parse_choice(reply: str, *, count: int) -> int | None:
    """Return a zero-based candidate index, or None for explicit ambiguity."""

    payload = _json_payload(reply)
    if not isinstance(payload, dict) or "candidate" not in payload:
        raise ValueError("review response must contain candidate")
    choice: Any = payload["candidate"]
    if choice is None:
        return None
    if isinstance(choice, bool) or not isinstance(choice, (int, float)):
        raise ValueError("review candidate must be an integer or null")
    index = int(choice)
    if float(choice) != float(index) or not 1 <= index <= count:
        raise ValueError(f"review candidate must be within [1, {count}]")
    return index - 1


def _json_payload(reply: str) -> Any:
    text = str(reply or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    starts = sorted(index for index, char in enumerate(text) if char in "[{")
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("response contains no valid JSON value")


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


__all__ = [
    "GroundingCandidate",
    "candidate_prompt",
    "parse_candidates",
    "parse_choice",
    "render_candidate_review",
    "resolve_grounding_coord_space",
    "review_prompt",
]
