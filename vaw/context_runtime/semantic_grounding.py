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


def candidate_prompt(
    query: str,
    *,
    width: int,
    height: int,
    coord_space: str,
) -> str:
    target = " ".join(str(query).split())
    coordinate_instruction = (
        "using 0-1000 normalized coordinates"
        if coord_space == "norm1000"
        else f"using real pixel coordinates for a width={width}, height={height} image"
    )
    return (
        f"Find up to three distinct plausible candidates for the exact semantic target "
        f"'{target}'. Compare visible attributes and relations; include competing "
        "same-category objects instead of silently choosing the nearest or largest one. "
        "Reply only as a JSON array "
        '[{"box":[x1,y1,x2,y2],"evidence":"visible cue"}] '
        f"{coordinate_instruction}, best candidate first. Return [] if none are visible."
    )


def review_prompt(query: str) -> str:
    target = " ".join(str(query).split())
    return (
        f"The numbered cards are candidate regions for the exact semantic target "
        f"'{target}'. The image contains the full scene and enlarged candidate crops. "
        "Inspect the actual pixels in every crop and compare visible semantic attributes "
        "and scene relations. Choose the single candidate that exactly matches; do not "
        "accept a generic same-category object. If the target is not identifiable or the "
        'candidates remain ambiguous, use null. Reply only JSON {"candidate":1} or '
        '{"candidate":null}.'
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

    if coord_space not in {"pixel", "norm1000"}:
        raise ValueError("candidate coordinate space must be pixel or norm1000")

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
        x1, x2 = sorted((float(box[0]), float(box[2])))
        y1, y2 = sorted((float(box[1]), float(box[3])))
        if coord_space == "norm1000":
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
    draw.text((20, 64), "FULL SCENE", fill="#111827", font=font)
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
            f"CANDIDATE {index + 1}",
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
    "review_prompt",
]
