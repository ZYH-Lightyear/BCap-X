"""Qwen (and other general-purpose VLM) adapter for visual pointing.

This module provides a drop-in replacement for :func:`capx.integrations.vision.molmo.init_molmo`
that targets generic VLMs served via the local OpenRouter proxy (``openrouter_server.py``).

The returned callable has the same signature and return type as the Molmo one, so it
can be swapped in wherever ``init_molmo()`` was used. The bridge between Molmo's
specialised ``<point>`` markup and a general VLM is handled here by:

1. A pointing-oriented prompt that asks the VLM to output a JSON array such as
   ``[{"point_2d": [x, y], "label": "<obj>"}]``. This is the format Qwen2.5-VL /
   Qwen3-VL are trained to emit and is also easy for other VLMs to imitate.
2. A dedicated JSON parser (:func:`_parse_qwen_json_points`) that tolerates markdown
   fences, surrounding prose and truncated arrays, supports both ``point_2d`` (points)
   and ``bbox_2d`` (bounding boxes - we take the centre), and assumes the canonical
   Qwen ``0-1000`` normalised coordinate convention.
3. A fallback to :func:`capx.integrations.vision.molmo._parse_points` for the rare case
   that the VLM responds with the Molmo-style ``<point x=".." y=".."/>`` markup.

Switch the active pointing backend at runtime via the ``CAPX_POINT_BACKEND`` env var
read by the Franka LIBERO API classes.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable

import PIL
import requests

from capx.integrations.vision.molmo import (
    _image_to_data_url,
    _parse_points as _parse_molmo_points,
    convert_to_pixel_coordinates,
)


SERVICE_URL = "http://127.0.0.1:8110"  # capx.serving.openrouter_server default
QWEN_NORM_SCALE = 1000.0  # Qwen2.5-VL / Qwen3-VL grounding convention


_POINT_PROMPT_TEMPLATE = (
    "Locate the {obj} in the image.\n"
    "\n"
    "Respond with a single JSON array and NOTHING else (no prose, no markdown fence):\n"
    '[{{"point_2d": [x, y], "label": "{obj}"}}]\n'
    "\n"
    "Coordinate convention (MANDATORY):\n"
    "- Coordinates are NORMALISED to the range 0 - 1000.\n"
    "- x = 0 is the LEFT edge of the image, x = 1000 is the RIGHT edge.\n"
    "- y = 0 is the TOP edge of the image, y = 1000 is the BOTTOM edge.\n"
    "- Aim for the visual centre of the {obj}.\n"
    "\n"
    "If the {obj} is not visible in the image, respond with: []"
)


# Match a fenced JSON code block: ```json ... ``` or ``` ... ```
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
# Match a JSON array containing at least one object, anywhere in the text.
_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _build_chat_url(base_url: str) -> str:
    """Resolve the chat completions endpoint for the configured base URL."""
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _extract_json_text(text: str) -> str:
    """Best-effort isolation of the JSON payload from a free-form VLM response."""
    if not text:
        return ""
    # 1) Prefer an explicit ```json ... ``` fenced block.
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2) Otherwise look for the first JSON array literal in the text.
    m = _JSON_ARRAY_RE.search(text)
    if m:
        return m.group(0).strip()
    # 3) Last resort: return the raw text and let json.loads try.
    return text.strip()


def _try_load_json(text: str):
    """Tolerant ``json.loads``: trims trailing junk after the last ``}]`` on failure."""
    try:
        return json.loads(text)
    except Exception:
        idx = text.rfind("}]")
        if idx == -1:
            return None
        try:
            return json.loads(text[: idx + 2])
        except Exception:
            return None


def _parse_qwen_json_points(text: str) -> list[tuple[float, float]]:
    """Parse Qwen-style grounding JSON into a list of (x, y) coordinates.

    The returned coordinates are in the 0-1000 normalised scale that Qwen-VL emits
    natively. Use :data:`QWEN_NORM_SCALE` when converting them to pixels.

    Supports:
    - ``{"point_2d": [x, y], ...}``
    - ``{"bbox_2d": [x1, y1, x2, y2], ...}`` (returns the centre of the box)
    Tolerates markdown fences, surrounding prose, and trailing junk.
    """
    payload = _try_load_json(_extract_json_text(text))
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    points: list[tuple[float, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "point_2d" in item:
            xy = item["point_2d"]
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                try:
                    points.append((float(xy[0]), float(xy[1])))
                except (TypeError, ValueError):
                    continue
        elif "bbox_2d" in item:
            bb = item["bbox_2d"]
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                try:
                    cx = (float(bb[0]) + float(bb[2])) / 2.0
                    cy = (float(bb[1]) + float(bb[3])) / 2.0
                    points.append((cx, cy))
                except (TypeError, ValueError):
                    continue
    return points


def _parse_points_any_format(text: str) -> tuple[list[tuple[float, float]], float]:
    """Try Qwen JSON first, fall back to Molmo-style tags.

    Returns:
        (points, norm_scale) tuple where ``norm_scale`` matches the format that won
        (1000.0 for Qwen JSON, 100.0 or 1000.0 for Molmo formats).
    """
    qwen_pts = _parse_qwen_json_points(text)
    if qwen_pts:
        return qwen_pts, QWEN_NORM_SCALE
    # Fallback to whatever Molmo's parser can salvage.
    return _parse_molmo_points(text or "")


def init_qwen_vlm_point(
    model_name: str = "openrouter/qwen/qwen3.6-plus",
    base_url: str = SERVICE_URL,
    api_key: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 256,
    request_timeout: float = 120.0,
    max_retries: int = 3,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Return a Molmo-compatible pointing callable backed by a generic VLM.

    Args:
        model_name: Model identifier accepted by the OpenRouter proxy
            (e.g. ``"openrouter/qwen/qwen3.6-plus"``).
        base_url: Base URL of the OpenAI-compatible endpoint. Defaults to the local
            ``openrouter_server`` on port 8110.
        api_key: Optional API key. The local proxy normally does not require one.
        temperature: Sampling temperature. Pointing benefits from deterministic output.
        max_tokens: Cap on generated tokens; the expected response is a single short
            JSON array.
        request_timeout: Per-request HTTP timeout (seconds).
        max_retries: Number of attempts before giving up on a query.

    Returns:
        A callable with the same signature as the one returned by
        :func:`capx.integrations.vision.molmo.init_molmo`: it takes an image and a
        list of object descriptions and returns ``{obj: (x_px, y_px)}``, with
        ``(None, None)`` when parsing fails or the object cannot be located.
    """
    chat_url = _build_chat_url(base_url)
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def det_fn(
        image: PIL.Image.Image, objects: list[str] | None = None
    ) -> dict[str, tuple[int | None, int | None]]:
        if not objects:
            return {}

        img_url = _image_to_data_url(image)
        all_points: dict[str, tuple[int | None, int | None]] = {}

        for obj in objects:
            prompt = _POINT_PROMPT_TEMPLATE.format(obj=obj)
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            generated_text = ""
            backoff = 1.0
            for attempt in range(max_retries):
                try:
                    resp = session.post(
                        chat_url, json=payload, headers=headers, timeout=request_timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    generated_text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        or ""
                    )
                    print(f"[qwen-vlm-point] '{obj}' -> {generated_text!r}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[qwen-vlm-point] request for '{obj}' failed "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2 ** attempt))

            points, norm_scale = _parse_points_any_format(generated_text)
            if points:
                px = convert_to_pixel_coordinates(points, image, norm_scale)[0]
                abs_coords: tuple[int | None, int | None] = px
                print(
                    f"[qwen-vlm-point] '{obj}' parsed point "
                    f"(norm_scale={norm_scale:.0f}) -> pixel {abs_coords}"
                )
            else:
                abs_coords = (None, None)
                if generated_text:
                    print(
                        f"[qwen-vlm-point] no valid point parsed for '{obj}'; "
                        "returning (None, None)."
                    )
            all_points[obj] = abs_coords

        return all_points

    return det_fn


__all__ = [
    "QWEN_NORM_SCALE",
    "SERVICE_URL",
    "init_qwen_vlm_point",
]
