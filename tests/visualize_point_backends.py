"""Visualize and compare Qwen vs Molmo visual pointing on the same image.

Both backends ultimately return pixel coordinates via the shared helper
``convert_to_pixel_coordinates(points, image, norm_scale)``. The norm_scale must
match the model output convention:

- Molmo2 / Qwen JSON: coordinates in [0, 1000]  -> norm_scale=1000
- Molmo1 tags:       coordinates in [0, 100]   -> norm_scale=100

This script prints raw model text, parsed normalised coords, pixel coords, and
saves an overlay PNG with crosshairs + labels.

Usage:
    uv run --no-sync --active tests/visualize_point_backends.py \\
        --image-path outputs/.../visual_feedback_00.png \\
        --objects "alphabet soup,basket" \\
        --out-dir outputs/point_debug

    # Include Molmo if the vLLM server on 8122 is up:
    uv run --no-sync --active tests/visualize_point_backends.py \\
        --image-path ... --objects "alphabet soup" --molmo-url http://127.0.0.1:8122/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Repo root on sys.path when run as a script
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass
class PointResult:
    backend: str
    object_name: str
    raw_text: str
    norm_points: list[tuple[float, float]]
    norm_scale: float
    pixel: tuple[int | None, int | None]

    @property
    def in_range_ok(self) -> bool:
        if not self.norm_points:
            return self.pixel == (None, None)
        lo, hi = 0.0, self.norm_scale
        for x, y in self.norm_points:
            if not (lo <= x <= hi and lo <= y <= hi):
                return False
        return True


def _query_qwen(
    image: Image.Image,
    objects: list[str],
    *,
    model_name: str,
    base_url: str,
) -> list[PointResult]:
    from capx.integrations.vision.qwen_vlm_point import (
        QWEN_NORM_SCALE,
        _build_chat_url,
        _parse_points_any_format,
        init_qwen_vlm_point,
    )
    from capx.integrations.vision.molmo import (
        _image_to_data_url,
        convert_to_pixel_coordinates,
    )
    import requests
    import time

    chat_url = _build_chat_url(base_url)
    session = requests.Session()
    headers = {"Content-Type": "application/json"}

    from capx.integrations.vision.qwen_vlm_point import _POINT_PROMPT_TEMPLATE

    results: list[PointResult] = []
    for obj in objects:
        prompt = _POINT_PROMPT_TEMPLATE.format(obj=obj)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        raw = ""
        for attempt in range(3):
            try:
                resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                raw = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
                break
            except Exception as e:
                if attempt == 2:
                    raw = f"<request failed: {e}>"
                else:
                    time.sleep(2**attempt)

        norm_pts, norm_scale = _parse_points_any_format(raw)
        if norm_pts:
            px = convert_to_pixel_coordinates(norm_pts, image, norm_scale)[0]
        else:
            px = (None, None)

        results.append(
            PointResult(
                backend="qwen",
                object_name=obj,
                raw_text=raw,
                norm_points=norm_pts,
                norm_scale=norm_scale if norm_pts else QWEN_NORM_SCALE,
                pixel=px,
            )
        )
    return results


def _query_molmo(
    image: Image.Image,
    objects: list[str],
    *,
    model_name: str,
    base_url: str,
) -> list[PointResult]:
    from capx.integrations.vision.molmo import (
        _image_to_data_url,
        _parse_points,
        convert_to_pixel_coordinates,
    )
    import requests
    import time

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()
    headers = {"Content-Type": "application/json"}

    results: list[PointResult] = []
    for obj in objects:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Point at {obj}"},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stop": ["<|endoftext|>"],
        }
        raw = ""
        for attempt in range(3):
            try:
                resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                raw = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )
                break
            except Exception as e:
                if attempt == 2:
                    raw = f"<request failed: {e}>"
                else:
                    time.sleep(2**attempt)

        norm_pts, norm_scale = _parse_points(raw)
        if norm_pts:
            px = convert_to_pixel_coordinates(norm_pts, image, norm_scale)[0]
        else:
            px = (None, None)

        results.append(
            PointResult(
                backend="molmo",
                object_name=obj,
                raw_text=raw,
                norm_points=norm_pts,
                norm_scale=norm_scale if norm_pts else 1000.0,
                pixel=px,
            )
        )
    return results


def _draw_overlay(
    image: Image.Image,
    results: list[PointResult],
    out_path: Path,
) -> None:
    """Draw all backend/object points on one image."""
    colors = {"qwen": "#FF3333", "molmo": "#3399FF"}
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    w, h = image.size
    radius = max(5, min(w, h) // 60)

    for r in results:
        if r.pixel[0] is None or r.pixel[1] is None:
            continue
        x, y = int(r.pixel[0]), int(r.pixel[1])
        color = colors.get(r.backend, "lime")
        # Crosshair
        arm = radius * 2
        draw.line((x - arm, y, x + arm, y), fill=color, width=2)
        draw.line((x, y - arm, x, y + arm), fill=color, width=2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        label = f"{r.backend}:{r.object_name}"
        if r.norm_points:
            nx, ny = r.norm_points[0]
            label += f" norm=({nx:.0f},{ny:.0f})/{r.norm_scale:.0f}"
        label += f" px=({x},{y})"
        draw.text((x + radius + 4, y - radius - 2), label, fill=color, font=font_sm)

    # Legend
    y0 = 8
    for backend, color in colors.items():
        if any(r.backend == backend for r in results):
            draw.rectangle((8, y0, 24, y0 + 14), fill=color)
            draw.text((30, y0), backend, fill="white", font=font, stroke_width=2, stroke_fill="black")
            y0 += 20

    draw.text((8, h - 22), f"{w}x{h}  |  norm: x/W*1000, y/H*1000 (Molmo2 & Qwen)", fill="white", font=font_sm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved overlay -> {out_path}")


def _print_report(results: list[PointResult], image: Image.Image) -> None:
    w, h = image.size
    print("\n" + "=" * 72)
    print(f"Image size: {w} x {h}")
    print("Pixel formula (shared with Molmo): px = int(norm_x / norm_scale * W)")
    print("=" * 72)
    for r in results:
        print(f"\n--- [{r.backend}] {r.object_name!r} ---")
        print(f"raw: {r.raw_text[:500]!r}{'...' if len(r.raw_text) > 500 else ''}")
        print(f"norm_scale: {r.norm_scale}")
        print(f"norm_points: {r.norm_points}")
        print(f"in_range [0, {r.norm_scale}]: {r.in_range_ok}")
        print(f"pixel (production API): {r.pixel}")
        if r.norm_points and r.pixel[0] is not None:
            nx, ny = r.norm_points[0]
            expected_x = int(nx / r.norm_scale * w)
            expected_y = int(ny / r.norm_scale * h)
            match = (expected_x, expected_y) == r.pixel
            print(f"recomputed pixel: ({expected_x}, {expected_y})  match={match}")
            print(f"percent of image: ({100*nx/r.norm_scale:.1f}%, {100*ny/r.norm_scale:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--objects", type=str, default="alphabet soup,basket")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/point_debug"))
    parser.add_argument("--qwen-model", type=str, default="openrouter/qwen/qwen3.6-plus")
    parser.add_argument("--qwen-url", type=str, default="http://127.0.0.1:8110")
    parser.add_argument("--molmo-model", type=str, default="allenai/Molmo2-8B")
    parser.add_argument(
        "--molmo-url",
        type=str,
        default=None,
        help="Molmo vLLM base URL (e.g. http://127.0.0.1:8122/v1). Omit to skip Molmo.",
    )
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"[ERROR] image not found: {args.image_path}", file=sys.stderr)
        return 1

    image = Image.open(args.image_path).convert("RGB")
    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    if not objects:
        print("[ERROR] no objects", file=sys.stderr)
        return 1

    all_results: list[PointResult] = []

    if not args.skip_qwen:
        print("Querying Qwen VLM pointer...")
        all_results.extend(
            _query_qwen(image, objects, model_name=args.qwen_model, base_url=args.qwen_url)
        )

    if args.molmo_url:
        print("Querying Molmo pointer...")
        try:
            all_results.extend(
                _query_molmo(
                    image, objects, model_name=args.molmo_model, base_url=args.molmo_url
                )
            )
        except Exception as e:
            print(f"[WARN] Molmo query failed: {e}")

    if not all_results:
        print("[ERROR] no backends queried", file=sys.stderr)
        return 1

    _print_report(all_results, image)

    stem = args.image_path.stem
    overlay_path = args.out_dir / f"{stem}_points_overlay.png"
    _draw_overlay(image, all_results, overlay_path)

    # Save JSON for later inspection
    json_path = args.out_dir / f"{stem}_points_report.json"
    report = [
        {
            "backend": r.backend,
            "object": r.object_name,
            "raw_text": r.raw_text,
            "norm_points": r.norm_points,
            "norm_scale": r.norm_scale,
            "pixel": list(r.pixel),
            "in_range_ok": r.in_range_ok,
        }
        for r in all_results
    ]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {json_path}")

    # Per-backend separate overlays (same style as molmo._overlay_and_save convention)
    for backend in {r.backend for r in all_results}:
        sub = [r for r in all_results if r.backend == backend]
        _draw_overlay(image, sub, args.out_dir / f"{stem}_{backend}_only.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
