"""Smoke test for the Qwen VLM pointing adapter.

Calls ``init_qwen_vlm_point()`` against the running OpenRouter proxy with a real
LIBERO ``visual_feedback`` frame and prints the parsed pixel coordinates plus an
optional overlay so you can sanity-check the result by eye.

Usage:
    CAPX_POINT_BACKEND=qwen uv run --no-sync --active tests/test_qwen_vlm_point.py \
        --image-path outputs/openrouter_qwen_qwen3.6-plus/franka_libero_cap_agent0_qwen_pointer/trial_01_sandboxrc_1_reward_0.000_taskcompleted_0/visual_feedback_00.png \
        --objects "alphabet soup,basket"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to an RGB image to query the pointer with.",
    )
    parser.add_argument(
        "--objects",
        type=str,
        default="alphabet soup,basket",
        help="Comma-separated object queries.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openrouter/qwen/qwen3.6-plus",
        help="Model identifier to send to the OpenRouter proxy.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8110",
        help="Base URL for the OpenRouter proxy (no trailing /chat/completions).",
    )
    parser.add_argument(
        "--save-overlay",
        type=Path,
        default=None,
        help="If set, save a visualisation of the detected points to this path.",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"[ERROR] image not found: {args.image_path}", file=sys.stderr)
        return 1

    image = Image.open(args.image_path).convert("RGB")
    print(f"Loaded image {args.image_path} ({image.size[0]}x{image.size[1]})")

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    if not objects:
        print("[ERROR] no objects specified.", file=sys.stderr)
        return 1
    print(f"Querying for: {objects}")

    from capx.integrations.vision.qwen_vlm_point import init_qwen_vlm_point

    det_fn = init_qwen_vlm_point(model_name=args.model_name, base_url=args.base_url)
    points = det_fn(image, objects=objects)

    print("\n=== Pointer results ===")
    for obj, xy in points.items():
        if xy[0] is None or xy[1] is None:
            print(f"  {obj!r}: NOT FOUND")
        else:
            x_pct = 100 * xy[0] / image.size[0]
            y_pct = 100 * xy[1] / image.size[1]
            print(f"  {obj!r}: (x={xy[0]}, y={xy[1]}) -> ({x_pct:.1f}%, {y_pct:.1f}%)")

    if args.save_overlay is not None:
        try:
            from PIL import ImageDraw, ImageFont
        except Exception as e:
            print(f"[WARN] cannot import PIL.ImageDraw for overlay: {e}", file=sys.stderr)
            return 0

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        radius = max(4, min(image.size) // 80)
        for obj, xy in points.items():
            if xy[0] is None or xy[1] is None:
                continue
            x, y = int(xy[0]), int(xy[1])
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=3,
            )
            draw.text((x + radius + 2, y - radius), obj, fill="yellow", font=font)
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(args.save_overlay)
        print(f"Saved overlay to {args.save_overlay}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
