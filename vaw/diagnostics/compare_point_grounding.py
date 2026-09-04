"""Compare point-grounding VLMs on one frozen LIBERO observation.

This diagnostic performs no robot command. It gives every model the same
policy-visible RGB-D observation and query, then runs the real VAW
``locate_point`` path, including pixel parsing and depth lifting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--query", default="center of the alphabet soup can")
    parser.add_argument(
        "--models",
        nargs="+",
        default=("vapi/gpt-5.5", "vapi/gemini-3.7-flash"),
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument(
        "--coord-space",
        choices=("auto", "pixel_xy", "norm1000_xy", "norm1000_yx"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def _overlay(rgb: np.ndarray, point: tuple[float, float] | None, label: str) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3], mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill="#07111f")
    draw.text((10, 8), label, fill="white")
    if point is not None:
        x, y = point
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ffd400")
        draw.line((x - 22, y, x + 22, y), fill="#111827", width=3)
        draw.line((x, y - 22, x, y + 22), fill="#111827", width=3)
    return image


def main() -> int:
    args = _parse_args()

    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.context_runtime.point_grounding import resolve_point_coord_space
    from vaw.context_runtime.services import preflight_services
    from vaw.context_runtime.workspace import ContextWorkspace

    preflight_services()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path("vaw/out/point_grounding_comparisons")
        / f"{args.suite}_t{args.task_id}_s{args.seed}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = FrankaLiberoTask(
        suite_name=args.suite,
        task_id=args.task_id,
        privileged=False,
        seed=args.seed,
    )
    _, reset_info = env.reset(seed=args.seed)
    task_prompt = str(reset_info["task_prompt"])
    api = FrankaLiberoApiReduced(env)
    api.configure_vlm_backend(server_url=args.server_url, api_key=args.api_key)
    rgb = np.asarray(api.get_observation()[api.camera_name]["images"]["rgb"], dtype=np.uint8)
    height, width = rgb.shape[:2]

    summaries: list[dict[str, object]] = []
    overlays: list[Image.Image] = []
    try:
        for model in args.models:
            condition_dir = output_dir / _slug(model)
            condition_dir.mkdir(parents=True, exist_ok=True)
            coord_space = resolve_point_coord_space(model, args.coord_space)
            workspace = ContextWorkspace(
                api,
                task_prompt,
                motion_backend="pyroki",
                point_model=model,
                point_coord_space=coord_space,
            )
            started = time.perf_counter()
            result = workspace.execute("locate_point", query=args.query)
            latency_s = time.perf_counter() - started
            diagnostics = (result.trace_diagnostics or {}).get("point_grounding", {})
            point_values = result.result.get("pixel_xy") if result.ok else None
            point = (
                (float(point_values[0]), float(point_values[1]))
                if isinstance(point_values, list) and len(point_values) == 2
                else None
            )
            shown = _overlay(
                rgb,
                point,
                f"{model} | {'OK' if point else 'ERROR'} | {latency_s:.1f}s",
            )
            shown.save(condition_dir / "point_overlay.png")
            Image.fromarray(rgb).save(condition_dir / "input.png")
            summary = {
                "model": model,
                "coord_space": coord_space,
                "latency_s": round(latency_s, 3),
                "ok": result.ok,
                "point_xy_px": list(point) if point is not None else None,
                "position_xyz": result.result.get("position_xyz"),
                "raw_reply": diagnostics.get("raw_reply"),
                "error": result.result.get("error"),
            }
            summaries.append(summary)
            overlays.append(shown)
            (condition_dir / "result.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        comparison = Image.new("RGB", (width * len(overlays), height), "#07111f")
        for index, shown in enumerate(overlays):
            comparison.paste(shown, (index * width, 0))
        comparison.save(output_dir / "comparison.png")
        manifest = {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "task": task_prompt,
            "query": args.query,
            "conditions": summaries,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[output] {output_dir}")
        return 0 if all(item["ok"] for item in summaries) else 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
