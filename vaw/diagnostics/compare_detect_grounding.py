"""Compare VAW ``detect_region`` grounding models on one frozen LIBERO scene.

The diagnostic runs the real VAW candidate-generation, candidate-review and
SAM3 segmentation path.  It never sends a robot command.  Only the VLM used by
``detect_region`` changes between conditions; the scene, query and SAM3 input
remain fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_goal_swap")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--query", default="stove knob")
    parser.add_argument(
        "--models",
        nargs="+",
        default=("vapi/gpt-5.5", "vapi/gemini-3.7-flash"),
        help="Grounding models to compare on the same frozen observation",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument(
        "--coord-space",
        choices=("auto", "pixel", "norm1000", "norm1000_yxyx"),
        default="auto",
        help="box coordinate protocol; auto resolves it independently per model",
    )
    parser.add_argument("--semantic-render-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "model"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _final_overlay(
    rgb: np.ndarray,
    *,
    mask: np.ndarray | None,
    box: list[float] | None,
    label: str,
) -> Image.Image:
    source = np.asarray(rgb, dtype=np.uint8)[..., :3]
    shown = source.astype(np.float32)
    if mask is not None and mask.shape == source.shape[:2]:
        amber = np.asarray([245.0, 158.0, 11.0], dtype=np.float32)
        shown[mask] = shown[mask] * 0.55 + amber * 0.45
    image = Image.fromarray(np.clip(shown, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    if box is not None and len(box) == 4:
        draw.rectangle(tuple(float(item) for item in box), outline="#00e5ff", width=4)
    draw.rectangle((0, 0, image.width, 30), fill="#07111f")
    draw.text((10, 8), label, fill="white")
    return image


def _comparison(images: list[tuple[str, Image.Image]]) -> Image.Image:
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    canvas = Image.new("RGB", (width * len(images), height), "#07111f")
    for index, (_label, image) in enumerate(images):
        canvas.paste(image, (index * width, 0))
    return canvas


def main() -> int:
    args = _parse_args()

    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.context_runtime.libero_sensor import make_libero_semantic_rgb_provider
    from vaw.context_runtime.semantic_grounding import (
        parse_candidates,
        render_candidate_review,
        resolve_grounding_coord_space,
    )
    from vaw.context_runtime.services import preflight_services
    from vaw.context_runtime.workspace import ContextWorkspace

    preflight_services()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path("vaw/out/grounding_comparisons")
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
    semantic_rgb_provider = make_libero_semantic_rgb_provider(
        env,
        scale=args.semantic_render_scale,
    )

    summaries: list[dict[str, Any]] = []
    comparison_images: list[tuple[str, Image.Image]] = []
    try:
        for model in args.models:
            condition_dir = output_dir / _slug(model)
            condition_dir.mkdir(parents=True, exist_ok=True)
            coord_space = resolve_grounding_coord_space(model, args.coord_space)
            api.configure_vlm_backend(
                model=model,
                server_url=args.server_url,
                api_key=args.api_key,
                coord_space=("norm1000" if coord_space == "norm1000_yxyx" else coord_space),
            )
            workspace = ContextWorkspace(
                api,
                task_prompt,
                motion_backend="pyroki",
                semantic_rgb_provider=semantic_rgb_provider,
                grounding_coord_space=coord_space,
            )
            policy_rgb = np.asarray(
                workspace._camera()["images"]["rgb"], dtype=np.uint8
            )[..., :3]
            semantic_rgb = semantic_rgb_provider()
            Image.fromarray(policy_rgb).save(condition_dir / "policy_input.png")
            Image.fromarray(semantic_rgb).save(condition_dir / "semantic_input.png")

            started = time.perf_counter()
            result = workspace.execute("detect_region", query=args.query)
            latency_s = time.perf_counter() - started
            diagnostics = (result.trace_diagnostics or {}).get("semantic_grounding", {})
            region_id = result.result.get("region_id") if result.ok else None
            mask = (
                workspace._private.region_masks.get(str(region_id))
                if region_id is not None
                else None
            )
            box = result.result.get("bbox_xyxy_px") if result.ok else None
            final = _final_overlay(
                policy_rgb,
                mask=mask,
                box=box,
                label=f"{model} | {'OK' if result.ok else 'ERROR'} | {latency_s:.1f}s",
            )
            final.save(condition_dir / "final_grounding.png")
            comparison_images.append((model, final))

            candidate_reply = diagnostics.get("candidate_reply")
            if isinstance(candidate_reply, str):
                try:
                    candidates = parse_candidates(
                        candidate_reply,
                        width=semantic_rgb.shape[1],
                        height=semantic_rgb.shape[0],
                        coord_space=str(diagnostics.get("coord_space", "pixel")),
                    )
                except (ValueError, TypeError):
                    candidates = []
                if candidates:
                    Image.fromarray(render_candidate_review(semantic_rgb, candidates)).save(
                        condition_dir / "candidate_review.png"
                    )

            summary = {
                "model": model,
                "coord_space": coord_space,
                "task": task_prompt,
                "query": args.query,
                "ok": result.ok,
                "latency_s": round(latency_s, 3),
                "result": result.result,
                "diagnostics": diagnostics,
            }
            summaries.append(summary)
            (condition_dir / "result.json").write_text(
                json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{model}] ok={result.ok} latency={latency_s:.1f}s "
                f"box={box} selected={diagnostics.get('selected_candidate')}"
            )

        _comparison(comparison_images).save(output_dir / "comparison.png")
        manifest = {
            "suite": args.suite,
            "task_id": args.task_id,
            "seed": args.seed,
            "task": task_prompt,
            "query": args.query,
            "semantic_render_scale": args.semantic_render_scale,
            "conditions": summaries,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
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
