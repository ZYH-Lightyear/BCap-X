from __future__ import annotations

import base64
import io, os
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from capx.utils.serve_utils import post_with_retries

"""SAM 3 integration via FastAPI service.

Supports:
  1) Local CapX SAM3 server (``/segment`` with ``image_base64`` + ``results``)
  2) Remote multi-engine SAM3 (``image`` + ``detections`` with PNG masks)
"""

# Configuration (overridable via env)
SERVICE_URL = os.environ.get("SAM3_SERVICE_URL", "http://127.0.0.1:8114").rstrip("/")

def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        image_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        pil_image = Image.fromarray(image_u8).convert("RGB")
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    # Using np.frombuffer directly on the decoded bytes
    return np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)


def _decode_mask_flexible(mask_b64: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Decode CapX raw-bytes masks or remote PNG/L masks to a boolean HxW array."""
    if not mask_b64:
        raise ValueError("empty mask")
    raw = base64.b64decode(mask_b64)
    # PNG / JPEG encoded mask (remote multi-engine service)
    if raw[:8] == b"\x89PNG\r\n\x1a\n" or raw[:2] == b"\xff\xd8":
        arr = np.array(Image.open(io.BytesIO(raw)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr > 0
    if shape is None:
        raise ValueError("raw mask requires shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(bool)


def _normalize_segment_response(resp: dict[str, Any], default_label: str = "") -> list[dict[str, Any]]:
    """Normalize local CapX or remote multi-engine responses into CapX result dicts."""
    results: list[dict[str, Any]] = []

    # Local CapX format
    if isinstance(resp.get("results"), list):
        for item in resp["results"]:
            shape = tuple(item["shape"]) if "shape" in item else None
            mask = _decode_mask_flexible(item.get("mask_base64") or item.get("mask", ""), shape)
            results.append(
                {
                    "mask": mask,
                    "box": item.get("box") or item.get("bbox"),
                    "score": float(item.get("score", 0.0)),
                    "label": item.get("label", default_label),
                }
            )
        return results

    # Remote multi-engine format
    if isinstance(resp.get("detections"), list):
        for item in resp["detections"]:
            mask = _decode_mask_flexible(item.get("mask", ""))
            box = item.get("bbox") or item.get("box")
            results.append(
                {
                    "mask": mask,
                    "box": box,
                    "score": float(item.get("score", 0.0)),
                    "label": item.get("prompt") or item.get("label") or default_label,
                }
            )
        return results

    return results


def _use_remote_schema() -> bool:
    """Prefer remote multi-engine request schema when pointing off-localhost."""
    host = SERVICE_URL.split("://", 1)[-1].split("/", 1)[0]
    return not (host.startswith("127.0.0.1") or host.startswith("localhost"))


def _post_segment(payload_local: dict[str, Any], payload_remote: dict[str, Any]) -> dict[str, Any]:
    """POST ``/segment`` using CapX or remote multi-engine payload schema."""
    url = f"{SERVICE_URL}/segment"
    primary, secondary = (
        (payload_remote, payload_local) if _use_remote_schema() else (payload_local, payload_remote)
    )
    try:
        resp = post_with_retries(url, primary, max_retries=2, timeout_seconds=90.0)
        if isinstance(resp, dict) and resp.get("success") is False and "error" in resp:
            raise RuntimeError(resp.get("error"))
        return resp
    except Exception as first_err:
        try:
            resp = post_with_retries(url, secondary, max_retries=2, timeout_seconds=90.0)
            if isinstance(resp, dict) and resp.get("success") is False and "error" in resp:
                raise RuntimeError(resp.get("error"))
            return resp
        except Exception as second_err:
            raise RuntimeError(
                f"SAM3 segment failed on {url}. "
                f"primary_err={first_err}; secondary_err={second_err}"
            ) from second_err


def init_sam3(
    checkpoint_path: str | None = None,
    device: str = "cuda",
    model_type: str = "vit_l",
) -> Any:
    """Initialize SAM3 segmentation client.

    Returns a callable `segment_fn(image: np.ndarray | Image.Image, text_prompt: str) -> list[dict]`.

    Note: checkpoint_path, device, and model_type are ignored in client mode.
    """

    def segment_fn(
        image: np.ndarray | Image.Image,
        text_prompt: str,
        box_prompt: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run SAM3 inference with a text prompt via service.
        """
        encoded_image = _encode_image(image)
        payload_local: dict[str, Any] = {
            "image_base64": encoded_image,
            "text_prompt": text_prompt,
        }
        payload_remote: dict[str, Any] = {
            "image": encoded_image,
            "text_prompt": text_prompt,
        }
        if box_prompt is not None:
            payload_remote["box_prompts"] = [list(box_prompt)]

        try:
            resp = _post_segment(payload_local, payload_remote)
            if isinstance(resp, dict) and resp.get("success") is False:
                print(f"SAM3 remote error: {resp.get('error')}")
                return []
            results = _normalize_segment_response(resp, default_label=text_prompt)
        except Exception as e:
            print(f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}")
            return []

        if not results:
            print(f"SAM3 returned no results for prompt: '{text_prompt}'")
            return []

        return results

    return segment_fn


def visualize_sam3_results(
    image: Image.Image,
    prompt: str,
    results: list[dict[str, Any]],
    output_dir: pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Visualize SAM3 masks and boxes on the image. Adapted from visualize_sam3.py"""
    if not results:
        print(f"No results found for prompt: '{prompt}'")
        return

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup plot: Original + Individual detections
    # Limit to top 3 results to avoid overcrowding
    top_results = results[:3]

    fig, axes = plt.subplots(1, len(top_results) + 1, figsize=(4 * (len(top_results) + 1), 4))
    if len(top_results) == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes]) if not hasattr(axes, "__len__") else axes

    # Column 1: Original Image with all boxes
    ax_main = axes[0]
    ax_main.imshow(image)
    ax_main.set_title(f"Prompt: '{prompt}'")
    ax_main.axis("off")

    # Draw all boxes on main image
    for res in top_results:
        box = res["box"]
        score = res["score"]
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax_main.add_patch(rect)
        ax_main.text(x1, y1, f"{score:.2f}", color="white", fontsize=8, backgroundcolor="red")

    # Subsequent columns: Individual Mask + Box
    image_np = np.array(image)

    for idx, res in enumerate(top_results, start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        mask = res["mask"]
        box = res["box"]
        score = res["score"]

        # Overlay mask
        overlay = image_np.copy()
        color_mask = np.array([30, 144, 255], dtype=np.uint8)  # Dodger Blue

        # mask is boolean (H, W)
        if mask.shape[:2] == overlay.shape[:2]:
            overlay[mask] = overlay[mask] * 0.5 + color_mask * 0.5

        ax.imshow(overlay)

        # Draw box
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)

        ax.set_title(f"Score: {score:.2f}")
        ax.axis("off")

        # Save individual mask if needed
        if output_dir:
            mask_path = output_dir / f"mask_{prompt.replace(' ', '_')}_{idx}_{score:.2f}.png"
            overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8, copy=False)
            overlay_img = Image.fromarray(overlay_u8, mode="RGB")
            overlay_img.save(str(mask_path))

    plt.tight_layout()

    if output_dir:
        grid_path = output_dir / f"sam3_{prompt.replace(' ', '_')}.png"
        plt.savefig(str(grid_path), format="png")
        print(f"Saved visualization to: {grid_path}")

    if show:
        plt.show()

    plt.close()


def init_sam3_point_prompt(
    device: str = "cuda",
) -> Any:
    """Initialize SAM3 point prompt client.

    Returns a callable:
        point_prompt_fn(image: np.ndarray | Image.Image,
                        point_coords: tuple[float, float])
            -> list[dict] with keys mask/score (and box/label when available)
    """

    def point_prompt_fn(
        image: np.ndarray | Image.Image, point_coords: tuple[float, float]
    ) -> list[dict[str, Any]]:
        encoded_image = _encode_image(image)
        x, y = float(point_coords[0]), float(point_coords[1])

        # Prefer CapX local ``/segment_point``; fall back to remote ``/segment`` with points.
        try:
            resp = post_with_retries(
                f"{SERVICE_URL}/segment_point",
                {"image_base64": encoded_image, "point_coords": [x, y]},
                max_retries=2,
                timeout_seconds=60.0,
            )
            scores = resp.get("scores", [])
            masks_shape = tuple(resp.get("masks_shape", (0, 0, 0)))
            masks_dtype = np.dtype(resp.get("masks_dtype", "float32"))
            masks = _decode_mask(resp.get("masks_base64", ""), masks_shape, dtype=masks_dtype)
            masks = masks.astype(bool)
            return [{"mask": mask, "score": float(score)} for mask, score in zip(masks, scores)]
        except Exception:
            payload_remote = {
                "image": encoded_image,
                "point_prompts": [[x, y]],
                "point_labels": [1],
            }
            try:
                resp = post_with_retries(
                    f"{SERVICE_URL}/segment",
                    payload_remote,
                    max_retries=3,
                    timeout_seconds=120.0,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}"
                ) from e
            if isinstance(resp, dict) and resp.get("success") is False:
                raise RuntimeError(f"SAM3 remote error: {resp.get('error')}")
            return _normalize_segment_response(resp)

    return point_prompt_fn
