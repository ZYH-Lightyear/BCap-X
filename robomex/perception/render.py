"""Evidence renderers: turn raw arrays into VLM-friendly images.

Rendered products are transient first-class inputs for the judge, not skill
assets. Phase 1 ships the before/after renderer (gate 3); mask/bbox/grasp
overlays for gate 1 can reuse CapX's debug-overlay drawing later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_rgb(path: str | Path, rgb: np.ndarray) -> str:
    """Save an (H, W, 3) uint8 array as PNG; returns the path as str."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return str(path)


def render_before_after(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """Side-by-side labeled comparison image for VLM judging."""

    before = Image.fromarray(before_rgb.astype(np.uint8))
    after = Image.fromarray(after_rgb.astype(np.uint8))
    if after.size != before.size:
        after = after.resize(before.size)

    width, height = before.size
    label_h = 28
    canvas = Image.new("RGB", (width * 2 + 8, height + label_h), color=(255, 255, 255))
    canvas.paste(before, (0, label_h))
    canvas.paste(after, (width + 8, label_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "BEFORE", fill=(200, 0, 0))
    draw.text((width + 16, 6), "AFTER", fill=(0, 140, 0))
    return np.asarray(canvas)
