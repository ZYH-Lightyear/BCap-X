"""证据渲染器:把原始数组变成对 VLM 友好的图像。

渲染产物是 judge 的临时一等输入,而非技能资产。Phase 1 提供 before/after 渲染器
(gate 3);gate 1 的 mask/bbox/grasp 叠加图以后可复用 CapX 的 debug 叠加绘制。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_rgb(path: str | Path, rgb: np.ndarray) -> str:
    """把一个 (H, W, 3) 的 uint8 数组存成 PNG;返回路径字符串。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return str(path)


def render_before_after(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """生成带标注的左右并排对比图,用于 VLM 评判。"""

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
