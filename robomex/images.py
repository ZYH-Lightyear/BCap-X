"""图像落盘与图像消息编码。

环境这一端负责把仿真器渲染出的数组写成 PNG(:func:`save_rgb`),planner 这一端
负责把 PNG 变成模型能吃的 image part(:func:`image_content_part`)。两个方向都很
薄,但刻意放在同一个模块里:它们共用同一套"图片以磁盘路径为准"的约定 ——
落盘产物是 trace 的一部分,人可以直接打开核对模型当时到底看到了什么。

本模块只依赖 numpy / Pillow,不碰仿真器,因此 planner 的单测引入它不会拖进
mujoco 这类重依赖。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb(path: str | Path, rgb: np.ndarray) -> str:
    """把一个 (H, W, 3) 的 uint8 数组存成 PNG;返回路径字符串。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(target)
    return str(target)


def image_content_part(path: str | Path, *, max_edge: int = 1024) -> dict:
    """把磁盘上的图片编码成 OpenAI 兼容的 ``image_url`` 消息片段。

    磁盘上的原图始终保持不动 —— 它是 trace 证据。只有送进 prompt 的那一份副本
    会被压到 ``max_edge`` 以内并转成 JPEG:观测每拍都要重发一次,不设上限的话
    单张几 MB 的 PNG 会在多步运行里迅速吃满上下文。
    """

    source = Path(path)
    if max_edge < 64:
        raise ValueError("max_edge must be at least 64")

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)

    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


__all__ = ["image_content_part", "save_rgb"]
