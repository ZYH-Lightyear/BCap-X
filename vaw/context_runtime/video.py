"""Episode video artifacts for the real M1.3 Context runner."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from itertools import chain
from typing import Any

import imageio
import numpy as np
from PIL import Image


def save_episode_videos(
    trace_dir: str | pathlib.Path,
    env: Any,
    *,
    environment_fps: int = 30,
    context_fps: int = 2,
) -> dict[str, Any]:
    """Save simulator camera streams and the policy-visible Context timeline."""

    root = pathlib.Path(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    sources: tuple[tuple[str, int, Iterable[np.ndarray]], ...] = (
        (
            "agentview",
            environment_fps,
            _environment_frames(env, "get_video_frames"),
        ),
        (
            "wrist",
            environment_fps,
            _environment_frames(env, "get_wrist_video_frames"),
        ),
        ("context", context_fps, _context_frames(root)),
    )
    for name, fps, frames in sources:
        path = root / f"video_{name}.mp4"
        try:
            frame_count = _write_mp4(path, frames, fps=fps)
        except Exception as exc:  # artifact export must not hide episode results
            path.unlink(missing_ok=True)
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if frame_count == 0:
            continue
        artifacts[name] = {
            "path": path.name,
            "fps": fps,
            "frames": frame_count,
        }

    result: dict[str, Any] = {"artifacts": artifacts}
    if errors:
        result["errors"] = errors
    return result


def _environment_frames(env: Any, method_name: str) -> Iterable[np.ndarray]:
    method = getattr(env, method_name, None)
    if not callable(method):
        return
    yield from method(clear=True)


def _context_frames(root: pathlib.Path) -> Iterable[np.ndarray]:
    for path in sorted(root.glob("context_*.png")):
        with Image.open(path) as image:
            yield np.asarray(image.convert("RGB"), dtype=np.uint8)


def _write_mp4(
    path: pathlib.Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int,
) -> int:
    if fps <= 0:
        raise ValueError("video fps must be positive")
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        path.unlink(missing_ok=True)
        return 0

    count = 0
    with imageio.get_writer(
        path,
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for frame in chain((first,), iterator):
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"video frame must be HxWx3, got {rgb.shape}")
            writer.append_data(np.ascontiguousarray(rgb))
            count += 1
    return count


__all__ = ["save_episode_videos"]
