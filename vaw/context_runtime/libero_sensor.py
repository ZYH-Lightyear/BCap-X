"""VAW-only sensor adapters for LIBERO-PRO.

The policy Canvas and all RGB-D geometry keep using the environment's normal
observation.  Semantic grounding may use a denser render of the same public
agentview so small package labels are not destroyed before detection.  The
result never leaves the episode-private context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def make_libero_semantic_rgb_provider(
    env: Any,
    *,
    scale: float = 2.0,
) -> Callable[[], np.ndarray]:
    """Return a lazy high-resolution agentview renderer for one LIBERO env."""

    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("semantic render scale must be finite and >= 1")
    base_width = int(env._render_width)
    base_height = int(env._render_height)
    width = max(2, int(round(base_width * scale)))
    height = max(2, int(round(base_height * scale)))

    def capture() -> np.ndarray:
        frame = env.handle.env.sim.render(
            camera_name="agentview",
            width=width,
            height=height,
            depth=False,
        )
        return np.ascontiguousarray(np.asarray(frame)[::-1, :, :3])

    return capture


__all__ = ["make_libero_semantic_rgb_provider"]
