"""Renderer boundary for the Visual Action Workspace.

The policy-facing contract remains an RGB numpy array.  Renderers may use PIL,
a browser, or another deterministic presentation layer, but they all consume
the same state, observation, cameras and scene cloud.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Protocol

import numpy as np
from PIL import Image

from vaw.cloud import SceneCloud
from vaw.render import CANVAS_H, CANVAS_W, render_canvas
from vaw.state import ActionState


class WorkspaceRenderer(Protocol):
    """A synchronous renderer owned by one :class:`vaw.workspace.Workspace`."""

    name: str
    width: int
    height: int

    def render(
        self,
        state: ActionState,
        obs: dict[str, Any] | None,
        *,
        camera_name: str,
        wrist_camera_name: str,
        cloud: SceneCloud,
    ) -> np.ndarray:
        """Return one ``(height, width, 3)`` uint8 policy observation."""

    def close(self) -> None:
        """Release renderer-owned resources.  Pure renderers may do nothing."""


class PILRenderer:
    """The existing deterministic numpy/PIL canvas."""

    name = "pil-v2"
    width = CANVAS_W
    height = CANVAS_H

    def render(
        self,
        state: ActionState,
        obs: dict[str, Any] | None,
        *,
        camera_name: str,
        wrist_camera_name: str,
        cloud: SceneCloud,
    ) -> np.ndarray:
        return render_canvas(
            state,
            obs,
            camera_name=camera_name,
            wrist_camera_name=wrist_camera_name,
            cloud=cloud,
        )

    def close(self) -> None:
        return None


class ComparisonRenderer:
    """Render a primary observation plus a non-policy comparison artifact.

    Only the primary image is returned to the agent.  Paired PNGs live below a
    separate ``_render_compare`` directory so existing ``canvas_*.png`` trace
    readers cannot mistake them for extra workspace steps.
    """

    def __init__(
        self,
        primary: WorkspaceRenderer,
        comparison: WorkspaceRenderer,
        output_dir: str | pathlib.Path,
    ) -> None:
        if (primary.width, primary.height) != (comparison.width, comparison.height):
            raise ValueError("comparison renderers must use the same viewport")
        self.primary = primary
        self.comparison = comparison
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("primary_*.png", "comparison_*.png"):
            for old_artifact in self.output_dir.glob(pattern):
                old_artifact.unlink()
        self.name = f"{primary.name}+compare:{comparison.name}"
        self.width = primary.width
        self.height = primary.height
        self._index = 0
        self._manifest = self.output_dir / "manifest.jsonl"
        self._manifest.unlink(missing_ok=True)

    def render(
        self,
        state: ActionState,
        obs: dict[str, Any] | None,
        *,
        camera_name: str,
        wrist_camera_name: str,
        cloud: SceneCloud,
    ) -> np.ndarray:
        kwargs = {
            "camera_name": camera_name,
            "wrist_camera_name": wrist_camera_name,
            "cloud": cloud,
        }
        primary = self.primary.render(state, obs, **kwargs)
        comparison = self.comparison.render(state, obs, **kwargs)
        primary_name = f"primary_{self._index:04d}.png"
        comparison_name = f"comparison_{self._index:04d}.png"
        Image.fromarray(primary).save(self.output_dir / primary_name)
        Image.fromarray(comparison).save(self.output_dir / comparison_name)
        record = {
            "index": self._index,
            "obs_revision": state.obs_revision,
            "primary_renderer": self.primary.name,
            "comparison_renderer": self.comparison.name,
            "primary": primary_name,
            "comparison": comparison_name,
        }
        with self._manifest.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self._index += 1
        return primary

    def close(self) -> None:
        first_error: Exception | None = None
        for renderer in (self.primary, self.comparison):
            try:
                renderer.close()
            except Exception as exc:  # release both even if one close fails
                first_error = first_error or exc
        if first_error is not None:
            raise first_error


def build_renderer(
    name: str,
    *,
    headed: bool = False,
    compare_dir: str | pathlib.Path | None = None,
) -> WorkspaceRenderer:
    """Construct a renderer without importing Playwright on the PIL path."""

    normalized = name.strip().lower()
    if normalized == "pil":
        primary: WorkspaceRenderer = PILRenderer()
    elif normalized == "web":
        from vaw.web_renderer import WebRenderer

        primary = WebRenderer(headed=headed)
    else:
        raise ValueError("renderer must be 'pil' or 'web'")

    if compare_dir is None:
        return primary
    if normalized == "web":
        comparison: WorkspaceRenderer = PILRenderer()
    else:
        from vaw.web_renderer import WebRenderer

        comparison = WebRenderer(headed=headed)
    return ComparisonRenderer(primary, comparison, compare_dir)


__all__ = [
    "ComparisonRenderer",
    "PILRenderer",
    "WorkspaceRenderer",
    "build_renderer",
]
