"""Playwright-backed renderer for the legacy read-only VAW workspace."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from vaw.browser_renderer import SnapshotBrowser
from vaw.cloud import SceneCloud
from vaw.state import ActionState
from vaw.web_presenter import build_web_snapshot


class WebRenderer:
    """Legacy schema-v1 renderer; its public behaviour remains unchanged."""

    name = "web-v1"
    width = 1024
    height = 576

    def __init__(
        self,
        *,
        asset_dir: str | pathlib.Path | None = None,
        headed: bool = False,
        timeout_ms: float = 5000.0,
    ) -> None:
        self._bridge = SnapshotBrowser(
            width=self.width,
            height=self.height,
            asset_dir=asset_dir,
            headed=headed,
            timeout_ms=timeout_ms,
        )
        # Kept for the existing deterministic-browser diagnostics.
        self._page = self._bridge.page
        self.url = self._bridge.url
        self.asset_dir = self._bridge.asset_dir
        self._render_index = 0
        self._closed = False

    def render(
        self,
        state: ActionState,
        obs: dict[str, Any] | None,
        *,
        camera_name: str,
        wrist_camera_name: str,
        cloud: SceneCloud,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("WebRenderer is already closed")
        render_id = f"frame-{self._render_index:06d}"
        self._render_index += 1
        snapshot = build_web_snapshot(
            state,
            obs,
            camera_name=camera_name,
            wrist_camera_name=wrist_camera_name,
            cloud=cloud,
            render_id=render_id,
        )
        return self._bridge.render(snapshot, render_id=render_id, full_page=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bridge.close()

    def __enter__(self) -> WebRenderer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["WebRenderer"]
