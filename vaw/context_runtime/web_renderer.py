"""Dual-agent fixed-viewport renderer that consumes only a ContextPacket."""

from __future__ import annotations

import pathlib

import numpy as np

from vaw.context_runtime.browser_renderer import SnapshotBrowser
from vaw.context_runtime.packet import (
    CONTEXT_HEIGHT,
    CONTEXT_WIDTH,
    ContextPacket,
)


class ContextWebRenderer:
    name = "context-web-v20-physical-verification"
    width = CONTEXT_WIDTH
    height = CONTEXT_HEIGHT

    def __init__(
        self,
        *,
        asset_dir: str | pathlib.Path | None = None,
        headed: bool = False,
        timeout_ms: float = 8000.0,
    ) -> None:
        self._bridge = SnapshotBrowser(
            width=self.width,
            height=self.height,
            asset_dir=asset_dir,
            headed=headed,
            timeout_ms=timeout_ms,
        )
        self._page = self._bridge.page
        self._render_index = 0
        self._closed = False

    def render(self, packet: ContextPacket) -> np.ndarray:
        if self._closed:
            raise RuntimeError("ContextWebRenderer is already closed")
        render_id = f"context-{self._render_index:06d}"
        self._render_index += 1
        snapshot = packet.web_snapshot(render_id=render_id)
        return self._bridge.render(
            snapshot,
            render_id=render_id,
            full_page=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bridge.close()

    def __enter__(self) -> ContextWebRenderer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["ContextWebRenderer"]
