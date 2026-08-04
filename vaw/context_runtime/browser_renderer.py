"""Persistent-browser bridge for read-only Context Runtime snapshots."""

from __future__ import annotations

import functools
import http.server
import io
import pathlib
import threading
from contextlib import suppress
from typing import Any

import numpy as np
from PIL import Image


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return None


class SnapshotBrowser:
    """One Chromium page that accepts versioned snapshots through JS."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        asset_dir: str | pathlib.Path | None = None,
        headed: bool = False,
        timeout_ms: float = 5000.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.asset_dir = pathlib.Path(
            asset_dir
            or pathlib.Path(__file__).resolve().parent.parent.parent / "vaw-ui" / "dist"
        ).resolve()
        index = self.asset_dir / "index.html"
        if not index.exists():
            raise RuntimeError(
                f"VAW Web assets are missing at {index}; run `npm run build` in vaw-ui"
            )
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Web rendering requires the optional `playwright` dependency") from exc

        handler = functools.partial(_QuietHandler, directory=str(self.asset_dir))
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="vaw-web-assets",
            daemon=True,
        )
        self._server_thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        self.closed = False
        self.page_errors: list[str] = []
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=not headed)
            self.context = self.browser.new_context(
                viewport={"width": self.width, "height": self.height},
                screen={"width": self.width, "height": self.height},
                device_scale_factor=1,
                color_scheme="light",
                reduced_motion="reduce",
            )
            self.page = self.context.new_page()
            self.page.set_default_timeout(timeout_ms)
            self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
            self.page.on(
                "console",
                lambda message: (
                    self.page_errors.append(message.text)
                    if message.type == "error"
                    else None
                ),
            )
            self.page.goto(self.url, wait_until="networkidle")
            self.page.wait_for_function("() => typeof window.__VAW_RENDER__ === 'function'")
        except Exception:
            self.close()
            raise

    def render(
        self,
        snapshot: dict[str, Any],
        *,
        render_id: str,
        full_page: bool,
        min_height: int | None = None,
    ) -> np.ndarray:
        if self.closed:
            raise RuntimeError("SnapshotBrowser is already closed")
        self.page_errors.clear()
        self.page.evaluate(
            """async (snapshot) => {
                if (typeof window.__VAW_RENDER__ !== "function") {
                    throw new Error("VAW render bridge is unavailable");
                }
                await window.__VAW_RENDER__(snapshot);
            }""",
            snapshot,
        )
        self.page.wait_for_function(
            "(id) => document.documentElement.dataset.renderId === id",
            arg=render_id,
        )
        if self.page_errors:
            raise RuntimeError("VAW Web page reported an error: " + "; ".join(self.page_errors))
        png = self.page.screenshot(type="png", full_page=full_page, animations="disabled")
        image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
        expected_height = min_height if full_page else self.height
        if image.ndim != 3 or image.shape[1] != self.width or image.shape[2] != 3:
            raise RuntimeError(
                f"VAW Web screenshot has shape {image.shape}; expected width {self.width}"
            )
        if expected_height is not None and image.shape[0] < expected_height:
            raise RuntimeError(
                f"VAW Web screenshot height {image.shape[0]} is below {expected_height}"
            )
        return image

    def close(self) -> None:
        if getattr(self, "closed", False):
            return
        self.closed = True
        for name in ("page", "context", "browser"):
            resource = getattr(self, name, None)
            if resource is not None:
                with suppress(Exception):
                    resource.close()
        playwright = getattr(self, "playwright", None)
        if playwright is not None:
            with suppress(Exception):
                playwright.stop()
        server = getattr(self, "_server", None)
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = getattr(self, "_server_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


__all__ = ["SnapshotBrowser"]
