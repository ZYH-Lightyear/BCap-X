"""Playwright-backed renderer for the read-only VAW Web workspace."""

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

from vaw.cloud import SceneCloud
from vaw.state import ActionState
from vaw.web_presenter import build_web_snapshot


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return None


class WebRenderer:
    """Render the React workspace in one persistent Chromium page.

    There is intentionally no PIL fallback.  A browser failure must invalidate
    the Web condition rather than quietly mixing two observation formats in one
    rollout.
    """

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
        self.asset_dir = pathlib.Path(
            asset_dir
            or pathlib.Path(__file__).resolve().parent.parent / "vaw-ui" / "dist"
        ).resolve()
        index = self.asset_dir / "index.html"
        if not index.exists():
            raise RuntimeError(
                f"VAW Web assets are missing at {index}; run `npm run build` in vaw-ui"
            )

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "WebRenderer requires the optional `playwright` dependency"
            ) from exc

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
        self._closed = False
        self._render_index = 0
        self._page_errors: list[str] = []

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=not headed)
            self._context = self._browser.new_context(
                viewport={"width": self.width, "height": self.height},
                screen={"width": self.width, "height": self.height},
                device_scale_factor=1,
                color_scheme="light",
                reduced_motion="reduce",
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(timeout_ms)
            self._page.on("pageerror", lambda error: self._page_errors.append(str(error)))
            self._page.on(
                "console",
                lambda message: (
                    self._page_errors.append(message.text)
                    if message.type == "error"
                    else None
                ),
            )
            self._page.goto(self.url, wait_until="networkidle")
            self._page.wait_for_function(
                "() => typeof window.__VAW_RENDER__ === 'function'"
            )
        except Exception:
            self.close()
            raise

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
        self._page_errors.clear()
        snapshot = build_web_snapshot(
            state,
            obs,
            camera_name=camera_name,
            wrist_camera_name=wrist_camera_name,
            cloud=cloud,
            render_id=render_id,
        )
        self._page.evaluate(
            """async (snapshot) => {
                if (typeof window.__VAW_RENDER__ !== "function") {
                    throw new Error("VAW render bridge is unavailable");
                }
                await window.__VAW_RENDER__(snapshot);
            }""",
            snapshot,
        )
        self._page.wait_for_function(
            "(renderId) => document.documentElement.dataset.renderId === renderId",
            arg=render_id,
        )
        if self._page_errors:
            raise RuntimeError(
                "VAW Web page reported an error: " + "; ".join(self._page_errors)
            )
        png = self._page.screenshot(
            type="png",
            full_page=False,
            animations="disabled",
        )
        image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise RuntimeError(
                f"VAW Web screenshot has shape {image.shape}, expected {expected}"
            )
        return image

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for name in ("_page", "_context", "_browser"):
            resource = getattr(self, name, None)
            if resource is not None:
                with suppress(Exception):
                    resource.close()
        playwright = getattr(self, "_playwright", None)
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

    def __enter__(self) -> WebRenderer:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["WebRenderer"]
