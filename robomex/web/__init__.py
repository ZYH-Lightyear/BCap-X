"""Read-only Trace API for RoboMEx offline/live run visualization."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    from robomex.web.server import create_app as _create_app

    return _create_app()
