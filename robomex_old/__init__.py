"""RoboMEx: robotic multimodal executable skills."""

from __future__ import annotations

from typing import Any

__all__ = [
    "configure_logging",
    "get_logger",
]


def __getattr__(name: str) -> Any:
    if name in {"configure_logging", "get_logger"}:
        from robomex.core import logging as _logging

        value = getattr(_logging, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
