"""Runtime dispatcher for the visual pointing backend.

Allows the LIBERO API classes to swap between the original Molmo pointer and
the generic-VLM (e.g. Qwen) adapter without touching the call sites.

Usage:
    from capx.integrations.vision.point_backend import init_point_backend

    self.molmo_point_fn = init_point_backend()

Selection happens via the ``CAPX_POINT_BACKEND`` environment variable. Recognised
values (case-insensitive):

    "molmo"   -> capx.integrations.vision.molmo.init_molmo                (default)
    "qwen"    -> capx.integrations.vision.qwen_vlm_point.init_qwen_vlm_point
    "vlm"     -> alias for "qwen"

Any extra keyword arguments are forwarded to the chosen backend constructor, so
callers can still override model name, base URL, etc.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import PIL


_QWEN_ALIASES = {"qwen", "vlm", "qwen_vlm", "openrouter"}
_MOLMO_ALIASES = {"molmo", "moldmo"}


def _resolve_backend(backend: str | None) -> str:
    raw = (backend if backend is not None else os.environ.get("CAPX_POINT_BACKEND", "molmo"))
    name = (raw or "molmo").strip().lower()
    if name in _QWEN_ALIASES:
        return "qwen"
    if name in _MOLMO_ALIASES:
        return "molmo"
    raise ValueError(
        f"Unknown CAPX_POINT_BACKEND={raw!r}; expected one of "
        f"{sorted(_MOLMO_ALIASES | _QWEN_ALIASES)}."
    )


def init_point_backend(
    backend: str | None = None,
    **kwargs,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Return a Molmo-compatible pointing callable for the selected backend.

    Args:
        backend: Override for the env var. When ``None`` the value of
            ``CAPX_POINT_BACKEND`` is used (default ``"molmo"``).
        **kwargs: Forwarded to the chosen ``init_*`` constructor.

    Returns:
        A callable ``det_fn(image, objects) -> dict[str, (x_px, y_px) | (None, None)]``
        with the same contract as :func:`capx.integrations.vision.molmo.init_molmo`.
    """
    resolved = _resolve_backend(backend)
    if resolved == "qwen":
        from capx.integrations.vision.qwen_vlm_point import init_qwen_vlm_point

        print(
            "[point_backend] using Qwen/VLM pointer "
            f"(model={kwargs.get('model_name', 'openrouter/qwen/qwen3.6-plus')})"
        )
        return init_qwen_vlm_point(**kwargs)

    from capx.integrations.vision.molmo import init_molmo

    print("[point_backend] using Molmo pointer")
    return init_molmo(**kwargs)


__all__ = ["init_point_backend"]
