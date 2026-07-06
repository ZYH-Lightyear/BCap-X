"""Runtime dispatcher for the visual pointing backend.

Allows the LIBERO API classes to swap between the original Molmo pointer and
the generic-VLM (e.g. Qwen) adapter without touching the call sites.

Usage:
    from capx.integrations.vision.point_backend import init_point_backend

    self.molmo_point_fn = init_point_backend()

Selection happens via the ``CAPX_POINT_BACKEND`` environment variable. Recognised
values (case-insensitive):

    "auto"    -> Molmo first, Qwen/OpenRouter fallback on init/call failure (default)
    "molmo"   -> capx.integrations.vision.molmo.init_molmo
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
_AUTO_ALIASES = {"auto", "fallback", "molmo_then_qwen"}


def _resolve_backend(backend: str | None) -> str:
    raw = (backend if backend is not None else os.environ.get("CAPX_POINT_BACKEND", "auto"))
    name = (raw or "auto").strip().lower()
    if name in _AUTO_ALIASES:
        return "auto"
    if name in _QWEN_ALIASES:
        return "qwen"
    if name in _MOLMO_ALIASES:
        return "molmo"
    raise ValueError(
        f"Unknown CAPX_POINT_BACKEND={raw!r}; expected one of "
        f"{sorted(_AUTO_ALIASES | _MOLMO_ALIASES | _QWEN_ALIASES)}."
    )


def _has_valid_point(result: dict[str, tuple[int | None, int | None]]) -> bool:
    for point in result.values():
        if point is None:
            continue
        x, y = point
        if x is not None and y is not None:
            return True
    return False


def init_point_backend(
    backend: str | None = None,
    **kwargs,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Return a Molmo-compatible pointing callable for the selected backend.

    Args:
        backend: Override for the env var. When ``None`` the value of
            ``CAPX_POINT_BACKEND`` is used (default ``"auto"``).
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
            f"(model={kwargs.get('model_name') or os.environ.get('CAPX_VLM_MODEL', 'openrouter/qwen/qwen3.6-plus')})"
        )
        return init_qwen_vlm_point(**kwargs)

    from capx.integrations.vision.molmo import init_molmo

    if resolved == "auto":
        try:
            molmo_fn = init_molmo(**kwargs)
            print("[point_backend] using Molmo pointer with Qwen/VLM fallback")
        except Exception as exc:  # noqa: BLE001
            print(f"[point_backend] Molmo init failed ({exc!r}); using Qwen/VLM pointer")
            from capx.integrations.vision.qwen_vlm_point import init_qwen_vlm_point

            return init_qwen_vlm_point(**kwargs)

        qwen_fn = None

        def det_fn(
            image: PIL.Image.Image, objects: list[str] | None = None
        ) -> dict[str, tuple[int | None, int | None]]:
            nonlocal qwen_fn
            try:
                result = molmo_fn(image, objects)
                if _has_valid_point(result):
                    return result
                print("[point_backend] Molmo returned no valid point; falling back to Qwen/VLM")
            except Exception as exc:  # noqa: BLE001
                print(f"[point_backend] Molmo request failed ({exc!r}); falling back to Qwen/VLM")

            from capx.integrations.vision.qwen_vlm_point import init_qwen_vlm_point

            if qwen_fn is None:
                qwen_fn = init_qwen_vlm_point(**kwargs)
            return qwen_fn(image, objects)

        return det_fn

    print("[point_backend] using Molmo pointer")
    return init_molmo(**kwargs)


__all__ = ["init_point_backend"]
