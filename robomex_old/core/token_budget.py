"""Deterministic conservative accounting for bounded chat-model calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_CHAT_MESSAGE_OVERHEAD = 16
_CHAT_REQUEST_OVERHEAD = 16
_IMAGE_PART_BUDGET = 16_384


def conservative_text_tokens(value: str) -> int:
    """Count UTF-8 bytes as a stable conservative visible-token estimate."""

    return len(str(value).encode("utf-8"))


def conservative_chat_prompt_tokens(
    prompt: Sequence[Mapping[str, Any]],
) -> int:
    """Estimate a canonical chat request, including multimodal and framing data.

    Inline image bytes are transport encoding, not text tokens. Counting their
    base64 length previously exhausted a Coding Agent's whole budget before it
    could inspect the first render. Each image therefore receives a fixed,
    deliberately generous vision-token allowance while ordinary text remains
    byte-counted.
    """

    encoded = json.dumps(
        _replace_inline_images(list(prompt)),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        len(encoded)
        + _IMAGE_PART_BUDGET * _count_inline_images(prompt)
        + _CHAT_REQUEST_OVERHEAD
        + _CHAT_MESSAGE_OVERHEAD * len(prompt)
    )


def _replace_inline_images(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, Mapping):
                url = str(image_url.get("url", ""))
                if url.startswith("data:image/"):
                    return {
                        "type": "image_url",
                        "image_url": {"url": "<inline image>"},
                    }
        return {str(key): _replace_inline_images(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_replace_inline_images(item) for item in value]
    return value


def _count_inline_images(value: Any) -> int:
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, Mapping) and str(image_url.get("url", "")).startswith(
                "data:image/"
            ):
                return 1
        return sum(_count_inline_images(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_count_inline_images(item) for item in value)
    return 0


__all__ = ["conservative_chat_prompt_tokens", "conservative_text_tokens"]
