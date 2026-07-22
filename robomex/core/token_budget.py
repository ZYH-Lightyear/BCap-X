"""Deterministic conservative accounting for bounded chat-model calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_CHAT_MESSAGE_OVERHEAD = 16
_CHAT_REQUEST_OVERHEAD = 16


def conservative_text_tokens(value: str) -> int:
    """Count UTF-8 bytes as a stable conservative visible-token estimate."""

    return len(str(value).encode("utf-8"))


def conservative_chat_prompt_tokens(
    prompt: Sequence[Mapping[str, Any]],
) -> int:
    """Estimate a canonical chat request, including multimodal and framing data."""

    encoded = json.dumps(
        list(prompt),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded) + _CHAT_REQUEST_OVERHEAD + _CHAT_MESSAGE_OVERHEAD * len(prompt)


__all__ = ["conservative_chat_prompt_tokens", "conservative_text_tokens"]
