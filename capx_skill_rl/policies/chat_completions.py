"""OpenAI-compatible multimodal tool-call policy."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import requests
from PIL import Image

from capx_skill_rl.env import ActionLike
from capx_skill_rl.loop import ToolExchange
from capx_skill_rl.prompts import M3_SYSTEM_PROMPT


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


RequestFn = Callable[..., HttpResponse]


@dataclass(frozen=True, slots=True)
class ChatPolicyConfig:
    model: str = "vapi/gpt-5.5"
    server_url: str = "http://localhost:8110/chat/completions"
    api_key: str | None = None
    max_completion_tokens: int = 1536
    timeout_seconds: float = 180.0
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not self.server_url:
            raise ValueError("server_url must be non-empty")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


class ChatCompletionsPolicy:
    """Produce exactly the provider-native tool call returned by the model."""

    def __init__(
        self,
        config: ChatPolicyConfig,
        *,
        system_prompt: str = M3_SYSTEM_PROMPT,
        request_fn: RequestFn | None = None,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self._request_fn = request_fn or requests.post
        self.records: list[dict[str, Any]] = []

    def act(
        self,
        *,
        task: str,
        rgb: np.ndarray,
        history: Sequence[ToolExchange],
        tools: Sequence[dict[str, Any]],
    ) -> ActionLike:
        image_url, image_digest = _encode_rgb(
            rgb,
            jpeg_quality=self.config.jpeg_quality,
        )
        history_text = _history_text(history)
        user_text = (
            f"Task: {task}\n\n"
            f"Prior tool exchanges:\n{history_text}\n\n"
            "The attached image is the current observation. "
            "Choose exactly one next tool call."
        )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
            "tools": list(tools),
            "parallel_tool_calls": False,
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        started = time.monotonic()
        response = self._request_fn(
            self.config.server_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        elapsed = time.monotonic() - started
        try:
            body = response.json()
        except Exception as exc:
            raise RuntimeError(
                "policy server returned non-JSON response "
                f"(status={response.status_code}, body={response.text[:500]!r})"
            ) from exc
        if response.status_code >= 400:
            raise RuntimeError(
                f"policy server returned HTTP {response.status_code}: "
                f"{json.dumps(body, ensure_ascii=False)[:1000]}"
            )
        if not isinstance(body, Mapping):
            raise RuntimeError("policy server response must be a JSON object")

        action, protocol_error = _parse_tool_calls(body)
        self.records.append(
            {
                "step": len(self.records) + 1,
                "model": self.config.model,
                "elapsed_seconds": elapsed,
                "rgb_sha256": image_digest,
                "rgb_shape": list(rgb.shape),
                "history_length": len(history),
                "action": action,
                "protocol_error": protocol_error,
                "response": dict(body),
            }
        )
        return action


def _encode_rgb(rgb: np.ndarray, *, jpeg_quality: int) -> tuple[str, str]:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = buffer.getvalue()
    digest = hashlib.sha256(encoded).hexdigest()
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return data_url, digest


def _history_text(history: Sequence[ToolExchange]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for index, exchange in enumerate(history, start=1):
        action = _jsonable_action(exchange.action)
        lines.append(
            f"{index}. action={json.dumps(action, ensure_ascii=False, separators=(',', ':'))} "
            f"result={json.dumps(exchange.result, ensure_ascii=False, separators=(',', ':'))}"
        )
    return "\n".join(lines)


def _jsonable_action(action: ActionLike) -> Any:
    if isinstance(action, str):
        try:
            return json.loads(action)
        except json.JSONDecodeError:
            return action
    if isinstance(action, Mapping):
        return dict(action)
    return list(action)


def _parse_tool_calls(
    body: Mapping[str, Any],
) -> tuple[ActionLike, str | None]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"name": "", "arguments": {}}, "response has no choices"
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return {"name": "", "arguments": {}}, "model returned no tool call"

    parsed_calls = [_tool_call_action(call) for call in tool_calls]
    if len(parsed_calls) != 1:
        return (
            [action for action, _ in parsed_calls],
            f"model returned {len(parsed_calls)} tool calls",
        )
    action, error = parsed_calls[0]
    return action, error


def _tool_call_action(call: Any) -> tuple[dict[str, Any], str | None]:
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        return {"name": "", "arguments": {}}, "tool call has no function object"
    name = function.get("name")
    raw_arguments = function.get("arguments")
    error: str | None = None
    if isinstance(raw_arguments, str):
        try:
            arguments: Any = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = raw_arguments
            error = "tool arguments are not valid JSON"
    else:
        arguments = raw_arguments
    return {"name": name, "arguments": arguments}, error


__all__ = ["ChatCompletionsPolicy", "ChatPolicyConfig"]
