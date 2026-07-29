"""Provider for OpenAI-compatible endpoints.

Forked from ``agentx/providers/openai.py`` unchanged apart from imports. Talks
to anything exposing ``POST /chat/completions``: OpenAI itself, a vLLM server
hosting the student, or the ``capx.serving.openrouter_server`` proxy used for
the teacher.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.agents.providers.base import ProviderError

#: The proxy's ``normalize_proxy_payload`` does ``budget = max_completion_tokens
#: or max_tokens or 256``. Omit the limit and output is silently capped at 256
#: tokens, which shows up as the model stopping mid-sentence with no error. So
#: max_tokens is always sent explicitly.
DEFAULT_MAX_TOKENS = 8192


class OpenAIProvider:
    """Synchronous ``requests``-based OpenAI-compatible provider.

    Non-streaming only. The hard part of streaming is not SSE but ``tool_calls``
    deltas, which arrive as arbitrarily split fragments keyed by index and can
    be out of order; that needs its own reassembler. VAW never needs token-level
    latency, so the seam stays unimplemented.
    """

    def __init__(
        self,
        model: str = "vapi/claude-opus-4-8",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.server_url = server_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post_with_retry(payload)
        return _parse_response(data)

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.server_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                # 4xx means the request itself is wrong; retrying only burns
                # quota. Only 5xx and rate limits get backoff.

                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        last_error = ProviderError(f"response is not valid JSON: {response.text[:300]}")
                        del exc
                elif response.status_code < 500 and response.status_code != 429:
                    raise ProviderError(
                        f"endpoint returned {response.status_code}: {response.text[:500]}"
                    )
                else:
                    last_error = ProviderError(
                        f"endpoint returned {response.status_code}: {response.text[:300]}"
                    )

            if attempt < self.max_retries - 1:
                time.sleep(2.0 * (attempt + 1))

        raise ProviderError(f"request failed after {self.max_retries} attempts: {last_error}")


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    """Normalise an OpenAI response body into a :class:`ModelResponse`."""

    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"response has no choices: {json.dumps(data)[:300]}")

    choice = choices[0] or {}
    message = choice.get("message") or {}

    text = message.get("content") or ""
    if isinstance(text, list):
        # A few upstreams split assistant content into parts as well.
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

    return ModelResponse(
        text=str(text),
        tool_calls=tuple(_parse_tool_calls(message.get("tool_calls"))),
        finish_reason=str(choice.get("finish_reason") or ""),
        usage=data.get("usage") or {},
    )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Parse ``tool_calls``, recording bad JSON instead of raising.

    Arguments are written by the model, so malformed ones are routine. They
    must go back to the model as an error result, which means producing a
    :class:`ToolCall` carrying ``parse_error`` rather than an exception — an
    exception would take the call id with it, leaving no way to answer it.
    """

    if not isinstance(raw, list):
        return []

    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        # Synthesise a missing id: every later pairing depends on it.
        call_id = str(item.get("id") or f"call_{index}")

        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            calls.append(ToolCall(id=call_id, name=name, args=raw_args))
            continue

        text = str(raw_args or "").strip()
        if not text:
            calls.append(ToolCall(id=call_id, name=name, args={}))
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    parse_error=f"arguments is not valid JSON ({exc}): {text[:200]}",
                )
            )
            continue

        if isinstance(parsed, dict):
            calls.append(ToolCall(id=call_id, name=name, args=parsed))
        else:
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    parse_error=f"arguments must be a JSON object, got {type(parsed).__name__}",
                )
            )

    return calls


__all__ = ["DEFAULT_MAX_TOKENS", "OpenAIProvider"]
