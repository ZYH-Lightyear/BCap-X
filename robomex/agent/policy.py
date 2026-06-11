"""Code-generation policies for the Skill Agent.

A policy turns a chat-style prompt into either a Python code block to execute or
the ``FINISH`` sentinel. ``LLMCodePolicy`` wraps CapX's ``query_model`` so the
agent is driven by a real LLM; ``ScriptedCodePolicy`` replays canned responses
for offline runs and tests.
"""

from __future__ import annotations

import re
from typing import Protocol

FINISH = "FINISH"

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block, or fall back to the raw text."""

    match = _CODE_FENCE.search(text)
    return (match.group(1) if match else text).strip()


class CodePolicy(Protocol):
    """Generates the next code block (or ``FINISH``) from a prompt."""

    def act(self, prompt: list[dict]) -> str: ...


class LLMCodePolicy:
    """Real LLM policy backed by ``capx.llm.client.query_model``.

    ``model``/``server_url`` follow CapX conventions; OpenRouter models are
    routed through the local proxy automatically by the underlying client.
    """

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def act(self, prompt: list[dict]) -> str:
        content = self._query_model(self._args, prompt)["content"]
        if FINISH in content and not _CODE_FENCE.search(content):
            return FINISH
        return extract_code(content)


class ScriptedCodePolicy:
    """Replays a fixed list of responses; ends with ``FINISH`` when exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def act(self, prompt: list[dict]) -> str:
        if self._index >= len(self._responses):
            return FINISH
        response = self._responses[self._index]
        self._index += 1
        return response if response == FINISH else extract_code(response)
