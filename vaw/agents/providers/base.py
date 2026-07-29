"""Model endpoint boundary.

Forked from ``agentx/providers/base.py``, unchanged in substance: one method
turning messages + tool schemas into a
:class:`~vaw.agents.contracts.ModelResponse`. Everything above it (chat,
runtime) stays ignorant of whether the teacher is a frontier API, the student
is a local vLLM server, or either is behind the Cap-X proxy.
"""

from __future__ import annotations

from typing import Any, Protocol

from vaw.agents.contracts import Message, ModelResponse


class ModelProvider(Protocol):
    """Completion boundary."""

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """Run one completion.

        :param messages: Full conversation in OpenAI wire format.
        :param tools: OpenAI function-calling schemas; ``None`` or empty means
            no tools this turn.
        """
        ...


class ProviderError(RuntimeError):
    """The endpoint returned something unusable."""


__all__ = ["ModelProvider", "ProviderError"]
