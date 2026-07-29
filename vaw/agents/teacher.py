"""Teacher role: a frontier model operating the workspace to produce traces.

There is no teacher-specific loop. The teacher is a provider configuration and
nothing else — it runs the same :class:`~vaw.agents.runtime.VAWRuntime` over the
same protocol as the student, which is what makes a teacher trace usable as a
student training sample without a conversion step.

    from vaw.agents.runtime import run_episode
    from vaw.agents.teacher import teacher_provider

    result = run_episode(
        teacher_provider(),
        api,
        "put the red mug on the plate",
        trace_dir="runs/vaw/teacher/ep0",
    )
"""

from __future__ import annotations

from vaw.agents.providers.base import ModelProvider
from vaw.agents.providers.openai import OpenAIProvider
from vaw.agents.providers.text_protocol import TextProtocolProvider

#: The Cap-X LLM proxy every client in this repo expects (``scripts/serve_up.sh``).
DEFAULT_PROXY_URL = "http://127.0.0.1:8110/chat/completions"

DEFAULT_TEACHER_MODEL = "vapi/gpt-5.5"


def teacher_provider(
    model: str = DEFAULT_TEACHER_MODEL,
    *,
    server_url: str = DEFAULT_PROXY_URL,
    api_key: str | None = None,
    protocol: str = "native",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> ModelProvider:
    """Build the teacher's model provider.

    :param protocol: ``"native"`` for function calling, ``"text"`` for the
        ``<tool_call>`` fallback. Native is preferred for the teacher — its
        arguments are schema-constrained, which means fewer malformed actions
        polluting the traces. Fall back to text for any model whose route
        mishandles ``tools``; ``scripts/test_vapi_tool_call_support.py`` is the
        way to find out which.
    """
    provider: ModelProvider = OpenAIProvider(
        model=model,
        server_url=server_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if protocol == "text":
        return TextProtocolProvider(provider)
    if protocol != "native":
        raise ValueError(f"protocol must be 'native' or 'text', got {protocol!r}")
    return provider


__all__ = ["DEFAULT_PROXY_URL", "DEFAULT_TEACHER_MODEL", "teacher_provider"]
