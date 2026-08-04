"""Student role: the policy being trained (Qwen3-VL behind a local server).

Like the teacher, this is a provider configuration only — same runtime, same
protocol, same context policy. Any divergence here would show up as a
train/inference mismatch, since the student's SFT data is teacher traces
recorded through exactly this path.

Defaults to the text protocol rather than native function calling. Two reasons:
``<tool_call>`` blocks are Qwen's own training format, and a locally served
checkpoint mid-RL cannot be relied on to hold a native tool schema — while the
text path degrades into something still parseable.

    from vaw.agents.student import student_provider
    from vaw.context_runtime.runtime import run_context_episode
    from vaw.context_runtime.web_renderer import ContextWebRenderer

    result = run_context_episode(
        student_provider(), api, instruction, ContextWebRenderer(), trace_dir=...
    )
"""

from __future__ import annotations

from vaw.agents.providers.base import ModelProvider
from vaw.agents.providers.openai import OpenAIProvider
from vaw.agents.providers.text_protocol import TextProtocolProvider

#: Local vLLM/SGLang server hosting the student. Deliberately not 8110: that
#: port is the frontier-model proxy, and pointing the student at it by accident
#: would silently evaluate the teacher instead.
DEFAULT_STUDENT_URL = "http://127.0.0.1:8120/v1/chat/completions"

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def student_provider(
    model: str = DEFAULT_STUDENT_MODEL,
    *,
    server_url: str = DEFAULT_STUDENT_URL,
    api_key: str | None = None,
    protocol: str = "text",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ModelProvider:
    """Build the student's model provider.

    :param temperature: 0 for evaluation so a reported success rate is one
        number rather than a sample from an unreported distribution. RL rollouts
        need sampling and set their own value.
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


__all__ = ["DEFAULT_STUDENT_MODEL", "DEFAULT_STUDENT_URL", "student_provider"]
