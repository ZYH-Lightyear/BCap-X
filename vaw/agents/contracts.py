"""Provider/runtime data contracts for the VAW Context Runtime.

This layer holds data only. Providers, context history, and the runtime depend
on it, so behavior belongs in those layers rather than in these records.

History is kept in OpenAI wire format (``{"role": ..., "content": ...}``)
rather than a custom content model: we only ever talk to OpenAI-compatible
endpoints, and an intermediate model would just add a lossy translation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

#: One part of a multimodal message: ``{"type": "text", ...}`` or
#: ``{"type": "image_url", ...}``.
ContentPart = dict[str, Any]

#: One chat message in OpenAI wire format.
Message = dict[str, Any]

Role = Literal["system", "user", "assistant", "tool"]


class TerminateMode(enum.Enum):
    """Why the episode loop stopped.

    ``GOAL`` means the agent called the ``done`` op — note that this says
    nothing about task success, which is ``EpisodeResult.claimed_success``
    (the agent's own claim) or an environment-side check.
    ``ENV_TERMINATED`` is an environment/controller boundary exit and is never
    inferred from model-visible state.
    """

    GOAL = "goal"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ENV_TERMINATED = "env_terminated"
    ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    """One op invocation emitted by the model.

    :param id: Issued by the model/provider. It must be echoed back verbatim
        on the tool message; it is the only thing pairing a result with its
        call.
    :param args: Already parsed from the JSON string. On malformed JSON this
        is empty and ``parse_error`` carries the reason — bad arguments are a
        normal failure to feed back to the model, not a program error, and
        raising here would lose the call id needed to answer it.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """A normalised single reply from the model."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    # The exact assistant content before a text-protocol parser removes the
    # operation block. This is trace-only diagnostic evidence; policy history
    # replays only structured calls and results.
    raw_response_text: str = ""
    # Some OpenAI-compatible routes return a separate reasoning field.  Keep it
    # separate from the concise, policy-visible decision basis in ``text``.
    provider_reasoning: str = ""


@dataclass(frozen=True)
class StepRecord:
    """One executed op, flattened for the episode result.

    Deliberately not the full ContextStepResult or rendered Context image:
    retaining every raster in the episode result would duplicate trace data.
    """

    turn: int
    op: str
    args: dict[str, Any]
    ok: bool
    physical: bool
    result: str
    thought: str = ""


@dataclass
class EpisodeResult:
    """Outcome of one VAW episode."""

    terminate_mode: TerminateMode
    turns: int
    #: What the agent claimed via ``done(success=...)``. Never a ground-truth
    #: success signal — the environment check is separate, and the gap between
    #: the two is itself a quantity the paper reports.
    claimed_success: bool = False
    #: The environment's own verdict, when the workspace was given an
    #: ``env_check`` (LIBERO's ``task_completed``). ``None`` means no checker
    #: was wired in. Lives in trace ``meta.json`` too; never shown to the agent.
    env_success: bool | None = None
    #: Extra context on the exit: timeout seconds, provider error, etc.
    detail: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    trace_dir: str | None = None

    @property
    def ended_by_agent(self) -> bool:
        return self.terminate_mode is TerminateMode.GOAL


__all__ = [
    "ContentPart",
    "EpisodeResult",
    "Message",
    "ModelResponse",
    "Role",
    "StepRecord",
    "TerminateMode",
    "ToolCall",
]
