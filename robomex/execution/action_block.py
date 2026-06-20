from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionBlockStatus(str, Enum):
    """Lifecycle status for a semantic action block."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SemanticActionBlock:
    """A small executable unit with a natural-language intent and Python code."""

    name: str
    intent: str
    code: str
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTraceEvent:
    """One normalized event from code execution or API calls."""

    event_type: str
    message: str
    block_name: str | None = None
    line_no: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockExecutionResult:
    """Result returned after executing a semantic action block."""

    block: SemanticActionBlock
    ok: bool
    status: ActionBlockStatus
    stdout: str = ""
    stderr: str = ""
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    observation: dict[str, Any] | None = None
    # The sandbox's structured ``RESULT`` after this block (first-class handoff to
    # the verifier; ``None`` if the code set nothing). Mirrors CapX ``info["result"]``.
    result: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    trace_events: tuple[ExecutionTraceEvent, ...] = ()

    @classmethod
    def skipped(cls, block: SemanticActionBlock, reason: str) -> BlockExecutionResult:
        """Build a skipped result while preserving the block identity."""

        return cls(
            block=block,
            ok=False,
            status=ActionBlockStatus.SKIPPED,
            stderr=reason,
        )

