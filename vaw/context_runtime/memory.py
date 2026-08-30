"""Lightweight agent memory for one VAW episode.

The memory is deliberately an interaction ledger, not a world model.  It
records which Function was requested, whether that Function is physically
effectful, and the revision transition reported by the workspace.  It never
turns a command into a semantic claim such as "the object is grasped".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

InteractionKind = Literal["call", "action"]
DEFAULT_PROMPT_WINDOW = 5


@dataclass(frozen=True)
class InteractionEvent:
    """One committed Function transaction in chronological order."""

    turn: int
    kind: InteractionKind
    function: str
    arguments: dict[str, Any]
    outcome: str
    revision_before: int
    revision_after: int

    def summary(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "kind": self.kind,
            "function": self.function,
            "arguments": dict(self.arguments),
            "outcome": self.outcome,
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
        }


@dataclass(frozen=True)
class _StoredInteraction:
    event: InteractionEvent
    effect_channel: str | None


@dataclass
class InteractionMemory:
    """Complete in-memory ledger with a compact, lossless prompt projection."""

    _records: list[_StoredInteraction] = field(default_factory=list, repr=False)

    @property
    def events(self) -> tuple[InteractionEvent, ...]:
        return tuple(record.event for record in self._records)

    def record(
        self,
        event: InteractionEvent,
        *,
        effect_channel: str | None = None,
    ) -> None:
        self._records.append(_StoredInteraction(event, effect_channel))

    @property
    def last_action(self) -> InteractionEvent | None:
        return self._latest(lambda record: record.event.kind == "action")

    @property
    def last_gripper_action(self) -> InteractionEvent | None:
        return self._latest(lambda record: record.effect_channel == "gripper")

    def snapshot(self) -> list[dict[str, Any]]:
        """Return every event exactly once for trace persistence and replay."""

        return [event.summary() for event in self.events]

    def prompt_lines(self, *, limit: int = DEFAULT_PROMPT_WINDOW) -> list[str]:
        """Render the latest committed calls without truncating the trace ledger."""

        if limit < 1 or not self._records:
            return []
        recent = self._records[-limit:]
        groups: list[list[InteractionEvent]] = []
        for record in recent:
            event = record.event
            if groups and _same_prompt_event(groups[-1][-1], event):
                groups[-1].append(event)
            else:
                groups.append([event])
        return [_render_group(group) for group in groups]

    def _latest(self, predicate: Any) -> InteractionEvent | None:
        for record in reversed(self._records):
            if predicate(record):
                return record.event
        return None


def interaction_outcome(
    result: dict[str, Any],
    *,
    physical_outcome: str | None = None,
) -> str:
    """Project a Function result without adding task-level interpretation."""

    status = result.get("status")
    if isinstance(status, str) and status.strip():
        outcome = status.strip()
    elif "error" in result:
        outcome = "failed"
    elif physical_outcome:
        outcome = str(physical_outcome)
    else:
        outcome = "ok"

    details: list[str] = []
    for key in ("reason", "error", "advisory"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            details.append(f"{key}={value.strip()}")
    return f"{outcome} ({'; '.join(details)})" if details else outcome


def _same_prompt_event(left: InteractionEvent, right: InteractionEvent) -> bool:
    return (
        left.kind == right.kind
        and left.function == right.function
        and left.arguments == right.arguments
        and left.outcome == right.outcome
    )


def _render_group(events: list[InteractionEvent]) -> str:
    first = events[0]
    last = events[-1]
    turn_label = f"t{first.turn}" if len(events) == 1 else f"t{first.turn}-t{last.turn}"
    arguments = json.dumps(
        first.arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    repeat = f" ×{len(events)}" if len(events) > 1 else ""
    revision = f"r{first.revision_before}->r{last.revision_after}"
    kind = "物理动作" if first.kind == "action" else "函数调用"
    return (
        f"{turn_label} [{kind}] {first.function}({arguments}) "
        f"-> 结果={first.outcome} ({revision}){repeat}"
    )


__all__ = [
    "DEFAULT_PROMPT_WINDOW",
    "InteractionEvent",
    "InteractionKind",
    "InteractionMemory",
    "interaction_outcome",
]
