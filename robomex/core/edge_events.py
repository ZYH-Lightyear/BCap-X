"""Closed edge-event vocabulary for subgoal graphs (GaP-style "declare, don't infer").

This module is the single source of truth shared by the Manager prompt, the
compile-time graph validator, and the runtime router.  It lives in ``core`` as
a dependency-free leaf so both ``robomex.prompts`` and ``robomex.authoring``
can import it without cycles.

Runtime exit events are read from structured node results (``failure_kind``,
verdict status, node status) — never inferred from free-text claims.  An edge
whose ``on`` label is not in this vocabulary is a compile error, so a Manager
typo surfaces as a repairable validation message instead of a silently dead
recovery edge.
"""

from __future__ import annotations

from typing import Any

EDGE_EVENT_SUCCESS = "success"

# Specific, recoverable failure kinds a specialist or the runtime may declare.
FAILURE_KINDS: tuple[str, ...] = (
    "failed_grasp",
    "failed_placement",
    "wrong_grounding",
    "stale_observation",
    "infeasible",
    "execution_fault",
)

EDGE_EVENTS: tuple[str, ...] = (
    EDGE_EVENT_SUCCESS,
    "failed",
    *FAILURE_KINDS,
    "exhausted",
    "uncertain",
)

KNOWN_EDGE_EVENTS = frozenset(EDGE_EVENTS)

# Events the runtime itself may emit for ANY node, independent of what the
# specialist skill declares: generic failure (an exception is always possible),
# budget exhaustion, an unverified checkpoint, and a stale input artifact.
# Edges on these events are therefore always routable; skill-declared
# ``exit_conditions`` constrain only the *specific* failure kinds a node can
# raise on its own (M2 contract upgrade).
RUNTIME_EDGE_EVENTS: tuple[str, ...] = (
    "failed",
    "exhausted",
    "uncertain",
    "stale_observation",
)

# Aliases models commonly emit; normalized once at parse time.
_EDGE_EVENT_ALIASES = {
    "failure": "failed",
    "fail": "failed",
    "error": "failed",
    "failed_execution": "failed",
    "ok": EDGE_EVENT_SUCCESS,
    "succeeded": EDGE_EVENT_SUCCESS,
    "stale": "stale_observation",
    "grasp_failed": "failed_grasp",
}


def normalize_edge_event(raw: Any) -> str:
    """Map one authored/emitted event label onto the closed vocabulary.

    Unknown labels are returned lowercased-as-is so graph validation can
    reject them with an actionable message rather than masking the typo here.
    """

    text = str(raw or "").strip().lower()
    if not text:
        return EDGE_EVENT_SUCCESS
    return _EDGE_EVENT_ALIASES.get(text, text)
