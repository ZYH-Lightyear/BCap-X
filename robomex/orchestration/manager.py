"""Bounded, snapshot-based Swarm Manager session for RoboMEx v2.

``SwarmManagerSession`` is a versioned serializable record, not a resident chat
agent.  Every authoring or reactivation call constructs one fresh bounded
request from a compact runtime snapshot.  Only receipts and aggregate usage are
carried across calls; model messages and hidden provider state are not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ManagerError(RuntimeError):
    """Base error for bounded Manager orchestration."""


class ManagerStateError(ManagerError):
    """Raised when a call conflicts with the serialized session state."""


class ManagerBudgetError(ManagerError):
    """Raised when an invoker reports work beyond its granted call budget."""


class ManagerSignal(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """Closed wake-up vocabulary.

    The last four values are deliberately non-waking routine signals.  Keeping
    them in the same vocabulary lets an event bridge make an explicit decision
    instead of relying on string heuristics.
    """

    INTENT_AUTHORING = "intent_authoring"
    RISK_EXPANSION = "risk_expansion"
    ALL_CANDIDATES_REJECTED = "all_candidates_rejected"
    DISAGREEMENT = "disagreement"
    RECOVERY = "recovery"
    LOOP_EXHAUSTED = "loop_exhausted"
    PATCH_REJECTED = "patch_rejected"
    NODE_SUCCESS = "node_success"
    FRAME_TICK = "frame_tick"
    NORMAL_CORRECTION = "normal_correction"
    ACTOR_LIFECYCLE = "actor_lifecycle"


WAKE_SIGNALS = frozenset(
    {
        ManagerSignal.RISK_EXPANSION,
        ManagerSignal.ALL_CANDIDATES_REJECTED,
        ManagerSignal.DISAGREEMENT,
        ManagerSignal.RECOVERY,
        ManagerSignal.LOOP_EXHAUSTED,
        ManagerSignal.PATCH_REJECTED,
    }
)


class ManagerAction(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    AUTHOR_SCAFFOLD = "author_scaffold"
    EXPAND_ROSTER = "expand_roster"
    REPAIR_FRONTIER = "repair_frontier"
    REQUEST_INTENT_REFINEMENT = "request_intent_refinement"
    CLOSE = "close"
    NOOP = "noop"


class ManagerCallKind(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    INITIAL = "initial"
    REACTIVATION = "reactivation"


class ManagerSessionStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    CLOSED = "closed"


class _StrictRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ManagerLimits(_StrictRecord):
    """Hard totals for one Manager session."""

    max_initial_calls: int = Field(default=1, ge=0)
    max_reactivations: int = Field(default=3, ge=0)
    max_tokens: int = Field(default=4096, ge=0)
    max_tokens_per_call: int = Field(default=1024, ge=0)
    max_candidates: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _per_call_cannot_exceed_total(self) -> ManagerLimits:
        if self.max_tokens_per_call > self.max_tokens:
            raise ValueError("max_tokens_per_call cannot exceed max_tokens.")
        return self


class ManagerUsage(_StrictRecord):
    initial_calls: int = Field(default=0, ge=0)
    reactivations: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)


class ManagerRemaining(_StrictRecord):
    initial_calls: int = Field(ge=0)
    reactivations: int = Field(ge=0)
    tokens: int = Field(ge=0)
    candidates: int = Field(ge=0)


class ManagerSnapshot(_StrictRecord):
    """Fresh compact projection supplied to exactly one Manager invocation."""

    schema_version: Literal["robomex.manager_snapshot.v1"] = "robomex.manager_snapshot.v1"
    snapshot_id: NonEmptyStr
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    state_revision: int = Field(ge=0)
    frontier_id: NonEmptyStr | None = None
    triggering_event: dict[str, JsonValue] = Field(default_factory=dict)
    compact_state: dict[str, JsonValue] = Field(default_factory=dict)
    candidate_cards: tuple[dict[str, JsonValue], ...] = ()
    artifact_refs: tuple[NonEmptyStr, ...] = ()
    catalog_refs: tuple[NonEmptyStr, ...] = ()

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ManagerInvocation(_StrictRecord):
    """Stateless request envelope passed to a Manager invoker."""

    schema_version: Literal["robomex.manager_invocation.v1"] = (
        "robomex.manager_invocation.v1"
    )
    invocation_id: NonEmptyStr
    session_id: NonEmptyStr
    session_revision: int = Field(ge=1)
    kind: ManagerCallKind
    signal: ManagerSignal
    snapshot: ManagerSnapshot
    remaining_before: ManagerRemaining
    token_limit: int = Field(ge=0)


class ManagerDecision(_StrictRecord):
    """Bounded structured result from one fresh Manager invocation."""

    schema_version: Literal["robomex.manager_decision.v1"] = "robomex.manager_decision.v1"
    action: ManagerAction
    summary: str = ""
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    candidate_delta: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)


class ManagerCallReceipt(_StrictRecord):
    """Compact durable audit record; it intentionally stores no chat messages."""

    invocation_id: NonEmptyStr
    kind: ManagerCallKind
    signal: ManagerSignal
    snapshot_id: NonEmptyStr
    snapshot_digest: NonEmptyStr
    triggering_event_id: NonEmptyStr | None = None
    succeeded: bool
    action: ManagerAction | None = None
    tokens_used: int = Field(default=0, ge=0)
    candidate_delta: int = Field(default=0, ge=0)
    error: str | None = None


@runtime_checkable
class ManagerInvoker(Protocol):
    """Fresh-call provider boundary; implementations must honor token_limit."""

    def invoke(self, request: ManagerInvocation) -> ManagerDecision:
        """Return one structured decision without retaining session chat."""


@dataclass(frozen=True)
class ManagerStep:
    """Result of attempting one initial or wake-up call."""

    session: SwarmManagerSession
    invoked: bool
    signal: ManagerSignal
    invocation: ManagerInvocation | None = None
    decision: ManagerDecision | None = None
    reason: str = ""


class SwarmManagerSession(_StrictRecord):
    """Serializable session record with pure, bounded transition methods."""

    schema_version: Literal["robomex.swarm_manager_session.v1"] = (
        "robomex.swarm_manager_session.v1"
    )
    session_id: NonEmptyStr
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    intent_id: NonEmptyStr
    record_revision: int = Field(default=1, ge=1)
    status: ManagerSessionStatus = ManagerSessionStatus.ACTIVE
    limits: ManagerLimits = Field(default_factory=ManagerLimits)
    usage: ManagerUsage = Field(default_factory=ManagerUsage)
    last_snapshot_id: NonEmptyStr | None = None
    last_graph_revision: int = Field(default=0, ge=0)
    receipts: tuple[ManagerCallReceipt, ...] = ()

    @model_validator(mode="after")
    def _usage_within_limits(self) -> SwarmManagerSession:
        if self.usage.initial_calls > self.limits.max_initial_calls:
            raise ValueError("initial call usage exceeds its hard limit.")
        if self.usage.reactivations > self.limits.max_reactivations:
            raise ValueError("reactivation usage exceeds its hard limit.")
        if self.usage.tokens > self.limits.max_tokens:
            raise ValueError("token usage exceeds its hard limit.")
        if self.usage.candidates > self.limits.max_candidates:
            raise ValueError("candidate usage exceeds its hard limit.")
        return self

    @property
    def remaining(self) -> ManagerRemaining:
        return ManagerRemaining(
            initial_calls=self.limits.max_initial_calls - self.usage.initial_calls,
            reactivations=self.limits.max_reactivations - self.usage.reactivations,
            tokens=self.limits.max_tokens - self.usage.tokens,
            candidates=self.limits.max_candidates - self.usage.candidates,
        )

    @staticmethod
    def should_wake(signal: ManagerSignal | str) -> bool:
        return ManagerSignal(signal) in WAKE_SIGNALS

    def author(
        self, snapshot: ManagerSnapshot, invoker: ManagerInvoker
    ) -> ManagerStep:
        """Perform one initial authoring call from a fresh snapshot."""

        return self._invoke_fresh(
            kind=ManagerCallKind.INITIAL,
            signal=ManagerSignal.INTENT_AUTHORING,
            snapshot=snapshot,
            invoker=invoker,
        )

    def wake(
        self,
        signal: ManagerSignal | str,
        snapshot: ManagerSnapshot,
        invoker: ManagerInvoker,
    ) -> ManagerStep:
        """Reactivate only for one of the six declared exceptional signals."""

        signal = ManagerSignal(signal)
        if signal not in WAKE_SIGNALS:
            return ManagerStep(
                session=self,
                invoked=False,
                signal=signal,
                reason="routine signal does not wake the Manager",
            )
        return self._invoke_fresh(
            kind=ManagerCallKind.REACTIVATION,
            signal=signal,
            snapshot=snapshot,
            invoker=invoker,
        )

    def close(self) -> SwarmManagerSession:
        if self.status is ManagerSessionStatus.CLOSED:
            return self
        return self.model_copy(
            update={
                "status": ManagerSessionStatus.CLOSED,
                "record_revision": self.record_revision + 1,
            }
        )

    def _invoke_fresh(
        self,
        *,
        kind: ManagerCallKind,
        signal: ManagerSignal,
        snapshot: ManagerSnapshot,
        invoker: ManagerInvoker,
    ) -> ManagerStep:
        if self.status is not ManagerSessionStatus.ACTIVE:
            return ManagerStep(
                session=self,
                invoked=False,
                signal=signal,
                reason=f"Manager session is {self.status.value}",
            )
        self._validate_snapshot(snapshot)
        remaining = self.remaining
        calls_remaining = (
            remaining.initial_calls
            if kind is ManagerCallKind.INITIAL
            else remaining.reactivations
        )
        if calls_remaining <= 0:
            # Exhausting one call class must not poison the other one: an
            # already-authored session may still have valid reactivations.
            return ManagerStep(
                session=self,
                invoked=False,
                signal=signal,
                reason=f"{kind.value} call budget exhausted",
            )
        if remaining.tokens <= 0:
            exhausted = self.model_copy(
                update={
                    "status": ManagerSessionStatus.EXHAUSTED,
                    "record_revision": self.record_revision + 1,
                }
            )
            return ManagerStep(
                session=exhausted,
                invoked=False,
                signal=signal,
                reason="Manager token budget exhausted",
            )

        token_limit = min(self.limits.max_tokens_per_call, remaining.tokens)
        sequence = (
            self.usage.initial_calls + 1
            if kind is ManagerCallKind.INITIAL
            else self.usage.reactivations + 1
        )
        invocation = ManagerInvocation(
            invocation_id=(
                f"{self.session_id}-r{self.record_revision}-{kind.value}-{sequence}"
            ),
            session_id=self.session_id,
            session_revision=self.record_revision,
            kind=kind,
            signal=signal,
            snapshot=snapshot,
            remaining_before=remaining,
            token_limit=token_limit,
        )

        decision: ManagerDecision | None = None
        error = ""
        try:
            decision = invoker.invoke(invocation)
            if not isinstance(decision, ManagerDecision):
                decision = ManagerDecision.model_validate(decision)
            if decision.tokens_used > token_limit:
                raise ManagerBudgetError(
                    f"Invoker reported {decision.tokens_used} tokens for a "
                    f"{token_limit}-token call."
                )
            if decision.candidate_delta > remaining.candidates:
                raise ManagerBudgetError(
                    f"Decision requests {decision.candidate_delta} candidates with "
                    f"only {remaining.candidates} remaining."
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            decision = None

        tokens_used = decision.tokens_used if decision is not None else 0
        candidate_delta = decision.candidate_delta if decision is not None else 0
        usage_update = {
            "initial_calls": self.usage.initial_calls
            + (1 if kind is ManagerCallKind.INITIAL else 0),
            "reactivations": self.usage.reactivations
            + (1 if kind is ManagerCallKind.REACTIVATION else 0),
            "tokens": self.usage.tokens + tokens_used,
            "candidates": self.usage.candidates + candidate_delta,
        }
        updated_usage = ManagerUsage(**usage_update)
        receipt = ManagerCallReceipt(
            invocation_id=invocation.invocation_id,
            kind=kind,
            signal=signal,
            snapshot_id=snapshot.snapshot_id,
            snapshot_digest=snapshot.digest(),
            triggering_event_id=(
                str(snapshot.triggering_event["event_id"])
                if isinstance(snapshot.triggering_event.get("event_id"), str)
                and str(snapshot.triggering_event["event_id"]).strip()
                else None
            ),
            succeeded=decision is not None,
            action=decision.action if decision is not None else None,
            tokens_used=tokens_used,
            candidate_delta=candidate_delta,
            error=error or None,
        )
        status = ManagerSessionStatus.ACTIVE
        if updated_usage.tokens >= self.limits.max_tokens:
            status = ManagerSessionStatus.EXHAUSTED
        updated = self.model_copy(
            update={
                "record_revision": self.record_revision + 1,
                "status": status,
                "usage": updated_usage,
                "last_snapshot_id": snapshot.snapshot_id,
                "last_graph_revision": snapshot.graph_revision,
                "receipts": (*self.receipts, receipt),
            }
        )
        return ManagerStep(
            session=updated,
            invoked=True,
            signal=signal,
            invocation=invocation,
            decision=decision,
            reason=error,
        )

    def _validate_snapshot(self, snapshot: ManagerSnapshot) -> None:
        if snapshot.episode_id != self.episode_id:
            raise ManagerStateError("Manager snapshot belongs to a different episode.")
        if snapshot.workflow_id != self.workflow_id:
            raise ManagerStateError("Manager snapshot belongs to a different workflow.")
        if snapshot.snapshot_id == self.last_snapshot_id:
            raise ManagerStateError(
                "Every Manager invocation requires a fresh snapshot_id."
            )
        if snapshot.graph_revision < self.last_graph_revision:
            raise ManagerStateError("Manager snapshot graph revision moved backwards.")


class ScriptedManagerInvoker:
    """Deterministic no-LLM invoker for tests and replay fixtures."""

    def __init__(self, decisions: tuple[ManagerDecision | Exception, ...]) -> None:
        self._decisions = list(decisions)
        self.requests: list[ManagerInvocation] = []

    def invoke(self, request: ManagerInvocation) -> ManagerDecision:
        self.requests.append(request)
        if not self._decisions:
            raise ManagerError("ScriptedManagerInvoker has no decision remaining.")
        outcome = self._decisions.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


__all__ = [
    "ManagerAction",
    "ManagerBudgetError",
    "ManagerCallKind",
    "ManagerCallReceipt",
    "ManagerDecision",
    "ManagerError",
    "ManagerInvocation",
    "ManagerInvoker",
    "ManagerLimits",
    "ManagerRemaining",
    "ManagerSessionStatus",
    "ManagerSignal",
    "ManagerSnapshot",
    "ManagerStateError",
    "ManagerStep",
    "ManagerUsage",
    "ScriptedManagerInvoker",
    "SwarmManagerSession",
    "WAKE_SIGNALS",
]
