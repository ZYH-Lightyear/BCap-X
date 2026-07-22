"""Closed, versioned event protocol for the RoboMEx v2 runtime.

The legacy edge-event vocabulary describes routing labels in the v1 DAG.  It
is intentionally not imported here: v2 separates one control transition from
multicast findings, artifacts, lifecycle changes, state proposals, and
orchestration requests.

``RuntimeEvent`` is a discriminated union.  Persisted input must match one of
the declared ``kind`` tags and every event model forbids unknown fields.  A new
kind or outcome therefore requires an explicit schema version rather than
being silently accepted by an older runtime.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RUNTIME_EVENT_SCHEMA_V1 = "robomex.runtime_event.v1"


def _new_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class RuntimeEventBase(BaseModel):
    """Stable envelope shared by every runtime event."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["robomex.runtime_event.v1"] = RUNTIME_EVENT_SCHEMA_V1
    kind: str
    event_id: NonEmptyStr = Field(default_factory=_new_id)
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    timestamp: datetime = Field(default_factory=_utc_now)
    source: NonEmptyStr

    @field_validator("timestamp")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class ControlOutcome(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """Closed control-transition vocabulary for runtime-event schema v1.

    Adding or changing a value is a protocol change and must be accompanied by
    a new runtime-event schema version.  Specific v1 failure labels remain
    available for a lossless legacy adapter; new workflows should prefer the
    compact generic outcomes where possible.
    """

    SUCCESS = "success"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NEEDS_ADJUSTMENT = "needs_adjustment"
    EXHAUSTED = "exhausted"
    TARGET_DRIFT = "target_drift"
    ATTACHMENT_NOT_CONFIRMED = "attachment_not_confirmed"
    INTERRUPTED = "interrupted"
    INFEASIBLE = "infeasible"
    STALE_INPUT = "stale_input"
    FAILED_GRASP = "failed_grasp"
    FAILED_PLACEMENT = "failed_placement"
    WRONG_GROUNDING = "wrong_grounding"
    STALE_OBSERVATION = "stale_observation"
    EXECUTION_FAULT = "execution_fault"


class NodeOutcomeEvent(RuntimeEventBase):
    """The sole primary control transition emitted by a node activation."""

    kind: Literal["node_outcome"] = "node_outcome"
    activation_id: NonEmptyStr
    node_id: NonEmptyStr
    command_id: NonEmptyStr | None = None
    attempt: int | None = Field(default=None, ge=1)
    outcome: ControlOutcome
    reason: str | None = None
    artifact_ids: tuple[NonEmptyStr, ...] = ()
    graph_revision: int | None = Field(default=None, ge=1)


# Compact compatibility name used in the architecture prose.
NodeOutcome = NodeOutcomeEvent


class LifecycleTransition(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    SPAWNED = "spawned"
    INVOKED = "invoked"
    SUSPENDED = "suspended"
    RESUMED = "resumed"
    RETIRED = "retired"
    FAILED = "failed"


class LifecycleEvent(RuntimeEventBase):
    """Observable lifecycle transition of a worker or long-lived sidecar."""

    kind: Literal["lifecycle"] = "lifecycle"
    actor_id: NonEmptyStr
    transition: LifecycleTransition
    actor_type: NonEmptyStr | None = None
    reason: str | None = None


class ServiceStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    STARTED = "started"
    FAILED = "failed"
    STOPPED = "stopped"


class ServiceOutcome(RuntimeEventBase):
    """Scheduler-visible lifecycle result for a sidecar/service activation."""

    kind: Literal["service_outcome"] = "service_outcome"
    activation_id: NonEmptyStr
    command_id: NonEmptyStr
    attempt: int = Field(ge=1)
    status: ServiceStatus
    reason: str | None = None
    graph_revision: int = Field(ge=1)


class ArtifactPublished(RuntimeEventBase):
    """Notification that an immutable artifact entered the episode ledger."""

    kind: Literal["artifact_published"] = "artifact_published"
    artifact_id: NonEmptyStr
    schema_id: NonEmptyStr
    digest: NonEmptyStr
    producer_activation_id: NonEmptyStr | None = None
    port: NonEmptyStr | None = None
    size_bytes: int | None = Field(default=None, ge=0)


class MonitorFindingKind(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    ATTACHMENT_ANOMALY = "attachment_anomaly"
    TARGET_MOTION = "target_motion"
    UNSAFE_DEVIATION = "unsafe_deviation"
    UNOBSERVABLE = "unobservable"


class FindingSeverity(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitorFinding(RuntimeEventBase):
    """Read-only monitor observation; never a direct state mutation."""

    kind: Literal["monitor_finding"] = "monitor_finding"
    finding_id: NonEmptyStr = Field(default_factory=_new_id)
    monitor_id: NonEmptyStr
    finding: MonitorFindingKind
    severity: FindingSeverity = FindingSeverity.WARNING
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    action_id: NonEmptyStr | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ActionStatus(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"
    EXECUTION_FAULT = "execution_fault"
    REJECTED = "rejected"
    STALE_PLAN = "stale_plan"
    INDETERMINATE_AFTER_CRASH = "indeterminate_after_crash"
    INDETERMINATE_AFTER_TIMEOUT = "indeterminate_after_timeout"


class ActionOutcome(RuntimeEventBase):
    """Runtime-owned outcome of an admitted authoritative action."""

    kind: Literal["action_outcome"] = "action_outcome"
    action_id: NonEmptyStr
    status: ActionStatus
    reason: str | None = None
    plan_digest: NonEmptyStr | None = None
    receipt_ref: NonEmptyStr | None = None
    triggering_finding_id: NonEmptyStr | None = None


class StateProposal(RuntimeEventBase):
    """Evidence-backed request for the sole reducer to revise embodied state."""

    kind: Literal["state_proposal"] = "state_proposal"
    proposal_id: NonEmptyStr = Field(default_factory=_new_id)
    base_state_revision: int = Field(ge=0)
    updates: dict[str, JsonValue]
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    reason: str | None = None


class PatchRequest(RuntimeEventBase):
    """Manager wake-up request; this event is not itself a graph patch."""

    kind: Literal["patch_request"] = "patch_request"
    request_id: NonEmptyStr = Field(default_factory=_new_id)
    base_graph_revision: int = Field(ge=1)
    reason: NonEmptyStr
    requested_scope: NonEmptyStr | None = None
    triggering_event_id: NonEmptyStr | None = None
    constraints: tuple[NonEmptyStr, ...] = ()


class RosterOperation(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    SPAWN = "spawn"
    SUSPEND = "suspend"
    RESUME = "resume"
    RETIRE = "retire"


class RosterUpdate(RuntimeEventBase):
    """Actor-roster change inside an authorized slot; topology is unchanged."""

    kind: Literal["roster_update"] = "roster_update"
    update_id: NonEmptyStr = Field(default_factory=_new_id)
    operation: RosterOperation
    actor_id: NonEmptyStr
    slot_id: NonEmptyStr | None = None
    actor_profile_ref: NonEmptyStr | None = None
    reason: str | None = None


class ArenaDecision(RuntimeEventBase):
    """Auditable candidate selection; never masquerades as a roster change."""

    kind: Literal["arena_decision"] = "arena_decision"
    arena_run_id: NonEmptyStr
    slot_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    selected_candidate_id: NonEmptyStr | None = None
    considered_candidate_ids: tuple[NonEmptyStr, ...] = ()
    rejected_candidate_ids: tuple[NonEmptyStr, ...] = ()
    selection_reason: NonEmptyStr
    selected_hypothesis_ref: NonEmptyStr | None = None
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class GraphPatchOutcome(RuntimeEventBase):
    """Atomic topology-patch receipt projected onto the runtime event log."""

    kind: Literal["graph_patch_outcome"] = "graph_patch_outcome"
    patch_id: NonEmptyStr
    slot_id: NonEmptyStr
    operation: Literal["fill_slot", "replace_unexecuted_fragment"]
    accepted: bool
    before_revision: int = Field(ge=1)
    after_revision: int = Field(ge=1)
    before_digest: NonEmptyStr
    after_digest: NonEmptyStr
    reason_codes: tuple[NonEmptyStr, ...] = ()
    receipt_ref: NonEmptyStr | None = None


class ManagerInvocationOutcome(RuntimeEventBase):
    """One bounded fresh Manager call; no chat transcript is persisted."""

    kind: Literal["manager_invocation_outcome"] = "manager_invocation_outcome"
    session_id: NonEmptyStr
    session_revision: int = Field(ge=1)
    signal: Literal[
        "intent_authoring",
        "risk_expansion",
        "all_candidates_rejected",
        "disagreement",
        "recovery",
        "loop_exhausted",
        "patch_rejected",
        "node_success",
        "frame_tick",
        "normal_correction",
        "actor_lifecycle",
    ]
    invoked: bool
    action: Literal[
        "author_scaffold",
        "expand_roster",
        "repair_frontier",
        "request_intent_refinement",
        "close",
        "noop",
    ] | None = None
    reason: str | None = None
    manager_record_ref: NonEmptyStr | None = None
    triggering_event_id: NonEmptyStr | None = None


RuntimeEvent: TypeAlias = Annotated[  # noqa: UP040 - Python 3.10 compatibility
    NodeOutcomeEvent
    | LifecycleEvent
    | ServiceOutcome
    | ArtifactPublished
    | MonitorFinding
    | ActionOutcome
    | StateProposal
    | PatchRequest
    | RosterUpdate
    | ArenaDecision
    | GraphPatchOutcome
    | ManagerInvocationOutcome,
    Field(discriminator="kind"),
]

_RUNTIME_EVENT_ADAPTER = TypeAdapter(RuntimeEvent)


def deserialize_runtime_event(
    value: Mapping[str, object] | str | bytes | bytearray | RuntimeEventBase,
) -> RuntimeEvent:
    """Validate one persisted event against the complete v1 union.

    Unknown ``kind`` tags, unknown outcomes, and extra payload fields raise a
    Pydantic ``ValidationError``.  Callers must not fall back to a generic
    dictionary because doing so would make an older runtime fail open.
    """

    if isinstance(value, (str, bytes, bytearray)):
        return _RUNTIME_EVENT_ADAPTER.validate_json(value)
    return _RUNTIME_EVENT_ADAPTER.validate_python(value)


def serialize_runtime_event(event: RuntimeEventBase) -> dict[str, JsonValue]:
    """Return the canonical JSON-compatible representation of one event."""

    validated = deserialize_runtime_event(event)
    return cast(dict[str, JsonValue], _RUNTIME_EVENT_ADAPTER.dump_python(validated, mode="json"))


def runtime_event_to_json(event: RuntimeEventBase) -> str:
    """Serialize one validated event as UTF-8 JSON text."""

    validated = deserialize_runtime_event(event)
    return _RUNTIME_EVENT_ADAPTER.dump_json(validated).decode("utf-8")


# Short verbs for consumers that prefer parser/dumper terminology.
parse_runtime_event = deserialize_runtime_event
dump_runtime_event = serialize_runtime_event


__all__ = [
    "ActionOutcome",
    "ActionStatus",
    "ArenaDecision",
    "ArtifactPublished",
    "ControlOutcome",
    "FindingSeverity",
    "GraphPatchOutcome",
    "LifecycleEvent",
    "LifecycleTransition",
    "ManagerInvocationOutcome",
    "MonitorFinding",
    "MonitorFindingKind",
    "NodeOutcome",
    "NodeOutcomeEvent",
    "PatchRequest",
    "RUNTIME_EVENT_SCHEMA_V1",
    "RosterOperation",
    "RosterUpdate",
    "RuntimeEvent",
    "RuntimeEventBase",
    "ServiceOutcome",
    "ServiceStatus",
    "StateProposal",
    "deserialize_runtime_event",
    "dump_runtime_event",
    "parse_runtime_event",
    "runtime_event_to_json",
    "serialize_runtime_event",
]
