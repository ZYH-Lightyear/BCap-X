"""Event-driven activation scheduler for RoboMEx v2 workflows.

The scheduler owns one primary control token per workflow.  Workflow/episode
services are separate activations and may stay running while the primary token
moves.  It never calls an LLM or a robot backend itself; it emits commands and
accepts closed runtime events.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.elastic.compiler import CompiledElasticGraph, ElasticGraphCompiler
from robomex.elastic.graph_spec import ActivationLane, EffectScope, ElasticGraphSpec
from robomex.runtime.events import (
    ControlOutcome,
    NodeOutcomeEvent,
    RuntimeEvent,
    RuntimeEventBase,
    ServiceOutcome,
    ServiceStatus,
    dump_runtime_event,
    parse_runtime_event,
)

_MutationResult = TypeVar("_MutationResult")


class SchedulerError(RuntimeError):
    """Base error for invalid workflow transitions."""


class AuthorityConflictError(SchedulerError):
    """Another admitted action owns the same authoritative resource."""


class WorkflowStatus(str, Enum):  # noqa: UP042 - package supports Python 3.10
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"


class ActivationStatus(str, Enum):  # noqa: UP042 - package supports Python 3.10
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    RETIRED = "retired"
    FAILED = "failed"


class _StrictStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _event_digest(event: RuntimeEventBase) -> str:
    canonical = json.dumps(
        dump_runtime_event(event),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class AuthorityLease:
    """Logical reservation for one future authoritative effect.

    This token is not physical admission. ``ActionSupervisor`` upgrades it
    only after checking the sealed spec, live snapshot, certificate and
    monitor binding.
    """

    lease_id: str
    world_id: str
    resource_id: str
    holder_id: str
    action_id: str


class AuthorityRegistry:
    """Resource-scoped logical reservation registry shared by workflows.

    Shadow worlds do not use this registry.  Independent authoritative
    resources (for example two arms with separate controllers) may each hold a
    reservation, but a single ``(world_id, resource_id)`` pair has at most one
    pending physical branch.
    """

    def __init__(self) -> None:
        self._leases: dict[tuple[str, str], AuthorityLease] = {}
        self._lock = threading.RLock()

    def acquire(
        self, *, world_id: str, resource_id: str, holder_id: str, action_id: str
    ) -> AuthorityLease:
        if not all(value.strip() for value in (world_id, resource_id, holder_id, action_id)):
            raise ValueError("Authority lease identity fields must not be empty.")
        with self._lock:
            key = (world_id, resource_id)
            current = self._leases.get(key)
            if current is not None:
                if current.holder_id == holder_id and current.action_id == action_id:
                    return current
                raise AuthorityConflictError(
                    f"Authoritative resource {world_id}/{resource_id} is held by "
                    f"{current.holder_id!r}."
                )
            lease = AuthorityLease(
                lease_id=uuid.uuid4().hex,
                world_id=world_id,
                resource_id=resource_id,
                holder_id=holder_id,
                action_id=action_id,
            )
            self._leases[key] = lease
            return lease

    def restore(self, lease: AuthorityLease) -> AuthorityLease:
        """Reinstall an exact durable lease without minting a new capability."""

        if not all(
            value.strip()
            for value in (
                lease.lease_id,
                lease.world_id,
                lease.resource_id,
                lease.holder_id,
                lease.action_id,
            )
        ):
            raise ValueError("Restored authority lease fields must not be empty.")
        with self._lock:
            key = (lease.world_id, lease.resource_id)
            current = self._leases.get(key)
            if current is not None and current != lease:
                raise AuthorityConflictError(
                    f"Cannot restore lease for {lease.world_id}/{lease.resource_id}; "
                    "the resource is held by a different capability."
                )
            self._leases[key] = lease
            return lease

    def release(self, lease: AuthorityLease) -> None:
        with self._lock:
            key = (lease.world_id, lease.resource_id)
            current = self._leases.get(key)
            if current is None:
                return
            if current != lease:
                raise AuthorityConflictError(
                    "Cannot release an authoritative lease owned by another action."
                )
            del self._leases[key]

    def validate(
        self,
        *,
        lease_id: str,
        world_id: str,
        resource_id: str,
        action_id: str | None = None,
        holder_id: str | None = None,
    ) -> bool:
        """Check that a live lease still names the exact admitted capability.

        This is deliberately read-only: physical admission may verify a
        scheduler-issued capability, but it cannot refresh or recreate one.
        """

        with self._lock:
            current = self._leases.get((world_id, resource_id))
            if current is None or current.lease_id != lease_id:
                return False
            if action_id is not None and current.action_id != action_id:
                return False
            return holder_id is None or current.holder_id == holder_id

    def active(self) -> tuple[AuthorityLease, ...]:
        with self._lock:
            return tuple(self._leases.values())


class TypedEventBus:
    """Append-only, idempotent event log with typed multicast subscriptions."""

    def __init__(self) -> None:
        self._events: list[RuntimeEvent] = []
        self._serialized_by_id: dict[str, dict] = {}
        self._subscriptions: dict[str, frozenset[str] | None] = {}
        self._queues: dict[str, list[RuntimeEvent]] = {}
        self._lock = threading.RLock()

    def subscribe(self, subscriber_id: str, *, kinds: Iterable[str] | None = None) -> None:
        with self._lock:
            if not subscriber_id.strip():
                raise ValueError("subscriber_id must not be empty")
            if subscriber_id in self._subscriptions:
                raise ValueError(f"Subscriber {subscriber_id!r} already exists.")
            selected = None if kinds is None else frozenset(str(kind) for kind in kinds)
            self._subscriptions[subscriber_id] = selected
            self._queues[subscriber_id] = []

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove one ephemeral delivery queue without changing event history."""

        with self._lock:
            if subscriber_id not in self._subscriptions:
                return False
            del self._subscriptions[subscriber_id]
            del self._queues[subscriber_id]
            return True

    def publish(self, event: RuntimeEventBase | dict) -> bool:
        validated = parse_runtime_event(event)
        serialized = dump_runtime_event(validated)
        with self._lock:
            previous = self._serialized_by_id.get(validated.event_id)
            if previous is not None:
                if previous != serialized:
                    raise SchedulerError(
                        f"Event id {validated.event_id!r} was reused with different content."
                    )
                return False
            self._serialized_by_id[validated.event_id] = serialized
            self._events.append(validated)
            for subscriber_id, selected in self._subscriptions.items():
                if selected is None or validated.kind in selected:
                    self._queues[subscriber_id].append(validated)
            return True

    def contains(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._serialized_by_id

    def drain(self, subscriber_id: str) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            if subscriber_id not in self._queues:
                raise KeyError(f"Unknown event subscriber {subscriber_id!r}.")
            queued = tuple(self._queues[subscriber_id])
            self._queues[subscriber_id].clear()
            return queued

    @property
    def history(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)


@dataclass(frozen=True)
class ActivationCommand:
    command_id: str
    episode_id: str
    workflow_id: str
    graph_id: str
    graph_revision: int
    activation_id: str
    attempt: int
    runner_kind: str
    runner_ref: str
    lane: ActivationLane
    effect_scope: EffectScope
    operation: str
    lease: AuthorityLease | None = None


@dataclass
class _ActivationRecord:
    status: ActivationStatus = ActivationStatus.PENDING
    attempts: int = 0
    active_command_id: str | None = None
    active_graph_revision: int | None = None
    lease: AuthorityLease | None = None
    last_outcome: ControlOutcome | None = None


@dataclass(frozen=True)
class ActivationSnapshot:
    activation_id: str
    status: ActivationStatus
    attempts: int
    active_command_id: str | None
    active_graph_revision: int | None
    last_outcome: ControlOutcome | None


@dataclass(frozen=True)
class WorkflowSnapshot:
    episode_id: str
    workflow_id: str
    graph_id: str
    graph_revision: int
    graph_digest: str
    status: WorkflowStatus
    activations: tuple[ActivationSnapshot, ...]
    loop_iterations: dict[str, int]
    terminal_activation: str | None
    terminal_outcome: ControlOutcome | None
    event_count: int


class AuthorityLeaseState(_StrictStateModel):
    lease_id: str = Field(min_length=1, max_length=256)
    world_id: str = Field(min_length=1, max_length=256)
    resource_id: str = Field(min_length=1, max_length=256)
    holder_id: str = Field(min_length=1, max_length=512)
    action_id: str = Field(min_length=1, max_length=256)

    @classmethod
    def from_lease(cls, lease: AuthorityLease) -> AuthorityLeaseState:
        return cls(
            lease_id=lease.lease_id,
            world_id=lease.world_id,
            resource_id=lease.resource_id,
            holder_id=lease.holder_id,
            action_id=lease.action_id,
        )

    def to_lease(self) -> AuthorityLease:
        return AuthorityLease(**self.model_dump())


class ActivationRecordState(_StrictStateModel):
    activation_id: str = Field(min_length=1, max_length=128)
    status: ActivationStatus
    attempts: int = Field(ge=0)
    active_command_id: str | None = Field(default=None, min_length=1, max_length=256)
    active_graph_revision: int | None = Field(default=None, ge=1)
    lease: AuthorityLeaseState | None = None
    last_outcome: ControlOutcome | None = None

    @model_validator(mode="after")
    def _active_shape(self) -> ActivationRecordState:
        active = self.status in {ActivationStatus.RUNNING, ActivationStatus.SUSPENDED}
        if active != (self.active_command_id is not None):
            raise ValueError("Only RUNNING/SUSPENDED records carry an active command.")
        if active != (self.active_graph_revision is not None):
            raise ValueError("Only RUNNING/SUSPENDED records carry an active graph revision.")
        if active and self.attempts < 1:
            raise ValueError("An active activation must have at least one attempt.")
        if self.lease is not None and self.status is not ActivationStatus.RUNNING:
            raise ValueError("Only a RUNNING authoritative activation may carry a lease.")
        return self


class ProcessedEventState(_StrictStateModel):
    event_id: str = Field(min_length=1, max_length=256)
    digest: str = Field(min_length=64, max_length=64)


class PendingServiceFailureState(_StrictStateModel):
    activation_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=4096)


class SchedulerState(_StrictStateModel):
    """Complete restart image of one workflow scheduler.

    The compiled tables are deterministically rebuilt from ``graph`` and
    checked against ``graph_digest``.  Active command identities and exact
    authority leases are retained so restart cannot mint a second physical
    continuation for the same attempt.
    """

    schema_version: Literal["robomex.scheduler_state.v1"] = (
        "robomex.scheduler_state.v1"
    )
    state_revision: int = Field(ge=0)
    episode_id: str = Field(min_length=1, max_length=256)
    workflow_id: str = Field(min_length=1, max_length=256)
    graph: ElasticGraphSpec
    graph_digest: str = Field(min_length=64, max_length=64)
    status: WorkflowStatus
    control_token_activation: str | None = Field(default=None, max_length=128)
    activations: tuple[ActivationRecordState, ...]
    loop_iterations: dict[str, int]
    terminal_activation: str | None = Field(default=None, max_length=128)
    terminal_outcome: ControlOutcome | None = None
    pending_service_failure: PendingServiceFailureState | None = None
    processed_events: tuple[ProcessedEventState, ...] = ()

    @model_validator(mode="after")
    def _unique_identity(self) -> SchedulerState:
        activation_ids = [item.activation_id for item in self.activations]
        if len(activation_ids) != len(set(activation_ids)):
            raise ValueError("Scheduler state repeats an activation id.")
        event_ids = [item.event_id for item in self.processed_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Scheduler state repeats a processed event id.")
        if any(value < 0 for value in self.loop_iterations.values()):
            raise ValueError("Loop iteration counters must be non-negative.")
        primary_ids = {
            spec.activation_id
            for spec in self.graph.activations
            if spec.lane is ActivationLane.PRIMARY
        }
        active_primary = [
            item.activation_id
            for item in self.activations
            if item.activation_id in primary_ids
            and item.status in {ActivationStatus.READY, ActivationStatus.RUNNING}
        ]
        if len(active_primary) > 1:
            raise ValueError("Scheduler state splits its primary control token.")
        expected_token = active_primary[0] if active_primary else None
        if self.control_token_activation != expected_token:
            raise ValueError("Scheduler control token does not match activation state.")
        terminal = self.status not in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}
        if terminal != (self.terminal_activation is not None):
            raise ValueError("Terminal scheduler status must name its terminal activation.")
        if terminal != (self.terminal_outcome is not None):
            raise ValueError("Terminal scheduler status must name its terminal outcome.")
        if self.pending_service_failure is not None and (
            self.status is not WorkflowStatus.RUNNING
        ):
            raise ValueError("A pending service failure requires a RUNNING workflow.")
        return self


class SchedulerStateSink(Protocol):
    """Durable sink used at scheduler commit boundaries."""

    def append(
        self,
        state: SchedulerState,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool: ...


@dataclass(frozen=True)
class _SchedulerImage:
    graph: CompiledElasticGraph
    status: WorkflowStatus
    records: dict[str, _ActivationRecord]
    loop_iterations: dict[str, int]
    terminal_activation: str | None
    terminal_outcome: ControlOutcome | None
    pending_service_failure: tuple[str, str] | None
    processed_event_digests: dict[str, str]
    state_revision: int
    recovery_pending_services: tuple[str, ...]
    recovery_service_inflight: str | None
    recovery_primary: str | None
    recovery_primary_emitted: bool


class ActivationScheduler:
    """Drive one compiled v2 workflow by commands and typed events."""

    def __init__(
        self,
        *,
        episode_id: str,
        workflow_id: str,
        graph: CompiledElasticGraph,
        event_bus: TypedEventBus | None = None,
        authority_registry: AuthorityRegistry | None = None,
        state_sink: SchedulerStateSink | None = None,
    ) -> None:
        if not episode_id.strip() or not workflow_id.strip():
            raise ValueError("episode_id and workflow_id must not be empty")
        self.episode_id = episode_id
        self.workflow_id = workflow_id
        self.graph = graph
        self.event_bus = event_bus or TypedEventBus()
        self.authority_registry = authority_registry or AuthorityRegistry()
        self._state_sink = state_sink
        self._lock = threading.RLock()
        self.status = WorkflowStatus.CREATED
        self._records = {
            spec.activation_id: _ActivationRecord() for spec in graph.spec.activations
        }
        self._loop_iterations = {loop.loop_id: 0 for loop in graph.spec.bounded_loops}
        self._terminal_activation: str | None = None
        self._terminal_outcome: ControlOutcome | None = None
        self._pending_service_failure: tuple[str, str] | None = None
        self._processed_event_digests: dict[str, str] = {}
        self._state_revision = 0
        self._recovery_pending_services: list[str] = []
        self._recovery_service_inflight: str | None = None
        self._recovery_primary: str | None = None
        self._recovery_primary_emitted = False
        if self._state_sink is not None:
            self._state_sink.append(self.export_state(), reason="created")

    def start(self) -> None:
        with self._lock:
            if self.status != WorkflowStatus.CREATED:
                raise SchedulerError(f"Workflow cannot start from {self.status.value!r}.")

            def mutate() -> None:
                self.status = WorkflowStatus.RUNNING
                self._records[self.graph.spec.entry_activation].status = ActivationStatus.READY
                self._enter_loop_if_needed(self.graph.spec.entry_activation)
                for spec in self.graph.spec.activations:
                    if spec.lane == ActivationLane.SERVICE:
                        self._records[spec.activation_id].status = ActivationStatus.READY

            self._commit_mutation("start", mutate)

    def next_commands(self) -> tuple[ActivationCommand, ...]:
        with self._lock:
            if self.status != WorkflowStatus.RUNNING:
                return ()
            if self._has_recovery_frontier():
                return ()
            commands: list[ActivationCommand] = []
            # Services are admitted independently of the primary control token.
            for spec in self.graph.spec.activations:
                record = self._records[spec.activation_id]
                if (
                    spec.lane == ActivationLane.SERVICE
                    and record.status == ActivationStatus.READY
                ):
                    commands.append(self._admit(spec.activation_id, operation="start_service"))
                    break

            # Required services must acknowledge startup before the primary token
            # is admitted.  Otherwise a failed tracker/monitor could race a newly
            # leased physical action in the same scheduler frontier.
            if commands:
                return tuple(commands)

            primary_running = any(
                self._records[spec.activation_id].status == ActivationStatus.RUNNING
                for spec in self.graph.spec.activations
                if spec.lane == ActivationLane.PRIMARY
            )
            if not primary_running:
                ready = [
                    spec.activation_id
                    for spec in self.graph.spec.activations
                    if spec.lane == ActivationLane.PRIMARY
                    and self._records[spec.activation_id].status == ActivationStatus.READY
                ]
                if len(ready) > 1:
                    raise SchedulerError(
                        "A workflow has more than one ready primary activation; "
                        "the control token split."
                    )
                if ready:
                    commands.append(self._admit(ready[0], operation="invoke"))
            return tuple(commands)

    def recovery_commands(self) -> tuple[ActivationCommand, ...]:
        """Re-deliver active commands once per runtime with stable identity.

        Recovery never increments ``attempts``, creates a command id, or mints
        a lease.  Services are rehydrated one at a time and must acknowledge
        ``STARTED`` before the already-running primary is returned.  A fresh
        process reconstructs this ephemeral delivery frontier and may safely
        re-deliver the same identities again.
        """

        with self._lock:
            if self.status is not WorkflowStatus.RUNNING:
                return ()
            if self._recovery_service_inflight is not None:
                return ()
            if self._recovery_pending_services:
                activation_id = self._recovery_pending_services[0]
                self._recovery_service_inflight = activation_id
                return (
                    self._command_for_active_record(
                        activation_id, operation="recover_service"
                    ),
                )
            if self._recovery_primary is not None and not self._recovery_primary_emitted:
                self._recovery_primary_emitted = True
                return (
                    self._command_for_active_record(
                        self._recovery_primary, operation="recover"
                    ),
                )
            return ()

    def _has_recovery_frontier(self) -> bool:
        return bool(
            self._recovery_pending_services
            or self._recovery_service_inflight is not None
            or (
                self._recovery_primary is not None
                and not self._recovery_primary_emitted
            )
        )

    def _command_for_active_record(
        self, activation_id: str, *, operation: str
    ) -> ActivationCommand:
        spec = self._spec(activation_id)
        record = self._records[activation_id]
        if (
            record.status is not ActivationStatus.RUNNING
            or record.active_command_id is None
            or record.active_graph_revision is None
        ):
            raise SchedulerError("Recovery command no longer names a RUNNING activation.")
        return ActivationCommand(
            command_id=record.active_command_id,
            episode_id=self.episode_id,
            workflow_id=self.workflow_id,
            graph_id=self.graph.spec.graph_id,
            graph_revision=record.active_graph_revision,
            activation_id=activation_id,
            attempt=record.attempts,
            runner_kind=spec.runner_kind.value,
            runner_ref=spec.runner_ref,
            lane=spec.lane,
            effect_scope=spec.effect_scope,
            operation=operation,
            lease=record.lease,
        )

    def _admit(self, activation_id: str, *, operation: str) -> ActivationCommand:
        spec = self._spec(activation_id)
        record = self._records[activation_id]
        if record.status != ActivationStatus.READY:
            raise SchedulerError(f"Activation {activation_id!r} is not ready.")
        attempt = record.attempts + 1
        command_id = uuid.uuid4().hex
        lease = None
        if spec.effect_scope == EffectScope.AUTHORITATIVE_WORLD:
            if not spec.authority_world_id or not spec.authoritative_resource:
                raise SchedulerError(
                    "authoritative activation lacks its typed world/resource binding"
                )
            holder_id = f"{self.workflow_id}:{activation_id}:{attempt}"
            lease = self.authority_registry.acquire(
                world_id=spec.authority_world_id,
                resource_id=spec.authoritative_resource,
                holder_id=holder_id,
                action_id=command_id,
            )
        command = ActivationCommand(
            command_id=command_id,
            episode_id=self.episode_id,
            workflow_id=self.workflow_id,
            graph_id=self.graph.spec.graph_id,
            graph_revision=self.graph.spec.revision,
            activation_id=activation_id,
            attempt=attempt,
            runner_kind=spec.runner_kind.value,
            runner_ref=spec.runner_ref,
            lane=spec.lane,
            effect_scope=spec.effect_scope,
            operation=operation,
            lease=lease,
        )

        def mutate() -> ActivationCommand:
            record.status = ActivationStatus.RUNNING
            record.attempts = attempt
            record.active_command_id = command_id
            record.active_graph_revision = self.graph.spec.revision
            record.lease = lease
            return command

        try:
            return self._commit_mutation("admit_command", mutate)
        except Exception:
            if lease is not None:
                self.authority_registry.release(lease)
            raise

    def on_event(self, event: RuntimeEventBase | dict) -> WorkflowSnapshot:
        validated = parse_runtime_event(event)
        with self._lock:
            if (
                validated.episode_id != self.episode_id
                or validated.workflow_id != self.workflow_id
            ):
                raise SchedulerError("Runtime event belongs to a different episode/workflow.")
            digest = _event_digest(validated)
            processed_digest = self._processed_event_digests.get(validated.event_id)
            if processed_digest is not None:
                if processed_digest != digest:
                    raise SchedulerError(
                        f"Processed event id {validated.event_id!r} was reused with "
                        "different content."
                    )
                # Refill a newly attached/recovered event bus if necessary and
                # have the bus itself verify an existing duplicate byte-for-byte.
                self.event_bus.publish(validated)
                return self.snapshot()

            if isinstance(validated, NodeOutcomeEvent):
                self._validate_node_outcome(validated)
            elif isinstance(validated, ServiceOutcome):
                self._validate_service_outcome(validated)

            # The durable event append is the transaction prepare record.  No
            # scheduler field or authority lease has changed if append/fsync
            # raises.  An event whose later state checkpoint fails remains in
            # the log and is safely replayable because it is not marked as
            # processed in the restored scheduler state.
            self.event_bus.publish(validated)
            before = self._capture_image()
            released_lease: AuthorityLease | None = None
            try:
                if isinstance(validated, NodeOutcomeEvent):
                    released_lease = self._apply_node_outcome(validated)
                elif isinstance(validated, ServiceOutcome):
                    self._apply_service_outcome(validated)
                self._processed_event_digests[validated.event_id] = digest
                self._state_revision += 1
                self._persist_state(reason="runtime_event")
            except Exception:
                self._restore_image(before)
                raise
            if released_lease is not None:
                self.authority_registry.release(released_lease)
            return self.snapshot()

    def _validate_node_outcome(self, event: NodeOutcomeEvent) -> None:
        if self.status != WorkflowStatus.RUNNING:
            raise SchedulerError("A terminal workflow cannot accept a new node outcome.")
        if event.activation_id not in self._records:
            raise SchedulerError(f"Unknown activation {event.activation_id!r}.")
        spec = self._spec(event.activation_id)
        if spec.lane != ActivationLane.PRIMARY:
            raise SchedulerError("A service event cannot advance the primary control token.")
        if event.node_id != event.activation_id:
            raise SchedulerError("v2 NodeOutcome node_id must equal its graph activation_id.")
        record = self._records[event.activation_id]
        if record.status != ActivationStatus.RUNNING:
            raise SchedulerError(f"Activation {event.activation_id!r} is not running.")
        if event.command_id != record.active_command_id or event.attempt != record.attempts:
            raise SchedulerError("NodeOutcome does not match the active command/attempt.")
        if event.graph_revision != record.active_graph_revision:
            raise SchedulerError("NodeOutcome graph revision does not match its active command.")
        if record.lease is not None and not self.authority_registry.validate(
            lease_id=record.lease.lease_id,
            world_id=record.lease.world_id,
            resource_id=record.lease.resource_id,
            action_id=record.lease.action_id,
            holder_id=record.lease.holder_id,
        ):
            raise SchedulerError("NodeOutcome authoritative lease is no longer live.")
        if self._pending_service_failure is not None:
            return
        target = self.graph.next_activation(event.activation_id, event.outcome)
        if target is None or not self._would_admit_loop_transition(
            event.activation_id, target
        ):
            return
        target_record = self._records[target]
        if target_record.status in {
            ActivationStatus.READY,
            ActivationStatus.RUNNING,
            ActivationStatus.SUSPENDED,
        }:
            raise SchedulerError(f"Control transition targets active activation {target!r}.")

    def _apply_node_outcome(self, event: NodeOutcomeEvent) -> AuthorityLease | None:
        record = self._records[event.activation_id]
        if self._recovery_primary == event.activation_id:
            self._recovery_primary = None
            self._recovery_primary_emitted = True
        released_lease = record.lease
        record.lease = None
        record.active_command_id = None
        record.active_graph_revision = None
        record.status = ActivationStatus.COMPLETED
        record.last_outcome = event.outcome

        if self._pending_service_failure is not None:
            service_id, reason = self._pending_service_failure
            self.status = WorkflowStatus.UNCERTAIN
            self._terminal_activation = service_id
            self._terminal_outcome = ControlOutcome.UNCERTAIN
            self._pending_service_failure = None
            return released_lease

        target = self.graph.next_activation(event.activation_id, event.outcome)
        if target is None:
            self._finish_without_transition(event)
            return released_lease
        if not self._admit_loop_transition(event.activation_id, target):
            self.status = WorkflowStatus.EXHAUSTED
            self._terminal_activation = event.activation_id
            self._terminal_outcome = ControlOutcome.EXHAUSTED
            return released_lease
        target_record = self._records[target]
        if target_record.status in {
            ActivationStatus.READY,
            ActivationStatus.RUNNING,
            ActivationStatus.SUSPENDED,
        }:
            raise SchedulerError(f"Control transition targets active activation {target!r}.")
        target_record.status = ActivationStatus.READY
        return released_lease

    def _validate_service_outcome(self, event: ServiceOutcome) -> None:
        if self.status != WorkflowStatus.RUNNING:
            raise SchedulerError("A terminal workflow cannot accept a service outcome.")
        if event.activation_id not in self._records:
            raise SchedulerError(f"Unknown service activation {event.activation_id!r}.")
        spec = self._spec(event.activation_id)
        if spec.lane is not ActivationLane.SERVICE:
            raise SchedulerError("ServiceOutcome references a primary activation.")
        record = self._records[event.activation_id]
        if record.status is not ActivationStatus.RUNNING:
            raise SchedulerError(f"Service {event.activation_id!r} is not running.")
        if event.command_id != record.active_command_id or event.attempt != record.attempts:
            raise SchedulerError("ServiceOutcome does not match the active command/attempt.")
        if event.graph_revision != record.active_graph_revision:
            raise SchedulerError("ServiceOutcome graph revision does not match its command.")

    def _apply_service_outcome(self, event: ServiceOutcome) -> None:
        record = self._records[event.activation_id]
        if event.activation_id in self._recovery_pending_services:
            self._recovery_pending_services.remove(event.activation_id)
        if self._recovery_service_inflight == event.activation_id:
            self._recovery_service_inflight = None
        if event.status is ServiceStatus.STARTED:
            return
        if event.status is ServiceStatus.STOPPED:
            record.status = ActivationStatus.RETIRED
            record.active_command_id = None
            record.active_graph_revision = None
            return

        record.status = ActivationStatus.FAILED
        record.active_command_id = None
        record.active_graph_revision = None
        reason = event.reason or "required service failed"
        primary_running = any(
            self._records[item.activation_id].status is ActivationStatus.RUNNING
            for item in self.graph.spec.activations
            if item.lane is ActivationLane.PRIMARY
        )
        if primary_running:
            self._pending_service_failure = (event.activation_id, reason)
        else:
            for item in self.graph.spec.activations:
                if (
                    item.lane is ActivationLane.PRIMARY
                    and self._records[item.activation_id].status is ActivationStatus.READY
                ):
                    self._records[item.activation_id].status = ActivationStatus.RETIRED
            self.status = WorkflowStatus.UNCERTAIN
            self._terminal_activation = event.activation_id
            self._terminal_outcome = ControlOutcome.UNCERTAIN

    def _finish_without_transition(self, event: NodeOutcomeEvent) -> None:
        self._terminal_activation = event.activation_id
        self._terminal_outcome = event.outcome
        is_terminal = event.activation_id in self.graph.spec.terminal_activations
        if is_terminal and event.outcome == ControlOutcome.SUCCESS:
            self.status = WorkflowStatus.SUCCEEDED
        elif event.outcome == ControlOutcome.UNCERTAIN:
            self.status = WorkflowStatus.UNCERTAIN
        elif event.outcome == ControlOutcome.EXHAUSTED:
            self.status = WorkflowStatus.EXHAUSTED
        else:
            self.status = WorkflowStatus.FAILED

    def _enter_loop_if_needed(self, activation_id: str) -> None:
        loop_id = self.graph.loop_by_activation.get(activation_id)
        if loop_id is None:
            return
        if self.graph.loop_entries[loop_id] == activation_id and self._loop_iterations[loop_id] == 0:
            self._loop_iterations[loop_id] = 1

    def _admit_loop_transition(self, source: str, target: str) -> bool:
        source_loop = self.graph.loop_by_activation.get(source)
        target_loop = self.graph.loop_by_activation.get(target)
        if target_loop is None:
            return True
        entry = self.graph.loop_entries[target_loop]
        if target == entry and source_loop == target_loop:
            current = self._loop_iterations[target_loop]
            if current >= self.graph.loop_limits[target_loop]:
                return False
            self._loop_iterations[target_loop] = current + 1
        elif target == entry and self._loop_iterations[target_loop] == 0:
            self._loop_iterations[target_loop] = 1
        return True

    def _would_admit_loop_transition(self, source: str, target: str) -> bool:
        source_loop = self.graph.loop_by_activation.get(source)
        target_loop = self.graph.loop_by_activation.get(target)
        if target_loop is None:
            return True
        entry = self.graph.loop_entries[target_loop]
        if target == entry and source_loop == target_loop:
            return self._loop_iterations[target_loop] < self.graph.loop_limits[target_loop]
        return True

    def suspend_service(self, activation_id: str) -> None:
        with self._lock:
            spec = self._spec(activation_id)
            record = self._records[activation_id]
            if (
                spec.lane != ActivationLane.SERVICE
                or record.status != ActivationStatus.RUNNING
            ):
                raise SchedulerError("Only a running service can be suspended.")
            self._commit_mutation(
                "suspend_service",
                lambda: setattr(record, "status", ActivationStatus.SUSPENDED),
            )

    def resume_service(self, activation_id: str) -> None:
        with self._lock:
            spec = self._spec(activation_id)
            record = self._records[activation_id]
            if (
                spec.lane != ActivationLane.SERVICE
                or record.status != ActivationStatus.SUSPENDED
            ):
                raise SchedulerError("Only a suspended service can be resumed.")
            self._commit_mutation(
                "resume_service",
                lambda: setattr(record, "status", ActivationStatus.RUNNING),
            )

    def retire_services(self, *, include_episode_services: bool = False) -> tuple[str, ...]:
        with self._lock:
            retired = tuple(
                spec.activation_id
                for spec in self.graph.spec.activations
                if spec.lane == ActivationLane.SERVICE
                and (include_episode_services or spec.lifecycle.value != "episode")
                and self._records[spec.activation_id].status != ActivationStatus.RETIRED
            )
            if not retired:
                return ()

            def mutate() -> tuple[str, ...]:
                for activation_id in retired:
                    record = self._records[activation_id]
                    record.status = ActivationStatus.RETIRED
                    record.active_command_id = None
                    record.active_graph_revision = None
                return retired

            return self._commit_mutation("retire_services", mutate)

    def replace_compiled_graph(
        self,
        graph: CompiledElasticGraph,
        *,
        expected_digest: str,
        removed_activation_ids: Iterable[str],
        replacement_entry_activation: str,
        commit_metadata: dict[str, Any] | None = None,
    ) -> WorkflowSnapshot:
        """Durably install a compiler-validated successor revision.

        When a state sink is configured, the successor graph and optional
        patch proposal/receipt metadata share one fsync record.  A failed
        checkpoint restores the complete predecessor scheduler image.
        """

        with self._lock:
            removed = tuple(removed_activation_ids)
            before = self._capture_image()
            try:
                self._replace_compiled_graph(
                    graph,
                    expected_digest=expected_digest,
                    removed_activation_ids=removed,
                    replacement_entry_activation=replacement_entry_activation,
                )
                self._state_revision += 1
                self._persist_state(reason="graph_patch", metadata=commit_metadata)
            except Exception:
                self._restore_image(before)
                raise
            return self.snapshot()

    def _replace_compiled_graph(
        self,
        graph: CompiledElasticGraph,
        *,
        expected_digest: str,
        removed_activation_ids: Iterable[str],
        replacement_entry_activation: str,
    ) -> None:
        """Atomically install one compiler-validated successor revision.

        This is intentionally narrower than a generic graph setter.  It keeps
        every surviving activation record (including unrelated long-lived
        services), refuses to erase anything that has ever been admitted, and
        transfers a not-yet-admitted READY token from the removed frontier to
        the replacement entry.  All checks run before any scheduler field is
        mutated, so a rejected patch leaves the workflow byte-for-byte
        observable at its previous revision.
        """

        removed = frozenset(removed_activation_ids)
        if not removed:
            raise SchedulerError("A graph replacement must remove at least one activation.")
        if self.status not in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}:
            raise SchedulerError("A terminal workflow cannot accept a graph patch.")
        if self.graph.digest != expected_digest:
            raise SchedulerError("The scheduler graph changed before patch commit.")
        if graph.spec.graph_id != self.graph.spec.graph_id:
            raise SchedulerError("A patch cannot change graph_id.")
        if graph.spec.revision != self.graph.spec.revision + 1:
            raise SchedulerError("A patch must advance graph revision by exactly one.")

        old_specs = {spec.activation_id: spec for spec in self.graph.spec.activations}
        new_specs = {spec.activation_id: spec for spec in graph.spec.activations}
        actual_removed = set(old_specs) - set(new_specs)
        if actual_removed != set(removed):
            raise SchedulerError("Patch removal set does not match the compiled successor.")
        if replacement_entry_activation not in new_specs:
            raise SchedulerError("Patch replacement entry is absent from the successor graph.")

        ready_removed: list[str] = []
        for activation_id in sorted(removed):
            record = self._records[activation_id]
            if record.attempts or record.active_command_id is not None:
                raise SchedulerError(
                    f"Activation {activation_id!r} has admitted history and is immutable."
                )
            if record.status not in {ActivationStatus.PENDING, ActivationStatus.READY}:
                raise SchedulerError(
                    f"Activation {activation_id!r} is not an unexecuted frontier member."
                )
            if record.status == ActivationStatus.READY:
                ready_removed.append(activation_id)
        if len(ready_removed) > 1:
            raise SchedulerError("A patch cannot merge multiple READY primary tokens.")

        # Surviving admitted nodes are immutable even if a malformed caller
        # bypasses GraphPatchCoordinator and invokes this integration point.
        for activation_id in set(old_specs).intersection(new_specs):
            record = self._records[activation_id]
            if (record.attempts or record.active_command_id is not None) and (
                old_specs[activation_id] != new_specs[activation_id]
            ):
                raise SchedulerError(
                    f"Admitted activation {activation_id!r} cannot be redefined."
                )

        new_records: dict[str, _ActivationRecord] = {}
        for spec in graph.spec.activations:
            old_record = self._records.get(spec.activation_id)
            if old_record is None:
                new_records[spec.activation_id] = _ActivationRecord()
            else:
                new_records[spec.activation_id] = old_record
        if ready_removed:
            destination = new_records[replacement_entry_activation]
            if destination.status != ActivationStatus.PENDING or destination.attempts:
                raise SchedulerError("Replacement entry cannot receive the READY token.")
            destination.status = ActivationStatus.READY

        new_loop_iterations = {
            loop.loop_id: self._loop_iterations.get(loop.loop_id, 0)
            for loop in graph.spec.bounded_loops
        }

        # Commit point: validation and allocation above cannot mutate scheduler
        # topology.  The three assignments form the in-memory atomic install.
        self.graph = graph
        self._records = new_records
        self._loop_iterations = new_loop_iterations
        return None

    @property
    def state_revision(self) -> int:
        with self._lock:
            return self._state_revision

    @property
    def durable_state_enabled(self) -> bool:
        return self._state_sink is not None

    def attach_state_sink(
        self, sink: SchedulerStateSink, *, persist_current: bool = True
    ) -> None:
        """Attach durable state storage before commands are exposed.

        Attaching a different sink later is forbidden because it would create
        two competing recovery authorities.
        """

        with self._lock:
            if self._state_sink is not None and self._state_sink is not sink:
                raise SchedulerError("Scheduler already has a different state sink.")
            self._state_sink = sink
            if persist_current:
                sink.append(self.export_state(), reason="attach_state_sink")

    def checkpoint(
        self,
        *,
        reason: str = "manual_checkpoint",
        metadata: dict[str, Any] | None = None,
    ) -> SchedulerState:
        with self._lock:
            if not reason.strip():
                raise ValueError("checkpoint reason must not be empty")
            before = self._capture_image()
            try:
                self._state_revision += 1
                self._persist_state(reason=reason, metadata=metadata)
            except Exception:
                self._restore_image(before)
                raise
            return self.export_state()

    def snapshot(self) -> WorkflowSnapshot:
        with self._lock:
            return WorkflowSnapshot(
                episode_id=self.episode_id,
                workflow_id=self.workflow_id,
                graph_id=self.graph.spec.graph_id,
                graph_revision=self.graph.spec.revision,
                graph_digest=self.graph.digest,
                status=self.status,
                activations=tuple(
                    ActivationSnapshot(
                        activation_id=spec.activation_id,
                        status=self._records[spec.activation_id].status,
                        attempts=self._records[spec.activation_id].attempts,
                        active_command_id=self._records[
                            spec.activation_id
                        ].active_command_id,
                        active_graph_revision=self._records[
                            spec.activation_id
                        ].active_graph_revision,
                        last_outcome=self._records[spec.activation_id].last_outcome,
                    )
                    for spec in self.graph.spec.activations
                ),
                loop_iterations=dict(self._loop_iterations),
                terminal_activation=self._terminal_activation,
                terminal_outcome=self._terminal_outcome,
                event_count=len(self._processed_event_digests),
            )

    def export_state(self) -> SchedulerState:
        with self._lock:
            return SchedulerState(
                state_revision=self._state_revision,
                episode_id=self.episode_id,
                workflow_id=self.workflow_id,
                graph=self.graph.spec,
                graph_digest=self.graph.digest,
                status=self.status,
                control_token_activation=next(
                    (
                        spec.activation_id
                        for spec in self.graph.spec.activations
                        if spec.lane is ActivationLane.PRIMARY
                        and self._records[spec.activation_id].status
                        in {ActivationStatus.READY, ActivationStatus.RUNNING}
                    ),
                    None,
                ),
                activations=tuple(
                    ActivationRecordState(
                        activation_id=spec.activation_id,
                        status=self._records[spec.activation_id].status,
                        attempts=self._records[spec.activation_id].attempts,
                        active_command_id=self._records[
                            spec.activation_id
                        ].active_command_id,
                        active_graph_revision=self._records[
                            spec.activation_id
                        ].active_graph_revision,
                        lease=(
                            AuthorityLeaseState.from_lease(
                                self._records[spec.activation_id].lease
                            )
                            if self._records[spec.activation_id].lease is not None
                            else None
                        ),
                        last_outcome=self._records[spec.activation_id].last_outcome,
                    )
                    for spec in self.graph.spec.activations
                ),
                loop_iterations=dict(self._loop_iterations),
                terminal_activation=self._terminal_activation,
                terminal_outcome=self._terminal_outcome,
                pending_service_failure=(
                    PendingServiceFailureState(
                        activation_id=self._pending_service_failure[0],
                        reason=self._pending_service_failure[1],
                    )
                    if self._pending_service_failure is not None
                    else None
                ),
                processed_events=tuple(
                    ProcessedEventState(event_id=event_id, digest=digest)
                    for event_id, digest in self._processed_event_digests.items()
                ),
            )

    @classmethod
    def restore(
        cls,
        state: SchedulerState | dict,
        *,
        event_bus: TypedEventBus | None = None,
        authority_registry: AuthorityRegistry | None = None,
        state_sink: SchedulerStateSink | None = None,
        compiler: ElasticGraphCompiler | None = None,
        replay_pending_events: bool = False,
    ) -> ActivationScheduler:
        """Rebuild a scheduler and optionally consume its durable event tail."""

        restored = (
            state if isinstance(state, SchedulerState) else SchedulerState.model_validate(state)
        )
        compiled = (compiler or ElasticGraphCompiler()).compile(restored.graph)
        if compiled.digest != restored.graph_digest:
            raise SchedulerError("Scheduler state graph digest does not match its graph spec.")
        registry = authority_registry or AuthorityRegistry()
        supplied_event_bus = event_bus is not None
        scheduler = cls(
            episode_id=restored.episode_id,
            workflow_id=restored.workflow_id,
            graph=compiled,
            event_bus=event_bus,
            authority_registry=registry,
            state_sink=None,
        )
        preexisting_leases = set(registry.active())
        restored_leases = {
            item.lease.to_lease()
            for item in restored.activations
            if item.lease is not None
        }
        try:
            scheduler._install_restored_state(restored)
            if supplied_event_bus:
                scheduler._validate_processed_event_history()
            scheduler._state_sink = state_sink
            if replay_pending_events:
                scheduler.replay_pending_events()
        except Exception:
            # A scheduler that fails event-history validation or tail replay
            # must not leave capabilities installed in the caller's shared
            # authority registry.  Preserve leases that predated this restore;
            # only this recovery attempt's newly installed identities belong
            # to its rollback boundary.
            for lease in restored_leases - preexisting_leases:
                if registry.validate(
                    lease_id=lease.lease_id,
                    world_id=lease.world_id,
                    resource_id=lease.resource_id,
                    action_id=lease.action_id,
                    holder_id=lease.holder_id,
                ):
                    registry.release(lease)
            raise
        return scheduler

    def replay_pending_events(self) -> WorkflowSnapshot:
        """Apply workflow events durable in the bus but absent from the checkpoint."""

        with self._lock:
            pending = tuple(
                event
                for event in self.event_bus.history
                if event.episode_id == self.episode_id
                and event.workflow_id == self.workflow_id
                and event.event_id not in self._processed_event_digests
            )
            for event in pending:
                self.on_event(event)
            return self.snapshot()

    def _validate_processed_event_history(self) -> None:
        durable = {event.event_id: _event_digest(event) for event in self.event_bus.history}
        for event_id, expected_digest in self._processed_event_digests.items():
            actual_digest = durable.get(event_id)
            if actual_digest is None:
                raise SchedulerError(
                    f"Scheduler checkpoint references missing event {event_id!r}."
                )
            if actual_digest != expected_digest:
                raise SchedulerError(
                    f"Scheduler checkpoint event {event_id!r} conflicts with the event log."
                )

    def _install_restored_state(self, state: SchedulerState) -> None:
        self._validate_restored_state(state)
        records: dict[str, _ActivationRecord] = {}
        installed: list[AuthorityLease] = []
        try:
            for item in state.activations:
                lease = item.lease.to_lease() if item.lease is not None else None
                if lease is not None:
                    self.authority_registry.restore(lease)
                    installed.append(lease)
                records[item.activation_id] = _ActivationRecord(
                    status=item.status,
                    attempts=item.attempts,
                    active_command_id=item.active_command_id,
                    active_graph_revision=item.active_graph_revision,
                    lease=lease,
                    last_outcome=item.last_outcome,
                )
        except Exception:
            for lease in installed:
                self.authority_registry.release(lease)
            raise
        self.status = state.status
        self._records = records
        self._loop_iterations = dict(state.loop_iterations)
        self._terminal_activation = state.terminal_activation
        self._terminal_outcome = state.terminal_outcome
        self._pending_service_failure = (
            (
                state.pending_service_failure.activation_id,
                state.pending_service_failure.reason,
            )
            if state.pending_service_failure is not None
            else None
        )
        self._processed_event_digests = {
            item.event_id: item.digest for item in state.processed_events
        }
        self._state_revision = state.state_revision
        self._recovery_pending_services = [
            spec.activation_id
            for spec in self.graph.spec.activations
            if spec.lane is ActivationLane.SERVICE
            and self._records[spec.activation_id].status is ActivationStatus.RUNNING
        ]
        self._recovery_service_inflight = None
        self._recovery_primary = next(
            (
                spec.activation_id
                for spec in self.graph.spec.activations
                if spec.lane is ActivationLane.PRIMARY
                and self._records[spec.activation_id].status is ActivationStatus.RUNNING
            ),
            None,
        )
        self._recovery_primary_emitted = False

    def _validate_restored_state(self, state: SchedulerState) -> None:
        specs = {spec.activation_id: spec for spec in self.graph.spec.activations}
        records = {item.activation_id: item for item in state.activations}
        if set(records) != set(specs):
            raise SchedulerError("Scheduler state activation set does not match its graph.")
        if set(state.loop_iterations) != set(self.graph.loop_limits):
            raise SchedulerError("Scheduler state loop set does not match its graph.")
        for loop_id, iteration in state.loop_iterations.items():
            if iteration > self.graph.loop_limits[loop_id]:
                raise SchedulerError(
                    f"Scheduler state loop {loop_id!r} exceeds its compiled bound."
                )

        active_primary = []
        for activation_id, item in records.items():
            spec = specs[activation_id]
            if item.status is ActivationStatus.SUSPENDED and (
                spec.lane is not ActivationLane.SERVICE
            ):
                raise SchedulerError("Only service activations may be restored suspended.")
            if item.active_graph_revision is not None and (
                item.active_graph_revision > self.graph.spec.revision
            ):
                raise SchedulerError("Active command originates from a future graph revision.")
            if spec.lane is ActivationLane.PRIMARY and item.status in {
                ActivationStatus.READY,
                ActivationStatus.RUNNING,
            }:
                active_primary.append(activation_id)
            if item.lease is None:
                if (
                    spec.effect_scope is EffectScope.AUTHORITATIVE_WORLD
                    and item.status is ActivationStatus.RUNNING
                ):
                    raise SchedulerError("Running authoritative activation lacks its lease.")
                continue
            if spec.effect_scope is not EffectScope.AUTHORITATIVE_WORLD:
                raise SchedulerError("A non-authoritative activation carries a lease.")
            expected_world = spec.authority_world_id
            expected_holder = f"{self.workflow_id}:{activation_id}:{item.attempts}"
            if (
                item.lease.world_id != expected_world
                or item.lease.resource_id != spec.authoritative_resource
                or item.lease.holder_id != expected_holder
                or item.lease.action_id != item.active_command_id
            ):
                raise SchedulerError("Restored authority lease is not bound to its command.")
        if len(active_primary) > 1:
            raise SchedulerError("Restored scheduler splits the primary control token.")
        if state.status is WorkflowStatus.RUNNING and len(active_primary) != 1:
            raise SchedulerError("A RUNNING scheduler must carry exactly one primary token.")
        if state.status is WorkflowStatus.CREATED and (
            active_primary or any(item.attempts for item in state.activations)
        ):
            raise SchedulerError("A CREATED scheduler cannot contain admitted history.")
        if state.status not in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING} and active_primary:
            raise SchedulerError("A terminal scheduler cannot carry a primary control token.")
        if state.pending_service_failure is not None:
            failure = records.get(state.pending_service_failure.activation_id)
            if failure is None or failure.status is not ActivationStatus.FAILED:
                raise SchedulerError("Pending service failure does not name a failed service.")
            if specs[failure.activation_id].lane is not ActivationLane.SERVICE:
                raise SchedulerError("Pending service failure names a primary activation.")

    def _capture_image(self) -> _SchedulerImage:
        return _SchedulerImage(
            graph=self.graph,
            status=self.status,
            records={
                activation_id: _ActivationRecord(
                    status=record.status,
                    attempts=record.attempts,
                    active_command_id=record.active_command_id,
                    active_graph_revision=record.active_graph_revision,
                    lease=record.lease,
                    last_outcome=record.last_outcome,
                )
                for activation_id, record in self._records.items()
            },
            loop_iterations=dict(self._loop_iterations),
            terminal_activation=self._terminal_activation,
            terminal_outcome=self._terminal_outcome,
            pending_service_failure=self._pending_service_failure,
            processed_event_digests=dict(self._processed_event_digests),
            state_revision=self._state_revision,
            recovery_pending_services=tuple(self._recovery_pending_services),
            recovery_service_inflight=self._recovery_service_inflight,
            recovery_primary=self._recovery_primary,
            recovery_primary_emitted=self._recovery_primary_emitted,
        )

    def _restore_image(self, image: _SchedulerImage) -> None:
        current_leases = {
            record.lease.lease_id: record.lease
            for record in self._records.values()
            if record.lease is not None
        }
        restored_leases = {
            record.lease.lease_id: record.lease
            for record in image.records.values()
            if record.lease is not None
        }
        for lease_id, lease in current_leases.items():
            if lease_id not in restored_leases:
                self.authority_registry.release(lease)
        for lease in restored_leases.values():
            self.authority_registry.restore(lease)
        self.graph = image.graph
        self.status = image.status
        self._records = image.records
        self._loop_iterations = image.loop_iterations
        self._terminal_activation = image.terminal_activation
        self._terminal_outcome = image.terminal_outcome
        self._pending_service_failure = image.pending_service_failure
        self._processed_event_digests = image.processed_event_digests
        self._state_revision = image.state_revision
        self._recovery_pending_services = list(image.recovery_pending_services)
        self._recovery_service_inflight = image.recovery_service_inflight
        self._recovery_primary = image.recovery_primary
        self._recovery_primary_emitted = image.recovery_primary_emitted

    def _persist_state(
        self, *, reason: str, metadata: dict[str, Any] | None = None
    ) -> None:
        if self._state_sink is not None:
            self._state_sink.append(
                self.export_state(), reason=reason, metadata=metadata
            )

    def _commit_mutation(
        self,
        reason: str,
        mutate: Callable[[], _MutationResult],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> _MutationResult:
        before = self._capture_image()
        try:
            result = mutate()
            self._state_revision += 1
            self._persist_state(reason=reason, metadata=metadata)
            return result
        except Exception:
            self._restore_image(before)
            raise

    def _spec(self, activation_id: str):
        for spec in self.graph.spec.activations:
            if spec.activation_id == activation_id:
                return spec
        raise SchedulerError(f"Unknown activation {activation_id!r}.")


__all__ = [
    "ActivationCommand",
    "ActivationRecordState",
    "ActivationScheduler",
    "ActivationSnapshot",
    "ActivationStatus",
    "AuthorityConflictError",
    "AuthorityLease",
    "AuthorityLeaseState",
    "AuthorityRegistry",
    "PendingServiceFailureState",
    "ProcessedEventState",
    "SchedulerError",
    "SchedulerState",
    "SchedulerStateSink",
    "TypedEventBus",
    "WorkflowSnapshot",
    "WorkflowStatus",
]
