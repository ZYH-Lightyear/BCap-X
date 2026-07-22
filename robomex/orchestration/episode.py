"""Episode-scoped orchestrator for the RoboMEx v2 baseline.

This is the integration boundary for intents, compiled workflows, actors,
typed events, append-only artifacts, embodied state, and authoritative effect
leases.  It intentionally does not call the v1 Session or graph executor.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robomex.authoring.monitoring import (
    MonitorCompiler,
    MonitorProgramSpec,
    MonitorRuntime,
)
from robomex.data import (
    STATE_TRANSITION_SCHEMA_MODELS,
    AdmissionFreshnessContext,
    AdmissionPurpose,
    AttachmentStatus,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    InputAdmission,
    ResolvedArtifactRef,
    SchemaRegistry,
    StateArtifactRef,
    StateCommitReceipt,
    StateTransitionProposal,
    StateTransitionProposalWire,
    StateTransitionRejected,
    core_schema_registry,
)
from robomex.elastic import (
    ACTION_SNAPSHOT_RUNNER_REF,
    ACTION_SNAPSHOT_SCHEMA_ID,
    ActivationLane,
    ArtifactBinding,
    CompiledElasticGraph,
    EffectScope,
    ExternalBinding,
    LifecycleScope,
    RunnerKind,
)
from robomex.elastic.graph_patch import (
    ComposableFrontier,
    GraphPatchCoordinator,
    GraphPatchProposal,
    PatchReceipt,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorNotFoundError,
    ActorProfile,
    ActorRegistry,
    ActorState,
    InvocationSpec,
)
from robomex.orchestration.arena import (
    ARENA_SCHEMA_MODELS,
    ArenaBinding,
    ArenaBindingRegistry,
    ArenaCandidateSpec,
    ArenaConsumptionLedger,
    ArenaContext,
    ArenaHypothesisSet,
    ArenaLedgerIntegrityError,
    ArenaPolicy,
    ArenaResult,
    ArenaStaleContextError,
    CandidateStatus,
    HypothesisAdapter,
    HypothesisHardGate,
    RiskReport,
    RuntimeArenaContextGuard,
    RuntimeMotionPromotionAuthority,
    ShadowBackendRegistry,
    SwarmArena,
)
from robomex.orchestration.intent import (
    IntentOutcome,
    IntentStatus,
    SubgoalIntent,
)
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerInvoker,
    ManagerSessionStatus,
    ManagerSignal,
    ManagerSnapshot,
    ManagerStep,
    SwarmManagerSession,
)
from robomex.orchestration.manager_store import ManagerSessionLedger
from robomex.orchestration.run_budget import (
    RunBudgetAuthority,
    RunBudgetExceededError,
    RunBudgetOperationStatus,
    RunBudgetReservation,
    RunBudgetVector,
)
from robomex.orchestration.workflow_store import (
    ArtifactRefRecord,
    WorkflowDescriptor,
    WorkflowDescriptorError,
    WorkflowDescriptorStore,
)
from robomex.runtime.action_protocol import (
    ACTION_PROTOCOL_SCHEMA_MODELS,
    ActionSpec,
    AdmissionSnapshot,
    ExecutionReceipt,
    ExecutionStatus,
    GripperCommand,
    GripperMode,
    MotionPlan,
    WorldKind,
    validate_action_spec,
)
from robomex.runtime.activation import (
    ActivationCommand,
    ActivationScheduler,
    AuthorityRegistry,
    SchedulerError,
    WorkflowSnapshot,
    WorkflowStatus,
)
from robomex.runtime.authority import (
    ActionBackend,
    ActionEvidenceRecorder,
    ActionSupervisor,
    AdmissionRejectedError,
    FeasibilityChecker,
    JsonlActionWAL,
    SealedActionRunner,
    validate_monitor_backend_compatibility,
)
from robomex.runtime.event_log import PersistentTypedEventBus
from robomex.runtime.events import (
    ActionOutcome,
    ActionStatus,
    ArenaDecision,
    ArtifactPublished,
    ControlOutcome,
    FindingSeverity,
    GraphPatchOutcome,
    LifecycleEvent,
    LifecycleTransition,
    ManagerInvocationOutcome,
    MonitorFinding,
    MonitorFindingKind,
    NodeOutcomeEvent,
    PatchRequest,
    RuntimeEventBase,
    ServiceOutcome,
    ServiceStatus,
    StateProposal,
    dump_runtime_event,
    parse_runtime_event,
)
from robomex.runtime.evidence_recorder import (
    EpisodeActionEvidenceRecorder,
    decode_evidence_ref,
    install_action_evidence_schemas,
)
from robomex.runtime.observation import ObservationRegistry, ObservationStream
from robomex.runtime.scheduler_store import (
    JsonlSchedulerStateStore,
    SchedulerStoreIntegrityError,
)
from robomex.runtime.service_delivery import (
    ServiceDeliveryIntegrityError,
    ServiceDeliveryLedger,
    ServiceDeliveryReservation,
    ServiceSubscriptionRecord,
    runtime_event_digest,
)


class EpisodeRuntimeError(RuntimeError):
    """A v2 episode invariant or dispatch contract was violated."""


_EMBODIED_STATE_REDUCER_RUNNER = "robomex.runtime.embodied_state_reducer"
_STATE_PROPOSAL_SCHEMA = "robomex.state_transition_proposal.v1"
_STATE_RECEIPT_SCHEMA = "robomex.state_commit_receipt.v1"


def install_runtime_schemas(registry: SchemaRegistry) -> SchemaRegistry:
    """Install every trusted v2 runtime payload validator and return the registry."""

    for schema_id, model in ACTION_PROTOCOL_SCHEMA_MODELS.items():
        registry.ensure(schema_id, model)
    install_action_evidence_schemas(registry)
    for schema_id, model in STATE_TRANSITION_SCHEMA_MODELS.items():
        registry.ensure(schema_id, model)
    registry.ensure("robomex.graph_patch_receipt.v1", PatchReceipt)
    registry.ensure("robomex.swarm_manager_session.v1", SwarmManagerSession)
    registry.ensure("robomex.monitor_program.v1", MonitorProgramSpec)
    for schema_id, model in ARENA_SCHEMA_MODELS.items():
        registry.ensure(schema_id, model)
    try:
        from robomex.manipulation.bowl_place import BOWL_PLACE_SCHEMA_MODELS
    except ImportError:
        return registry
    for schema_id, model in BOWL_PLACE_SCHEMA_MODELS.items():
        registry.ensure(schema_id, model)
    return registry


@dataclass(frozen=True)
class ArtifactEmission:
    port: str
    schema_id: str
    payload: Mapping[str, Any]
    lineage: tuple[ResolvedArtifactRef, ...] = ()


@dataclass(frozen=True)
class InvocationUsage:
    """Trusted provider measurement for one completed invocation boundary.

    ``None`` usage remains deliberately distinct from zero: it means the
    runtime cannot prove actual consumption and must charge the full reserved
    grant.  A present value is exact and is checked against both the invocation
    fingerprint and the reserved upper bound before settlement.
    """

    invocation_fingerprint: str
    model_calls: int
    tokens: int
    wall_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_fingerprint, str) or not (
            self.invocation_fingerprint.startswith("sha256:")
            and len(self.invocation_fingerprint) == 71
        ):
            raise ValueError("usage invocation_fingerprint must be canonical sha256")
        if (
            isinstance(self.model_calls, bool)
            or not isinstance(self.model_calls, int)
            or self.model_calls < 0
        ):
            raise ValueError("usage model_calls must be a non-negative integer")
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0:
            raise ValueError("usage tokens must be a non-negative integer")
        if (
            isinstance(self.wall_time_s, bool)
            or not isinstance(self.wall_time_s, (int, float))
            or not math.isfinite(self.wall_time_s)
            or self.wall_time_s < 0
        ):
            raise ValueError("usage wall_time_s must be finite and non-negative")
        object.__setattr__(self, "wall_time_s", float(self.wall_time_s))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "invocation_fingerprint": self.invocation_fingerprint,
            "model_calls": self.model_calls,
            "tokens": self.tokens,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InvocationUsage:
        return cls(
            invocation_fingerprint=str(value["invocation_fingerprint"]),
            model_calls=value["model_calls"],
            tokens=value["tokens"],
            wall_time_s=value["wall_time_s"],
        )


@dataclass(frozen=True)
class ActivationExecutionResult:
    """Provider-neutral result returned by one activation actor."""

    outcome: ControlOutcome | None = None
    artifacts: tuple[ArtifactEmission, ...] = ()
    reason: str = ""
    usage: InvocationUsage | None = None


@dataclass(frozen=True)
class ServiceEventResult:
    """Typed, non-control result of one subscribed service event delivery."""

    artifacts: tuple[ArtifactEmission, ...] = ()
    events: tuple[RuntimeEventBase, ...] = ()
    usage: InvocationUsage | None = None


@dataclass(frozen=True)
class ServiceDeliveryReport:
    """Observable result returned by :meth:`EpisodeRuntime.pump_service_events`."""

    workflow_id: str
    activation_id: str
    event_id: str
    delivery_id: str
    invocation_id: str
    status: str
    artifact_ids: tuple[str, ...] = ()
    emitted_event_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ScheduledInvocation:
    command: ActivationCommand
    admission: InputAdmission
    actor_id: str


@dataclass(frozen=True)
class ManagerRuntimeResult:
    step: ManagerStep
    patch_receipt: PatchReceipt | None = None


@dataclass(frozen=True)
class ArenaRuntimeResult:
    result: ArenaResult
    hypothesis_refs: Mapping[str, ResolvedArtifactRef]
    result_ref: ResolvedArtifactRef | None


@dataclass(frozen=True)
class _ActiveServiceSubscription:
    record: ServiceSubscriptionRecord
    actor_id: str
    command: ActivationCommand


@dataclass
class _WorkflowRuntime:
    intent: SubgoalIntent
    scheduler: ActivationScheduler
    external_refs: dict[str, ResolvedArtifactRef]
    actor_ids: dict[str, str]
    patch_coordinator: GraphPatchCoordinator | None = None
    manager_session: SwarmManagerSession | None = None
    manager_invoker: ManagerInvoker | None = None
    manager_snapshot_sequence: int = 0
    pending_manager_signals: list[tuple[ManagerSignal, Mapping[str, Any]]] = field(
        default_factory=list
    )
    service_subscriptions: dict[str, _ActiveServiceSubscription] = field(default_factory=dict)
    closed: bool = False
    recovered: bool = False


class EpisodeRuntime:
    """Run multiple intent workflows over one episode-owned state/data plane."""

    def __init__(
        self,
        *,
        episode_id: str,
        episode_root: str | Path,
        actors: ActorRegistry,
        action_backends: Mapping[tuple[str, str], ActionBackend] | None = None,
        feasibility_checkers: Mapping[tuple[str, str], FeasibilityChecker] | None = None,
        observation_registry: ObservationRegistry | None = None,
        schema_registry: SchemaRegistry | None = None,
        freshness_context_provider: (
            Callable[[AdmissionPurpose], AdmissionFreshnessContext] | None
        ) = None,
        action_evidence_recorder: ActionEvidenceRecorder | None = None,
        authority_lock_root: str | Path | None = None,
        max_action_snapshot_age_s: float | None = 5.0,
        strict_physical_evidence: bool = True,
        shadow_backends: ShadowBackendRegistry | None = None,
        arena_bindings: Sequence[ArenaBinding] = (),
        run_budget_authority: RunBudgetAuthority | None = None,
        allowed_initial_graph_digests: frozenset[str] | None = None,
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must not be empty")
        self.episode_id = episode_id
        self.episode_root = Path(episode_root)
        if run_budget_authority is not None and not isinstance(
            run_budget_authority, RunBudgetAuthority
        ):
            raise TypeError("run_budget_authority must be a RunBudgetAuthority")
        self.run_budget_authority = run_budget_authority
        if allowed_initial_graph_digests is None:
            self.allowed_initial_graph_digests: frozenset[str] | None = None
        else:
            normalized_graph_digests = frozenset(allowed_initial_graph_digests)
            if any(
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(character not in "0123456789abcdef" for character in value[7:])
                for value in normalized_graph_digests
            ):
                raise ValueError(
                    "allowed_initial_graph_digests must contain canonical sha256 digests"
                )
            self.allowed_initial_graph_digests = normalized_graph_digests
        self.actors = actors
        self.schema_registry = schema_registry or core_schema_registry()
        install_runtime_schemas(self.schema_registry)
        self._freshness_context_provider = freshness_context_provider
        # ``None`` is the production path: dispatch creates one recorder bound
        # to the exact workflow/action attempt. Injection is retained as a
        # narrow test seam for fault and protocol fakes.
        self._action_evidence_recorder = action_evidence_recorder
        self.data_plane = EpisodeDataPlane(
            self.episode_root,
            episode_id=episode_id,
            schema_registry=self.schema_registry,
            strict_schema_prefixes=("robomex.",),
        )
        self.state_reducer = EmbodiedStateReducer(
            self.episode_root,
            episode_id=episode_id,
            resolver=self.data_plane.resolver,
            strict_evidence=strict_physical_evidence,
        )
        self._state_commit_lock = threading.RLock()
        if observation_registry is not None and observation_registry.episode_id != episode_id:
            raise EpisodeRuntimeError("Injected ObservationRegistry belongs to another episode.")
        self.observations = observation_registry or ObservationRegistry(
            episode_id=episode_id,
            stream=ObservationStream(
                episode_id=episode_id,
                resolver=self.data_plane.resolver,
            ),
        )
        self.event_bus = PersistentTypedEventBus(self.episode_root / "runtime_events.v1.jsonl")
        self.service_delivery_ledger = ServiceDeliveryLedger(
            self.episode_root / "service_deliveries.v1.jsonl"
        )
        self._service_pump_lock = threading.RLock()
        self.manager_ledger = ManagerSessionLedger(self.episode_root / "manager_sessions.v1.jsonl")
        self.workflow_descriptors = WorkflowDescriptorStore(
            self.episode_root / "workflows.v1", episode_id=episode_id
        )
        self.scheduler_store = JsonlSchedulerStateStore(
            self.episode_root / "scheduler_states.v1.jsonl"
        )
        self.authority_registry = AuthorityRegistry()
        self.action_wal = JsonlActionWAL(self.episode_root / "action_wal.v1.jsonl")
        self.action_supervisor = ActionSupervisor(
            self.action_wal,
            require_certificate=True,
            interprocess_lock_root=(
                Path(authority_lock_root)
                if authority_lock_root is not None
                else self.episode_root.parent / ".robomex_authority_locks"
            ),
            max_snapshot_age_s=max_action_snapshot_age_s,
            reservation_validator=(
                lambda lease_id, world_id, resource_id, action_id: self.authority_registry.validate(
                    lease_id=lease_id,
                    world_id=world_id,
                    resource_id=resource_id,
                    action_id=action_id,
                )
            ),
        )
        self.orphan_action_receipts = self.action_supervisor.reconcile_orphans()
        self._action_backends = dict(action_backends or {})
        self._feasibility_checkers = dict(feasibility_checkers or {})
        if shadow_backends is not None and not isinstance(shadow_backends, ShadowBackendRegistry):
            raise TypeError("shadow_backends must be a ShadowBackendRegistry")
        self.shadow_backends = shadow_backends or ShadowBackendRegistry()
        self.arena_bindings = ArenaBindingRegistry(arena_bindings)
        self.arena_consumption_ledger = ArenaConsumptionLedger(
            self.episode_root / "arena_consumption.v1.jsonl"
        )
        self._arena_context_guard_token = object()
        self._profiles: dict[str, ActorProfile] = {}
        self._workflows: dict[str, _WorkflowRuntime] = {}
        self._closed = False

    def register_action_backend(
        self,
        *,
        world_id: str,
        resource_id: str,
        backend: ActionBackend,
        feasibility_checker: FeasibilityChecker,
    ) -> None:
        """Bind one trusted backend to an authoritative world/resource."""

        key = (str(world_id).strip(), str(resource_id).strip())
        if not all(key):
            raise ValueError("world_id and resource_id must not be empty")
        current = self._action_backends.get(key)
        if current is not None and current is not backend:
            raise EpisodeRuntimeError(f"Action backend for {key!r} is already registered.")
        self._action_backends[key] = backend
        checker = self._feasibility_checkers.get(key)
        if checker is not None and checker is not feasibility_checker:
            raise EpisodeRuntimeError(f"Feasibility checker for {key!r} is already registered.")
        self._feasibility_checkers[key] = feasibility_checker

    def create_swarm_arena(
        self,
        *,
        gates: Sequence[HypothesisHardGate] = (),
        adapters: Mapping[str, HypothesisAdapter] | None = None,
        policy: ArenaPolicy | None = None,
    ) -> SwarmArena:
        """Construct an Arena bound to this Episode's live trust roots.

        Callers choose proposal policies and optional advisory gates, but cannot
        replace artifact resolution, graph revision checks, durable budget
        consumption, feasibility checkers, or the manifest-admitted shadow
        backend registry.
        """

        return SwarmArena(
            self.actors,
            promotion_authority=RuntimeMotionPromotionAuthority(
                episode_id=self.episode_id,
                artifacts=self.data_plane,
                snapshot_providers={
                    key: backend.snapshot for key, backend in self._action_backends.items()
                },
                feasibility_checkers=self._feasibility_checkers,
            ),
            context_guard=RuntimeArenaContextGuard(
                episode_id=self.episode_id,
                current_revision=self._arena_graph_revision,
                binding_token=self._arena_context_guard_token,
            ),
            consumption_ledger=self.arena_consumption_ledger,
            shadow_backends=self.shadow_backends,
            gates=gates,
            adapters=adapters,
            policy=policy,
        )

    def register_arena_binding(self, binding: ArenaBinding) -> None:
        """Admit one trusted graph Arena implementation before workflows open."""

        if self._closed:
            raise EpisodeRuntimeError("Cannot register Arena bindings after the episode is closed.")
        if self._workflows:
            raise EpisodeRuntimeError(
                "Arena bindings are immutable after the first workflow opens."
            )
        self.arena_bindings.register(binding)

    def register_actor_profile(self, runner_ref: str, profile: ActorProfile) -> None:
        if self._closed:
            raise EpisodeRuntimeError("Cannot register actors after the episode is closed.")
        key = str(runner_ref).strip()
        if not key:
            raise ValueError("runner_ref must not be empty")
        existing = self._profiles.get(key)
        if existing is not None and existing != profile:
            raise EpisodeRuntimeError(f"Runner {key!r} already has a different ActorProfile.")
        self._profiles[key] = profile

    def open_workflow(
        self,
        *,
        intent: SubgoalIntent,
        graph: CompiledElasticGraph,
        workflow_id: str | None = None,
        external_refs: Mapping[str, ResolvedArtifactRef | Mapping[str, Any]] | None = None,
        frontier: ComposableFrontier | None = None,
        manager_session: SwarmManagerSession | None = None,
        manager_invoker: ManagerInvoker | None = None,
    ) -> str:
        if self._closed:
            raise EpisodeRuntimeError("Episode is closed.")
        workflow_id = workflow_id or f"wf_{uuid.uuid4().hex}"
        self._validate_initial_graph_digest(graph.digest)
        self._validate_system_action_contracts(graph)
        self._validate_arena_contracts(graph)
        self._validate_graph_profiles(graph)
        if (manager_session is None) != (manager_invoker is None):
            raise EpisodeRuntimeError(
                "manager_session and manager_invoker must be supplied together."
            )
        if manager_session is not None and (
            manager_session.episode_id != self.episode_id
            or manager_session.workflow_id != workflow_id
            or manager_session.intent_id != intent.intent_id
        ):
            raise EpisodeRuntimeError(
                "Manager session identity does not match the workflow intent."
            )
        normalized_external = {
            key: ResolvedArtifactRef.from_any(value) for key, value in (external_refs or {}).items()
        }
        for ref in normalized_external.values():
            self.data_plane.resolve(ref)
        desired_descriptor = WorkflowDescriptor(
            episode_id=self.episode_id,
            workflow_id=workflow_id,
            intent=intent,
            initial_graph_id=graph.spec.graph_id,
            initial_graph_revision=graph.spec.revision,
            initial_graph_digest=graph.digest,
            external_refs={
                name: ArtifactRefRecord.from_ref(ref) for name, ref in normalized_external.items()
            },
            frontier=frontier,
            manager_session_id=(
                manager_session.session_id if manager_session is not None else None
            ),
        )
        try:
            persisted_descriptor = self.workflow_descriptors.load(workflow_id)
            existing_commit = self.scheduler_store.latest(
                episode_id=self.episode_id, workflow_id=workflow_id
            )
            if persisted_descriptor is None and existing_commit is not None:
                raise EpisodeRuntimeError(
                    "Scheduler state exists without an immutable workflow descriptor."
                )
            descriptor = self.workflow_descriptors.bind(desired_descriptor)
        except (WorkflowDescriptorError, SchedulerStoreIntegrityError) as exc:
            raise EpisodeRuntimeError(str(exc)) from exc

        if workflow_id in self._workflows:
            current = self._workflows[workflow_id]
            if descriptor != desired_descriptor or current.intent != intent:
                raise EpisodeRuntimeError(f"Workflow id {workflow_id!r} is already bound.")
            if manager_invoker is not None and current.manager_invoker is not manager_invoker:
                raise EpisodeRuntimeError(
                    "Workflow is already attached to a different Manager invoker."
                )
            return workflow_id

        if existing_commit is not None:
            return self._recover_bound_workflow(
                descriptor,
                manager_invoker=manager_invoker,
                supplied_manager_session=manager_session,
            )

        self.data_plane.open_workflow(workflow_id)
        bound_manager = self._restore_manager_session(
            descriptor,
            supplied=manager_session,
            require_existing=False,
        )
        scheduler = ActivationScheduler(
            episode_id=self.episode_id,
            workflow_id=workflow_id,
            graph=graph,
            event_bus=self.event_bus,
            authority_registry=self.authority_registry,
            state_sink=self.scheduler_store,
        )
        workflow_runtime = _WorkflowRuntime(
            intent=intent,
            scheduler=scheduler,
            external_refs=normalized_external,
            actor_ids={},
            manager_session=bound_manager,
            manager_invoker=manager_invoker,
            manager_snapshot_sequence=(
                len(bound_manager.receipts) if bound_manager is not None else 0
            ),
        )
        if frontier is not None:
            workflow_runtime.patch_coordinator = GraphPatchCoordinator(
                scheduler=scheduler,
                frontier=frontier,
            )
        scheduler.start()
        self._workflows[workflow_id] = workflow_runtime
        return workflow_id

    def recover_workflow(
        self,
        workflow_id: str,
        *,
        manager_invoker: ManagerInvoker | None = None,
    ) -> str:
        """Rehydrate a durable workflow without reconstructing its input graph.

        Actor providers and a Manager invoker are process-local dependencies;
        graph/state/artifact/lease identities come only from the episode logs.
        """

        if self._closed:
            raise EpisodeRuntimeError("Episode is closed.")
        if workflow_id in self._workflows:
            current = self._workflows[workflow_id]
            if manager_invoker is not None:
                if (
                    current.manager_invoker is not None
                    and current.manager_invoker is not manager_invoker
                ):
                    raise EpisodeRuntimeError(
                        "Workflow is already attached to a different Manager invoker."
                    )
                current.manager_invoker = manager_invoker
            return workflow_id
        try:
            descriptor = self.workflow_descriptors.load(workflow_id)
            if descriptor is None:
                raise EpisodeRuntimeError(f"Workflow {workflow_id!r} has no durable descriptor.")
            self._validate_initial_graph_digest(descriptor.initial_graph_digest)
            if (
                self.scheduler_store.latest(episode_id=self.episode_id, workflow_id=workflow_id)
                is None
            ):
                raise EpisodeRuntimeError(
                    f"Workflow {workflow_id!r} has no durable scheduler state."
                )
        except (WorkflowDescriptorError, SchedulerStoreIntegrityError) as exc:
            raise EpisodeRuntimeError(str(exc)) from exc
        return self._recover_bound_workflow(
            descriptor,
            manager_invoker=manager_invoker,
            supplied_manager_session=None,
        )

    def _recover_bound_workflow(
        self,
        descriptor: WorkflowDescriptor,
        *,
        manager_invoker: ManagerInvoker | None,
        supplied_manager_session: SwarmManagerSession | None,
    ) -> str:
        workflow_id = descriptor.workflow_id
        self._validate_initial_graph_digest(descriptor.initial_graph_digest)
        preexisting_leases = set(self.authority_registry.active())
        try:
            history = self.scheduler_store.history(
                episode_id=self.episode_id, workflow_id=workflow_id
            )
            if not history:
                raise EpisodeRuntimeError("Cannot recover an empty scheduler history.")
            initial = history[0].state
            if (
                initial.graph.graph_id != descriptor.initial_graph_id
                or initial.graph.revision != descriptor.initial_graph_revision
                or initial.graph_digest != descriptor.initial_graph_digest
            ):
                raise EpisodeRuntimeError(
                    "Workflow descriptor does not bind the scheduler's initial graph."
                )
            scheduler = self.scheduler_store.recover_scheduler(
                episode_id=self.episode_id,
                workflow_id=workflow_id,
                event_bus=self.event_bus,
                authority_registry=self.authority_registry,
            )
            patch_coordinator = self.scheduler_store.recover_patch_coordinator(scheduler=scheduler)
        except (SchedulerError, ValueError) as exc:
            raise EpisodeRuntimeError(str(exc)) from exc

        restored_leases = {
            item.lease.to_lease()
            for item in scheduler.export_state().activations
            if item.lease is not None
        }
        try:
            return self._bind_recovered_workflow(
                descriptor,
                scheduler=scheduler,
                patch_coordinator=patch_coordinator,
                manager_invoker=manager_invoker,
                supplied_manager_session=supplied_manager_session,
            )
        except Exception:
            # Process-local dependencies (profiles, Manager invoker/history,
            # external artifact availability) are validated after scheduler
            # reconstruction.  A failure there must roll back only the exact
            # capabilities newly installed by this recovery attempt.
            for lease in restored_leases - preexisting_leases:
                if self.authority_registry.validate(
                    lease_id=lease.lease_id,
                    world_id=lease.world_id,
                    resource_id=lease.resource_id,
                    action_id=lease.action_id,
                    holder_id=lease.holder_id,
                ):
                    self.authority_registry.release(lease)
            raise

    def _bind_recovered_workflow(
        self,
        descriptor: WorkflowDescriptor,
        *,
        scheduler: ActivationScheduler,
        patch_coordinator: GraphPatchCoordinator | None,
        manager_invoker: ManagerInvoker | None,
        supplied_manager_session: SwarmManagerSession | None,
    ) -> str:
        workflow_id = descriptor.workflow_id
        self._validate_system_action_contracts(scheduler.graph)
        self._validate_arena_contracts(scheduler.graph)
        self._validate_graph_profiles(scheduler.graph)
        if descriptor.frontier is None and patch_coordinator is not None:
            raise EpisodeRuntimeError(
                "Recovered scheduler contains an undeclared elastic frontier."
            )
        if descriptor.frontier is not None and patch_coordinator is None:
            if (
                scheduler.graph.spec.revision != descriptor.initial_graph_revision
                or scheduler.graph.digest != descriptor.initial_graph_digest
            ):
                raise EpisodeRuntimeError(
                    "Patched graph is missing its durable frontier/commit chain."
                )
            patch_coordinator = GraphPatchCoordinator(
                scheduler=scheduler,
                frontier=descriptor.frontier,
            )

        external_refs = {name: record.to_ref() for name, record in descriptor.external_refs.items()}
        for ref in external_refs.values():
            self.data_plane.resolve(ref)
        data_status = self.data_plane.workflows.get(workflow_id)
        if data_status == "closed":
            if scheduler.status in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}:
                raise EpisodeRuntimeError(
                    "Data plane is closed while the recovered scheduler is active."
                )
            closed = True
        else:
            self.data_plane.open_workflow(workflow_id)
            closed = False

        manager_session = self._restore_manager_session(
            descriptor,
            supplied=supplied_manager_session,
            require_existing=descriptor.manager_session_id is not None,
        )
        if manager_session is not None and manager_invoker is None:
            # The graph can recover and execute deterministic work without an
            # LLM object; a future exceptional Manager wake remains fail-closed
            # until attach_manager supplies an invoker.
            manager_invoker = None
        runtime = _WorkflowRuntime(
            intent=descriptor.intent,
            scheduler=scheduler,
            external_refs=external_refs,
            actor_ids={},
            patch_coordinator=patch_coordinator,
            manager_session=manager_session,
            manager_invoker=manager_invoker,
            manager_snapshot_sequence=(
                len(manager_session.receipts) if manager_session is not None else 0
            ),
            closed=closed,
            recovered=True,
        )
        if scheduler.status is WorkflowStatus.CREATED:
            scheduler.start()
        runtime.pending_manager_signals.extend(self._recover_pending_manager_signals(runtime))
        self._workflows[workflow_id] = runtime
        return workflow_id

    def _restore_manager_session(
        self,
        descriptor: WorkflowDescriptor,
        *,
        supplied: SwarmManagerSession | None,
        require_existing: bool,
    ) -> SwarmManagerSession | None:
        session_id = descriptor.manager_session_id
        if session_id is None:
            if supplied is not None:
                raise EpisodeRuntimeError("Workflow descriptor does not declare a Manager session.")
            return None
        if supplied is not None and (
            supplied.session_id != session_id
            or supplied.episode_id != self.episode_id
            or supplied.workflow_id != descriptor.workflow_id
            or supplied.intent_id != descriptor.intent.intent_id
        ):
            raise EpisodeRuntimeError(
                "Supplied Manager session does not match the workflow descriptor."
            )
        latest = self.manager_ledger.latest(session_id)
        if latest is None:
            if require_existing or supplied is None:
                raise EpisodeRuntimeError(
                    "Workflow descriptor references a missing Manager session history."
                )
            self.manager_ledger.append(supplied)
            return supplied
        if (
            latest.episode_id != self.episode_id
            or latest.workflow_id != descriptor.workflow_id
            or latest.intent_id != descriptor.intent.intent_id
        ):
            raise EpisodeRuntimeError(
                "Persisted Manager session identity differs from the workflow."
            )
        return latest

    def configure_frontier(
        self, workflow_id: str, frontier: ComposableFrontier
    ) -> GraphPatchCoordinator:
        runtime = self._workflow(workflow_id)
        if runtime.patch_coordinator is not None:
            if runtime.patch_coordinator.frontier == frontier:
                return runtime.patch_coordinator
            raise EpisodeRuntimeError("Workflow already has a different elastic frontier.")
        runtime.patch_coordinator = GraphPatchCoordinator(
            scheduler=runtime.scheduler,
            frontier=frontier,
        )
        return runtime.patch_coordinator

    def attach_manager(
        self,
        workflow_id: str,
        *,
        session: SwarmManagerSession,
        invoker: ManagerInvoker,
    ) -> None:
        runtime = self._workflow(workflow_id)
        if (
            session.episode_id != self.episode_id
            or session.workflow_id != workflow_id
            or session.intent_id != runtime.intent.intent_id
        ):
            raise EpisodeRuntimeError("Manager session identity does not match workflow.")
        if runtime.manager_session is not None:
            current = runtime.manager_session
            if (
                current.session_id == session.session_id
                and current.episode_id == session.episode_id
                and current.workflow_id == session.workflow_id
                and current.intent_id == session.intent_id
                and runtime.manager_invoker is None
            ):
                runtime.manager_invoker = invoker
                self._drain_manager_wakeups(runtime)
                return
            if current == session and runtime.manager_invoker is invoker:
                return
            raise EpisodeRuntimeError("Workflow already has a different Manager session.")
        latest = self.manager_ledger.latest(session.session_id)
        if latest is None:
            self.manager_ledger.append(session)
        elif latest != session:
            raise EpisodeRuntimeError("Persisted Manager session differs from attachment.")
        runtime.manager_session = session
        runtime.manager_invoker = invoker
        self._drain_manager_wakeups(runtime)

    def apply_graph_patch(self, workflow_id: str, proposal: GraphPatchProposal) -> PatchReceipt:
        runtime = self._workflow(workflow_id)
        coordinator = runtime.patch_coordinator
        if coordinator is None:
            raise EpisodeRuntimeError("Workflow has no configured ComposableFrontier.")
        try:
            self._validate_fragment_profiles(proposal)
        except EpisodeRuntimeError as exc:
            receipt = coordinator.reject_preflight(proposal, reason=str(exc))
        else:
            receipt = coordinator.apply(proposal)
        record = self._publish_system_artifact(
            runtime,
            activation_id="graph_patch",
            attempt=max(receipt.after_revision, receipt.before_revision),
            port="receipt",
            schema="robomex.graph_patch_receipt.v1",
            payload=receipt.model_dump(mode="json"),
        )
        runtime.scheduler.on_event(
            GraphPatchOutcome(
                episode_id=self.episode_id,
                workflow_id=workflow_id,
                source="graph_patch_coordinator",
                patch_id=receipt.patch_id,
                slot_id=receipt.slot_id,
                operation=receipt.operation.value,
                accepted=receipt.accepted,
                before_revision=receipt.before_revision,
                after_revision=receipt.after_revision,
                before_digest=receipt.before_digest,
                after_digest=receipt.after_digest,
                reason_codes=tuple(code.value for code in receipt.reason_codes),
                receipt_ref=record.artifact_id,
            )
        )
        return receipt

    def invoke_manager(
        self,
        workflow_id: str,
        *,
        signal: ManagerSignal,
        triggering_event: Mapping[str, Any] | None = None,
        frontier_id: str | None = None,
        candidate_cards: tuple[Mapping[str, Any], ...] = (),
        catalog_refs: tuple[str, ...] = (),
    ) -> ManagerRuntimeResult:
        runtime = self._workflow(workflow_id)
        session = runtime.manager_session
        invoker = runtime.manager_invoker
        if session is None or invoker is None:
            raise EpisodeRuntimeError("Workflow has no attached bounded Manager session.")
        runtime.manager_snapshot_sequence += 1
        artifacts = tuple(
            record.artifact_id
            for record in self.data_plane.artifacts
            if record.workflow_id == workflow_id
        )
        snapshot = ManagerSnapshot(
            snapshot_id=(f"{workflow_id}-manager-snapshot-{runtime.manager_snapshot_sequence}"),
            episode_id=self.episode_id,
            workflow_id=workflow_id,
            graph_id=runtime.scheduler.graph.spec.graph_id,
            graph_revision=runtime.scheduler.graph.spec.revision,
            state_revision=self.state_reducer.state.revision,
            frontier_id=frontier_id,
            triggering_event=dict(triggering_event or {}),
            compact_state=self.state_reducer.state.to_mapping(),
            candidate_cards=tuple(dict(card) for card in candidate_cards),
            artifact_refs=artifacts,
            catalog_refs=catalog_refs,
        )
        manager_budget: RunBudgetReservation | None = None
        remaining = session.remaining
        initial_call = signal is ManagerSignal.INTENT_AUTHORING
        should_attempt = (
            session.status is ManagerSessionStatus.ACTIVE
            and remaining.tokens > 0
            and (
                (initial_call and remaining.initial_calls > 0)
                or (
                    not initial_call and session.should_wake(signal) and remaining.reactivations > 0
                )
            )
        )
        if self.run_budget_authority is not None and should_attempt:
            token_grant = min(session.limits.max_tokens_per_call, remaining.tokens)
            invoker_owns_model_budget = bool(getattr(invoker, "manages_run_model_budget", False))
            manager_request = RunBudgetVector(
                model_calls=0 if invoker_owns_model_budget else 1,
                tokens=0 if invoker_owns_model_budget else token_grant,
                recoveries=0 if initial_call else 1,
            )
            manager_operation_id = (
                f"manager:{session.session_id}:r{session.record_revision}:"
                f"{signal.value}:{snapshot.snapshot_id}"
            )
            if not manager_request.is_zero:
                try:
                    manager_budget = self.run_budget_authority.reserve(
                        operation_id=manager_operation_id,
                        requested=manager_request,
                        binding={
                            "kind": "manager_invocation",
                            "session_id": session.session_id,
                            "session_revision": session.record_revision,
                            "signal": signal.value,
                            "snapshot_digest": snapshot.digest(),
                            "invoker_owns_model_budget": invoker_owns_model_budget,
                        },
                    )
                except RunBudgetExceededError as exc:
                    reason = f"run_budget_exhausted:{exc}"
                    step = ManagerStep(
                        session=session,
                        invoked=False,
                        signal=signal,
                        reason=reason,
                    )
                    event_id = f"budget-manager:{hashlib.sha256(manager_operation_id.encode()).hexdigest()}"
                    existing = self._event_by_id(event_id)
                    event = ManagerInvocationOutcome(
                        event_id=event_id,
                        episode_id=self.episode_id,
                        workflow_id=workflow_id,
                        source="run_budget_authority",
                        session_id=session.session_id,
                        session_revision=session.record_revision,
                        signal=signal.value,
                        invoked=False,
                        reason=reason,
                        triggering_event_id=self._triggering_event_id(triggering_event),
                        **({"timestamp": existing.timestamp} if existing is not None else {}),
                    )
                    runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))
                    return ManagerRuntimeResult(step=step)
        try:
            if initial_call:
                step = session.author(snapshot, invoker)
            else:
                step = session.wake(signal, snapshot, invoker)
        except Exception:
            if (
                manager_budget is not None
                and self.run_budget_authority is not None
                and manager_budget.status is RunBudgetOperationStatus.RESERVED
            ):
                self.run_budget_authority.complete(manager_budget)
            raise
        if manager_budget is not None and self.run_budget_authority is not None:
            if step.invoked:
                tokens = (
                    step.decision.tokens_used
                    if step.decision is not None
                    else manager_budget.requested.tokens
                )
                self.run_budget_authority.complete(
                    manager_budget,
                    settlement=RunBudgetVector(
                        model_calls=manager_budget.requested.model_calls,
                        tokens=(
                            min(tokens, manager_budget.requested.tokens)
                            if manager_budget.requested.tokens
                            else 0
                        ),
                        recoveries=manager_budget.requested.recoveries,
                    ),
                )
            elif manager_budget.status is RunBudgetOperationStatus.RESERVED:
                self.run_budget_authority.release(manager_budget)
        runtime.manager_session = step.session
        manager_record_ref = None
        if step.session.record_revision != session.record_revision:
            self.manager_ledger.append(step.session)
            record = self._publish_system_artifact(
                runtime,
                activation_id="swarm_manager",
                attempt=step.session.record_revision,
                port="session",
                schema=step.session.schema_version,
                payload=step.session.model_dump(mode="json"),
            )
            manager_record_ref = record.artifact_id
        runtime.scheduler.on_event(
            ManagerInvocationOutcome(
                episode_id=self.episode_id,
                workflow_id=workflow_id,
                source="episode_runtime",
                session_id=step.session.session_id,
                session_revision=step.session.record_revision,
                signal=signal.value,
                invoked=step.invoked,
                action=(step.decision.action.value if step.decision else None),
                reason=step.reason or None,
                manager_record_ref=manager_record_ref,
                triggering_event_id=self._triggering_event_id(triggering_event),
            )
        )
        patch_receipt = None
        if step.decision is not None:
            if (
                step.decision.action
                in {
                    ManagerAction.AUTHOR_SCAFFOLD,
                    ManagerAction.REPAIR_FRONTIER,
                }
                and "graph_patch" in step.decision.payload
            ):
                patch_receipt = self.apply_graph_patch(
                    workflow_id,
                    GraphPatchProposal.model_validate(step.decision.payload["graph_patch"]),
                )
            elif step.decision.action is ManagerAction.CLOSE:
                closed = step.session.close()
                self.manager_ledger.append(closed)
                runtime.manager_session = closed
        return ManagerRuntimeResult(step=step, patch_receipt=patch_receipt)

    def run_arena(
        self,
        workflow_id: str,
        *,
        arena: SwarmArena,
        context: ArenaContext,
        risk_report: RiskReport,
        candidates: tuple[ArenaCandidateSpec, ...],
        candidate_budget_remaining: int,
        recovery_safe: bool = False,
        run_binding_digest: str | None = None,
        publish_runtime_artifacts: bool = True,
    ) -> ArenaRuntimeResult:
        runtime = self._workflow(workflow_id)
        if (
            context.episode_id != self.episode_id
            or context.workflow_id != workflow_id
            or context.graph_id != runtime.scheduler.graph.spec.graph_id
            or context.graph_revision != runtime.scheduler.graph.spec.revision
        ):
            raise EpisodeRuntimeError("Arena context is stale or belongs to another workflow.")
        arena.assert_bound_to(
            episode_id=self.episode_id,
            artifacts=self.data_plane,
            context_guard_token=self._arena_context_guard_token,
            consumption_ledger=self.arena_consumption_ledger,
            shadow_backends=self.shadow_backends,
        )
        arena_budget: RunBudgetReservation | None = None
        _requested, potential_candidates, potential_shadow = arena.budget_request(
            risk_report=risk_report,
            candidates=candidates,
            candidate_budget_remaining=candidate_budget_remaining,
        )
        arena_operation_id = f"arena:{self.episode_id}:{workflow_id}:{context.arena_run_id}"
        potential_specs = tuple(candidates[:potential_candidates])
        if self.run_budget_authority is not None:
            for candidate in potential_specs:
                if candidate.profile.runner_kind == RunnerKind.CODING_WORKER.value and (
                    candidate.estimated_budget.model_calls < 1
                    or candidate.estimated_budget.tokens < 1
                    or candidate.estimated_budget.wall_time_ms < 1
                ):
                    raise EpisodeRuntimeError(
                        f"Arena coding candidate {candidate.candidate_id!r} must "
                        "declare positive model_calls, tokens, and wall_time_ms"
                    )
        potential_model_calls = sum(
            candidate.estimated_budget.model_calls for candidate in potential_specs
        )
        potential_tokens = sum(candidate.estimated_budget.tokens for candidate in potential_specs)
        potential_wall_time_s = (
            sum(candidate.estimated_budget.wall_time_ms for candidate in potential_specs) / 1000.0
        )
        if self.run_budget_authority is not None:
            candidate_bindings = []
            for candidate in candidates[:potential_candidates]:
                candidate_invocation = InvocationSpec(
                    invocation_id=(f"{context.arena_run_id}-{candidate.candidate_id}-invoke"),
                    objective=candidate.objective,
                    inputs=candidate.inputs,
                    output_contract=candidate.output_contract,
                    requested_capabilities=candidate.requested_capabilities,
                    requested_effects=candidate.requested_effects,
                    budget={
                        name: float(value)
                        for name, value in candidate.estimated_budget.model_dump(
                            mode="python"
                        ).items()
                    },
                    idempotency_key=(f"{context.arena_run_id}-{candidate.candidate_id}"),
                )
                candidate_bindings.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "strategy": candidate.strategy,
                        "effect_scope": candidate.effect_scope.value,
                        "profile_id": candidate.profile.profile_id,
                        "adapter_id": candidate.adapter_id,
                        "shadow_backend_id": candidate.shadow_backend_id,
                        "shadow_resource_id": candidate.shadow_resource_id,
                        "estimated_budget": candidate.estimated_budget.model_dump(mode="json"),
                        "invocation_fingerprint": candidate_invocation.fingerprint(),
                    }
                )
            try:
                arena_budget = self.run_budget_authority.reserve(
                    operation_id=arena_operation_id,
                    requested=RunBudgetVector(
                        model_calls=potential_model_calls,
                        tokens=potential_tokens,
                        wall_time_s=potential_wall_time_s,
                        candidates=potential_candidates,
                        shadow_rollouts=potential_shadow,
                    ),
                    binding={
                        "kind": "arena_run",
                        "context": context.model_dump(mode="json"),
                        "risk_report": risk_report.model_dump(mode="json"),
                        "candidates": candidate_bindings,
                        "candidate_budget_remaining": candidate_budget_remaining,
                    },
                )
            except RunBudgetExceededError as exc:
                reason = f"run_budget_exhausted:{exc}"
                event_id = f"budget-arena:{context.arena_run_id}"
                existing = self._event_by_id(event_id)
                event = ArenaDecision(
                    event_id=event_id,
                    episode_id=self.episode_id,
                    workflow_id=workflow_id,
                    source="run_budget_authority",
                    arena_run_id=context.arena_run_id,
                    slot_id=context.slot_id,
                    graph_revision=context.graph_revision,
                    considered_candidate_ids=(),
                    rejected_candidate_ids=tuple(
                        candidate.candidate_id for candidate in candidates
                    ),
                    selection_reason=reason,
                    **({"timestamp": existing.timestamp} if existing is not None else {}),
                )
                runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))
                raise
        try:
            result = arena.run(
                context=context,
                risk_report=risk_report,
                candidates=candidates,
                candidate_budget_remaining=candidate_budget_remaining,
                recovery_safe=recovery_safe,
                run_binding_digest=run_binding_digest,
                deadline_monotonic_s=self._run_deadline_monotonic_s(),
            )
        except Exception:
            if (
                arena_budget is not None
                and self.run_budget_authority is not None
                and arena_budget.status is not RunBudgetOperationStatus.COMPLETED
            ):
                # Arena may already have invoked candidates or a shadow backend;
                # an unclosed failure conservatively charges the reservation.
                self.run_budget_authority.complete(arena_budget)
            raise
        if arena_budget is not None and self.run_budget_authority is not None:
            candidate_by_id = {item.candidate_id: item for item in candidates}
            actual_specs = tuple(
                candidate_by_id[item.candidate_id]
                for item in result.candidate_results
                if item.provider_entered
            )
            actual_shadow = sum(
                candidate_by_id[item.candidate_id].effect_scope is EffectScope.SHADOW_WORLD
                for item in result.candidate_results
            )
            self.run_budget_authority.complete(
                arena_budget,
                settlement=RunBudgetVector(
                    model_calls=sum(item.estimated_budget.model_calls for item in actual_specs),
                    tokens=sum(item.estimated_budget.tokens for item in actual_specs),
                    wall_time_s=sum(item.estimated_budget.wall_time_ms for item in actual_specs)
                    / 1000.0,
                    candidates=result.candidate_budget_used,
                    shadow_rollouts=actual_shadow,
                ),
            )
        current_graph_id, current_graph_revision = self._arena_graph_revision(context)
        if current_graph_id != context.graph_id or current_graph_revision != context.graph_revision:
            raise EpisodeRuntimeError(
                "Arena graph revision changed after selection; refusing promotion publish."
            )
        for update in result.roster_updates:
            if runtime.patch_coordinator is not None:
                runtime.patch_coordinator.record_roster_update(update)
            else:
                runtime.scheduler.on_event(update)

        if not publish_runtime_artifacts:
            return ArenaRuntimeResult(
                result=result,
                hypothesis_refs={},
                result_ref=None,
            )

        hypothesis_refs: dict[str, ResolvedArtifactRef] = {}
        for candidate in result.candidate_results:
            if candidate.hypothesis is None:
                continue
            record = self._publish_system_artifact(
                runtime,
                activation_id="swarm_arena",
                attempt=runtime.scheduler.graph.spec.revision,
                port="hypothesis",
                schema=candidate.hypothesis.schema_version,
                payload=candidate.hypothesis.model_dump(mode="json"),
            )
            hypothesis_refs[candidate.candidate_id] = record.ref
        promotion_ref = None
        if result.promotion_receipt is not None:
            promotion = result.promotion_receipt
            promotion_record = self._publish_system_artifact(
                runtime,
                activation_id="swarm_arena",
                attempt=runtime.scheduler.graph.spec.revision,
                port="promotion_receipt",
                schema=promotion.schema_version,
                payload=promotion.model_dump(mode="json"),
                lineage=(promotion.action_spec_ref, promotion.snapshot_ref),
            )
            promotion_ref = promotion_record.ref
        result_record = self._publish_system_artifact(
            runtime,
            activation_id="swarm_arena",
            attempt=runtime.scheduler.graph.spec.revision,
            port="result",
            schema=result.schema_version,
            payload=result.model_dump(mode="json"),
            lineage=tuple(hypothesis_refs.values())
            + ((promotion_ref,) if promotion_ref is not None else ()),
        )
        selected_ref = hypothesis_refs.get(result.selected_candidate_id or "")
        rejected = tuple(
            candidate.candidate_id
            for candidate in result.candidate_results
            if candidate.status is not CandidateStatus.ACCEPTED
        )
        runtime.scheduler.on_event(
            ArenaDecision(
                episode_id=self.episode_id,
                workflow_id=workflow_id,
                source="swarm_arena",
                arena_run_id=result.arena_run_id,
                slot_id=context.slot_id,
                graph_revision=result.graph_revision,
                selected_candidate_id=result.selected_candidate_id,
                considered_candidate_ids=tuple(
                    candidate.candidate_id for candidate in result.candidate_results
                ),
                rejected_candidate_ids=rejected,
                selection_reason=result.selection_reason,
                selected_hypothesis_ref=(
                    selected_ref.artifact_id if selected_ref is not None else None
                ),
                evidence_refs=tuple(
                    ref
                    for ref in (
                        result_record.artifact_id,
                        promotion_ref.artifact_id if promotion_ref is not None else None,
                    )
                    if ref is not None
                ),
            )
        )
        return ArenaRuntimeResult(
            result=result,
            hypothesis_refs=hypothesis_refs,
            result_ref=result_record.ref,
        )

    def next_invocations(self, workflow_id: str) -> tuple[ScheduledInvocation, ...]:
        runtime = self._workflow(workflow_id)
        if runtime.closed:
            return ()
        invocations: list[ScheduledInvocation] = []
        commands = runtime.scheduler.recovery_commands()
        if not commands:
            commands = runtime.scheduler.next_commands()
        for command in commands:
            try:
                admission = self._admit_command_inputs(runtime, command)
                actor_id = self._actor_id(runtime, command)
            except Exception as exc:
                self._reject_unadmitted_command(runtime, command, exc)
                raise
            invocations.append(
                ScheduledInvocation(command=command, admission=admission, actor_id=actor_id)
            )
        return tuple(invocations)

    def _reject_unadmitted_command(
        self,
        runtime: _WorkflowRuntime,
        command: ActivationCommand,
        exc: Exception,
    ) -> None:
        """Release scheduler reservations when data admission fails."""

        reason = f"input_admission_failed:{type(exc).__name__}:{exc}"
        if command.lane is ActivationLane.SERVICE:
            runtime.scheduler.on_event(
                ServiceOutcome(
                    episode_id=self.episode_id,
                    workflow_id=command.workflow_id,
                    source="episode_data_plane",
                    activation_id=command.activation_id,
                    command_id=command.command_id,
                    attempt=command.attempt,
                    status=ServiceStatus.FAILED,
                    reason=reason,
                    graph_revision=command.graph_revision,
                )
            )
            return
        runtime.scheduler.on_event(
            NodeOutcomeEvent(
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="episode_data_plane",
                activation_id=command.activation_id,
                node_id=command.activation_id,
                command_id=command.command_id,
                attempt=command.attempt,
                outcome=ControlOutcome.STALE_INPUT,
                reason=reason,
                graph_revision=command.graph_revision,
            )
        )

    def execute_ready(self, workflow_id: str) -> WorkflowSnapshot:
        """Synchronously dispatch all currently ready commands.

        Real controller integrations may instead consume ``next_invocations``
        and report events asynchronously.  This method is the deterministic
        baseline/replay path and is intentionally bounded to one scheduler
        frontier per call.
        """

        runtime = self._workflow(workflow_id)
        self._drain_manager_wakeups(runtime)
        for scheduled in self.next_invocations(workflow_id):
            self._dispatch(runtime, scheduled)
        return runtime.scheduler.snapshot()

    def run_until_terminal(
        self, workflow_id: str, *, max_frontiers: int = 1000
    ) -> WorkflowSnapshot:
        runtime = self._workflow(workflow_id)
        for _ in range(max_frontiers):
            snapshot = runtime.scheduler.snapshot()
            if snapshot.status not in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}:
                return snapshot
            before = snapshot
            after = self.execute_ready(workflow_id)
            # The synchronous baseline owns the whole event tick: long-lived
            # services must observe action/artifact events before terminality is
            # returned to the task orchestrator.  Explicit async integrations
            # may still call ``pump_service_events`` themselves.
            for _service_round in range(64):
                if not self.pump_service_events(workflow_id):
                    break
            else:
                raise EpisodeRuntimeError("service event cascade exceeded 64 deterministic rounds")
            after = runtime.scheduler.snapshot()
            if after == before:
                raise EpisodeRuntimeError(
                    f"Workflow {workflow_id!r} is running but produced no executable frontier."
                )
        raise EpisodeRuntimeError(
            f"Workflow {workflow_id!r} exceeded max_frontiers={max_frontiers}."
        )

    def accept_node_outcome(self, workflow_id: str, event: NodeOutcomeEvent) -> WorkflowSnapshot:
        """Asynchronous integration point for externally executed activations."""

        return self._workflow(workflow_id).scheduler.on_event(event)

    def pump_service_events(
        self,
        workflow_id: str,
        *,
        max_deliveries: int | None = None,
    ) -> tuple[ServiceDeliveryReport, ...]:
        """Deterministically deliver the durable event tail to live services.

        This method performs no polling or background-thread work.  Each input
        is reserved and fsynced before provider invocation.  A process crash
        leaves the reservation incomplete, so recovery delivers the same event
        with the same ``invocation_id``.  A completed delivery is never invoked
        again, even when its cursor checkpoint was interrupted.
        """

        if max_deliveries is not None and (isinstance(max_deliveries, bool) or max_deliveries < 1):
            raise ValueError("max_deliveries must be a positive integer or None")
        runtime = self._workflow(workflow_id)
        if runtime.closed:
            return ()
        with self._service_pump_lock:
            reports: list[ServiceDeliveryReport] = []
            remaining = max_deliveries
            for activation_id in sorted(runtime.service_subscriptions):
                binding = runtime.service_subscriptions.get(activation_id)
                if binding is None:
                    continue
                try:
                    # Draining exercises the typed multicast queue online.  The
                    # durable history/cursor scan below also covers recovery and
                    # the narrow publish-before-subscribe race.
                    self.event_bus.drain(binding.record.subscriber_id)
                except KeyError as exc:
                    raise EpisodeRuntimeError(
                        "live service is missing its typed EventBus subscription"
                    ) from exc
                history = self.event_bus.history
                cursor = self.service_delivery_ledger.cursor(binding.record.subscriber_id)
                through_offset = cursor
                service_failed = False
                for offset in range(cursor + 1, len(history) + 1):
                    event = history[offset - 1]
                    if event.workflow_id != workflow_id:
                        through_offset = offset
                        continue
                    if event.kind not in binding.record.kinds:
                        through_offset = offset
                        continue
                    if self._is_own_service_bookkeeping(binding, event):
                        through_offset = offset
                        continue
                    if remaining == 0:
                        break
                    report = self._deliver_service_event(
                        runtime,
                        binding,
                        event=event,
                        event_offset=offset,
                    )
                    through_offset = offset
                    if report is not None:
                        reports.append(report)
                        if remaining is not None:
                            remaining -= 1
                        service_failed = report.status == "failed"
                    if service_failed:
                        break
                if through_offset > cursor:
                    self.service_delivery_ledger.advance_cursor(
                        binding.record.subscriber_id, through_offset
                    )
                if remaining == 0:
                    break
            self._drain_manager_wakeups(runtime)
            return tuple(reports)

    def close_workflow(self, workflow_id: str) -> IntentOutcome:
        runtime = self._workflow(workflow_id)
        snapshot = runtime.scheduler.snapshot()
        if snapshot.status in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}:
            raise EpisodeRuntimeError("A running workflow cannot be closed as an IntentOutcome.")
        if not runtime.closed:
            for activation_id in runtime.scheduler.retire_services():
                actor_id = runtime.actor_ids.get(activation_id)
                if actor_id:
                    handle = self.actors.get(actor_id)
                    if handle.state is not ActorState.RETIRED:
                        handle.retire()
                subscription = runtime.service_subscriptions.pop(activation_id, None)
                if subscription is not None:
                    self.event_bus.unsubscribe(subscription.record.subscriber_id)
            self.data_plane.close_workflow(workflow_id)
            self.observations.close_workflow(workflow_id)
            runtime.closed = True
        terminal_events = [
            event
            for event in self.event_bus.history
            if isinstance(event, NodeOutcomeEvent)
            and event.workflow_id == workflow_id
            and event.activation_id == snapshot.terminal_activation
        ]
        evidence_refs = terminal_events[-1].artifact_ids if terminal_events else ()
        return IntentOutcome(
            episode_id=self.episode_id,
            workflow_id=workflow_id,
            intent_id=runtime.intent.intent_id,
            intent_revision=runtime.intent.revision,
            status=self._intent_status(snapshot.status),
            summary=(
                f"Workflow {snapshot.graph_id}@{snapshot.graph_revision} "
                f"finished as {snapshot.status.value}."
            ),
            reason=(snapshot.terminal_outcome.value if snapshot.terminal_outcome else None),
            evidence_refs=evidence_refs,
            final_state_revision=self.state_reducer.state.revision,
            metrics={
                "runtime_events": float(snapshot.event_count),
                "graph_revision": float(snapshot.graph_revision),
            },
        )

    def close_episode(self) -> None:
        if self._closed:
            return
        running = [
            workflow_id
            for workflow_id, runtime in self._workflows.items()
            if runtime.scheduler.status in {WorkflowStatus.CREATED, WorkflowStatus.RUNNING}
        ]
        if running:
            raise EpisodeRuntimeError(
                "Cannot close an episode with running workflows: " + ", ".join(sorted(running))
            )
        for workflow_id, runtime in self._workflows.items():
            if not runtime.closed:
                self.close_workflow(workflow_id)
        self.observations.close()
        self.actors.retire_all()
        self._closed = True

    def _register_service_subscription(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        actor_id: str,
    ) -> None:
        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        if not node.subscriptions:
            return
        subscriber_id = self._stable_service_id(
            "subscriber",
            self.episode_id,
            command.workflow_id,
            command.activation_id,
            command.command_id,
        )
        existing = self.service_delivery_ledger.subscription(subscriber_id)
        start_offset = existing.start_offset if existing is not None else self.event_bus.offset
        record = self.service_delivery_ledger.bind_subscription(
            subscriber_id=subscriber_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            activation_id=command.activation_id,
            command_id=command.command_id,
            graph_revision=command.graph_revision,
            kinds=tuple(node.subscriptions),
            start_offset=start_offset,
        )
        current = runtime.service_subscriptions.get(command.activation_id)
        binding = _ActiveServiceSubscription(
            record=record,
            actor_id=actor_id,
            command=command,
        )
        if current is not None:
            if current != binding:
                raise EpisodeRuntimeError("service activation was rebound to another subscription")
            return
        try:
            self.event_bus.subscribe(subscriber_id, kinds=record.kinds)
        except ValueError as exc:
            raise EpisodeRuntimeError(
                "service subscriber id is already active in this runtime"
            ) from exc
        runtime.service_subscriptions[command.activation_id] = binding

    def _deliver_service_event(
        self,
        runtime: _WorkflowRuntime,
        binding: _ActiveServiceSubscription,
        *,
        event: RuntimeEventBase,
        event_offset: int,
    ) -> ServiceDeliveryReport | None:
        delivery_id = self._stable_service_id(
            "delivery", binding.record.subscriber_id, event.event_id
        )
        invocation_id = self._stable_service_id(
            "event", binding.record.subscriber_id, event.event_id
        )
        reservation = self.service_delivery_ledger.reserve(
            delivery_id=delivery_id,
            subscriber_id=binding.record.subscriber_id,
            event_id=event.event_id,
            event_kind=event.kind,
            event_digest=runtime_event_digest(event),
            event_offset=event_offset,
            invocation_id=invocation_id,
            artifact_attempt=event_offset,
        )
        if self.service_delivery_ledger.completion(delivery_id) is not None:
            return None

        node = self._graph_node(runtime, binding.command.activation_id)
        artifact_ids: tuple[str, ...] = ()
        emitted_event_ids: tuple[str, ...] = ()
        try:
            handle = self.actors.get(binding.actor_id)
            if handle.state is not ActorState.ACTIVE:
                raise EpisodeRuntimeError(f"service actor {binding.actor_id!r} is not active")
            invocation = InvocationSpec(
                invocation_id=reservation.invocation_id,
                idempotency_key=reservation.invocation_id,
                objective=(f"Handle subscribed runtime event {event.kind!r} for {node.runner_ref}"),
                inputs={"event": dump_runtime_event(event)},
                output_contract={port.name: port.schema_id for port in node.outputs},
                requested_capabilities=frozenset(node.required_capabilities),
                requested_effects=self._requested_effects(node.effect_scope),
                budget=self._invocation_budget(node),
                deadline_monotonic_s=self._provider_deadline_monotonic_s(node),
                metadata={
                    "episode_id": self.episode_id,
                    "workflow_id": binding.command.workflow_id,
                    "activation_id": binding.command.activation_id,
                    "service_command_id": binding.command.command_id,
                    "service_attempt": binding.command.attempt,
                    "graph_id": binding.command.graph_id,
                    "graph_revision": binding.command.graph_revision,
                    "graph_digest": runtime.scheduler.graph.digest,
                    "delivery_id": reservation.delivery_id,
                    "event_id": event.event_id,
                    "event_kind": event.kind,
                    "event_digest": reservation.event_digest,
                    "event_offset": event_offset,
                },
            )
            budget_reservation = self._reserve_provider_budget(
                operation_id=f"service-event:{reservation.invocation_id}",
                node=node,
                invocation=invocation,
            )
            provider_entered = False
            raw_result: Any = None
            try:
                provider_entered = True
                raw_result = handle.invoke(invocation)
            finally:
                # Entering AgentHandle.invoke is the provider-attempt boundary;
                # provider failures are conservatively charged their grant.
                self._settle_provider_budget(
                    budget_reservation,
                    provider_entered=provider_entered,
                    invocation=invocation,
                    usage=(
                        raw_result.usage
                        if isinstance(raw_result, ServiceEventResult)
                        else None
                    ),
                )
            self._emit_lifecycle(runtime, binding.actor_id, LifecycleTransition.INVOKED)
            if raw_result in (None, True):
                result = ServiceEventResult()
            elif isinstance(raw_result, ServiceEventResult):
                result = raw_result
            else:
                raise EpisodeRuntimeError(
                    "A service event must return None/True or ServiceEventResult."
                )
            artifact_ids = self._publish_service_outputs(
                runtime,
                binding,
                reservation,
                source_event=event,
                emissions=result.artifacts,
            )
            emitted_event_ids = self._publish_service_runtime_events(
                runtime,
                binding,
                events=result.events,
            )
            result_digest = self._service_result_digest(
                status="succeeded",
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted_event_ids,
                reason="",
            )
            self.service_delivery_ledger.complete(
                delivery_id=delivery_id,
                status="succeeded",
                result_digest=result_digest,
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted_event_ids,
            )
            return ServiceDeliveryReport(
                workflow_id=binding.command.workflow_id,
                activation_id=binding.command.activation_id,
                event_id=event.event_id,
                delivery_id=delivery_id,
                invocation_id=invocation_id,
                status="succeeded",
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted_event_ids,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failure_event_id = self._complete_service_event_failure(
                runtime,
                binding,
                reservation=reservation,
                source_event=event,
                reason=reason,
            )
            emitted = (*emitted_event_ids, failure_event_id)
            result_digest = self._service_result_digest(
                status="failed",
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted,
                reason=reason,
            )
            self.service_delivery_ledger.complete(
                delivery_id=delivery_id,
                status="failed",
                result_digest=result_digest,
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted,
                reason=reason,
            )
            return ServiceDeliveryReport(
                workflow_id=binding.command.workflow_id,
                activation_id=binding.command.activation_id,
                event_id=event.event_id,
                delivery_id=delivery_id,
                invocation_id=invocation_id,
                status="failed",
                artifact_ids=artifact_ids,
                emitted_event_ids=emitted,
                reason=reason,
            )

    def _publish_service_outputs(
        self,
        runtime: _WorkflowRuntime,
        binding: _ActiveServiceSubscription,
        reservation: ServiceDeliveryReservation,
        *,
        source_event: RuntimeEventBase,
        emissions: tuple[ArtifactEmission, ...],
    ) -> tuple[str, ...]:
        node = self._graph_node(runtime, binding.command.activation_id)
        declared = {port.name: port for port in node.outputs}
        emitted: dict[str, ArtifactEmission] = {}
        for emission in emissions:
            if not isinstance(emission, ArtifactEmission):
                raise EpisodeRuntimeError("service artifacts must use ArtifactEmission")
            if emission.port in emitted:
                raise EpisodeRuntimeError("A service event emitted the same output port twice.")
            emitted[emission.port] = emission
        unknown = set(emitted) - set(declared)
        if unknown:
            raise EpisodeRuntimeError(
                f"Service {node.activation_id!r} emitted undeclared ports: "
                f"{', '.join(sorted(unknown))}."
            )
        default_lineage = self._runtime_event_lineage(source_event)
        for port_name, emission in emitted.items():
            expected = declared[port_name]
            if emission.schema_id != expected.schema_id:
                raise EpisodeRuntimeError(
                    f"Output {node.activation_id}.{port_name} expected "
                    f"{expected.schema_id!r}, got {emission.schema_id!r}."
                )
            self.data_plane.validate_payload(emission.schema_id, emission.payload)
            for ref in emission.lineage or default_lineage:
                self.data_plane.resolve(ref)

        artifact_ids: list[str] = []
        for port_name, emission in emitted.items():
            record = self.data_plane.publish_once(
                workflow_id=binding.command.workflow_id,
                activation_id=binding.command.activation_id,
                attempt=reservation.artifact_attempt,
                port=port_name,
                schema=emission.schema_id,
                payload=emission.payload,
                lineage=emission.lineage or default_lineage,
            )
            artifact_ids.append(record.artifact_id)
            published_event_id = self._stable_service_id("artifact", record.artifact_id)
            if not self.event_bus.contains(published_event_id):
                runtime.scheduler.on_event(
                    ArtifactPublished(
                        event_id=published_event_id,
                        episode_id=self.episode_id,
                        workflow_id=binding.command.workflow_id,
                        timestamp=source_event.timestamp,
                        source="episode_data_plane.service_delivery",
                        artifact_id=record.artifact_id,
                        schema_id=record.schema,
                        digest=record.content_digest,
                        producer_activation_id=binding.command.activation_id,
                        port=port_name,
                    )
                )
        return tuple(artifact_ids)

    def _publish_service_runtime_events(
        self,
        runtime: _WorkflowRuntime,
        binding: _ActiveServiceSubscription,
        *,
        events: tuple[RuntimeEventBase, ...],
    ) -> tuple[str, ...]:
        parsed = tuple(parse_runtime_event(event) for event in events)
        allowed_types = (MonitorFinding, StateProposal, PatchRequest)
        for event in parsed:
            if not isinstance(event, allowed_types):
                raise EpisodeRuntimeError(
                    f"Service actors may not emit runtime-owned event kind {event.kind!r}."
                )
            if (
                event.episode_id != self.episode_id
                or event.workflow_id != binding.command.workflow_id
            ):
                raise EpisodeRuntimeError(
                    "service-emitted event belongs to another episode/workflow"
                )
            for value in getattr(event, "evidence_refs", ()):
                self._resolve_service_evidence_ref(value)

        event_ids: list[str] = []
        for event in parsed:
            if isinstance(event, MonitorFinding):
                self._record_monitor_finding(runtime, event)
            elif isinstance(event, PatchRequest):
                self._record_patch_request(runtime, event)
            else:
                runtime.scheduler.on_event(event)
            event_ids.append(event.event_id)
        return tuple(event_ids)

    def _complete_service_event_failure(
        self,
        runtime: _WorkflowRuntime,
        binding: _ActiveServiceSubscription,
        *,
        reservation: ServiceDeliveryReservation,
        source_event: RuntimeEventBase,
        reason: str,
    ) -> str:
        self._emit_lifecycle(
            runtime,
            binding.actor_id,
            LifecycleTransition.FAILED,
            reason=reason,
        )
        failure_event = ServiceOutcome(
            event_id=self._stable_service_id("failure", reservation.delivery_id),
            episode_id=self.episode_id,
            workflow_id=binding.command.workflow_id,
            timestamp=source_event.timestamp,
            source="episode_runtime.service_delivery",
            activation_id=binding.command.activation_id,
            command_id=binding.command.command_id,
            attempt=binding.command.attempt,
            status=ServiceStatus.FAILED,
            reason=reason,
            graph_revision=binding.command.graph_revision,
        )
        runtime.scheduler.on_event(failure_event)
        runtime.service_subscriptions.pop(binding.command.activation_id, None)
        self.event_bus.unsubscribe(binding.record.subscriber_id)
        return failure_event.event_id

    def _runtime_event_lineage(self, event: RuntimeEventBase) -> tuple[ResolvedArtifactRef, ...]:
        values: list[str] = []
        artifact_ids = getattr(event, "artifact_ids", ())
        values.extend(str(value) for value in artifact_ids)
        evidence_refs = getattr(event, "evidence_refs", ())
        values.extend(str(value) for value in evidence_refs)
        for name in ("receipt_ref", "selected_hypothesis_ref"):
            value = getattr(event, name, None)
            if value:
                values.append(str(value))
        if isinstance(event, ArtifactPublished):
            values.append(event.artifact_id)
        resolved: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        records = {record.artifact_id: record for record in self.data_plane.artifacts}
        for value in values:
            ref = records[value].ref if value in records else None
            if ref is None:
                try:
                    decoded = decode_evidence_ref(value)
                    self.data_plane.resolve(decoded)
                    ref = decoded
                except ValueError:
                    continue
            identity = (ref.artifact_id, ref.content_digest)
            if identity not in seen:
                seen.add(identity)
                resolved.append(ref)
        return tuple(resolved)

    def _resolve_service_evidence_ref(self, value: str) -> ResolvedArtifactRef:
        try:
            return self.data_plane.artifact_record(value).ref
        except ValueError:
            try:
                ref = decode_evidence_ref(value)
                self.data_plane.resolve(ref)
                return ref
            except ValueError as exc:
                raise EpisodeRuntimeError(
                    f"service event cites unknown evidence {value!r}"
                ) from exc

    def _service_result_digest(
        self,
        *,
        status: str,
        artifact_ids: tuple[str, ...],
        emitted_event_ids: tuple[str, ...],
        reason: str,
    ) -> str:
        event_index = {event.event_id: event for event in self.event_bus.history}
        payload = {
            "status": status,
            "artifacts": [
                self.data_plane.artifact_record(artifact_id).to_mapping()
                for artifact_id in artifact_ids
            ],
            "events": [dump_runtime_event(event_index[event_id]) for event_id in emitted_event_ids],
            "reason": reason,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _is_own_service_bookkeeping(
        binding: _ActiveServiceSubscription, event: RuntimeEventBase
    ) -> bool:
        if isinstance(event, ServiceOutcome):
            return (
                event.activation_id == binding.command.activation_id
                and event.command_id == binding.command.command_id
            )
        return isinstance(event, LifecycleEvent) and event.actor_id == binding.actor_id

    @staticmethod
    def _stable_service_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
        return f"service-{prefix}-{digest}"

    def _dispatch(self, runtime: _WorkflowRuntime, scheduled: ScheduledInvocation) -> None:
        command = scheduled.command
        graph_node = self._graph_node(runtime, command.activation_id)
        if graph_node.runner_kind == RunnerKind.ACTION_SNAPSHOT:
            self._dispatch_action_snapshot(runtime, scheduled)
            return
        if graph_node.runner_kind == RunnerKind.SYSTEM_ACTION:
            self._dispatch_system_action(runtime, scheduled)
            return
        if graph_node.runner_kind == RunnerKind.ARENA:
            self._dispatch_graph_arena(runtime, scheduled)
            return
        if graph_node.runner_kind == RunnerKind.REDUCER:
            self._dispatch_embodied_state_reducer(runtime, scheduled)
            return
        try:
            self._dispatch_actor(runtime, scheduled)
        except Exception as exc:
            self._complete_actor_failure(runtime, scheduled, exc)

    def _dispatch_action_snapshot(
        self, runtime: _WorkflowRuntime, scheduled: ScheduledInvocation
    ) -> None:
        """Capture one fresh authoritative admission snapshot without an Agent.

        The graph selects only an already registered world/resource binding.
        Publication is immutable per activation attempt.  Recovery therefore
        checks the data plane before touching the backend: a crash after the
        snapshot was published can never silently rebind the plan's physical
        precondition to a later world state.
        """

        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        try:
            self._validate_action_snapshot_node(node)
            existing = tuple(
                record
                for record in self.data_plane.artifacts
                if record.workflow_id == command.workflow_id
                and record.activation_id == command.activation_id
                and record.attempt == command.attempt
                and record.port == "snapshot"
            )
            if len(existing) > 1:
                raise EpisodeRuntimeError(
                    "action_snapshot attempt has multiple durable publications"
                )
            if existing:
                resolved = self.data_plane.resolve(existing[0].ref)
                snapshot = AdmissionSnapshot.model_validate(resolved.payload)
            else:
                key = (node.authority_world_id, node.authoritative_resource)
                backend = self._action_backends[key]
                snapshot = self.action_supervisor.validate_snapshot_freshness(
                    AdmissionSnapshot.model_validate(backend.snapshot(*key))
                )
            if snapshot.world_kind is not WorldKind.AUTHORITATIVE:
                raise EpisodeRuntimeError(
                    "action_snapshot backend returned a non-authoritative world"
                )
            if (
                snapshot.world_id != node.authority_world_id
                or snapshot.resource_id != node.authoritative_resource
            ):
                raise EpisodeRuntimeError(
                    "action_snapshot backend response differs from its typed binding"
                )
            artifact_ids = self._publish_outputs(
                runtime,
                scheduled,
                ActivationExecutionResult(
                    outcome=ControlOutcome.SUCCESS,
                    artifacts=(
                        ArtifactEmission(
                            port="snapshot",
                            schema_id=ACTION_SNAPSHOT_SCHEMA_ID,
                            payload=snapshot.model_dump(mode="json"),
                        ),
                    ),
                ),
            )
        except KeyError:
            self._complete_action_snapshot_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.INFEASIBLE,
                reason=(
                    "no authoritative backend is registered for the typed "
                    "action_snapshot world/resource"
                ),
            )
            return
        except (
            AdmissionRejectedError,
            EpisodeRuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._complete_action_snapshot_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.STALE_INPUT,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return

        event_id = self._node_outcome_event_id(command.command_id)
        existing_event = self._event_by_id(event_id)
        event = NodeOutcomeEvent(
            event_id=event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="action_snapshot_runner",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=ControlOutcome.SUCCESS,
            artifact_ids=artifact_ids,
            graph_revision=command.graph_revision,
            **({"timestamp": existing_event.timestamp} if existing_event is not None else {}),
        )
        runtime.scheduler.on_event(self._require_same_persisted_event(event, existing_event))

    def _complete_action_snapshot_failure(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        outcome: ControlOutcome,
        reason: str,
    ) -> None:
        command = scheduled.command
        event_id = self._node_outcome_event_id(command.command_id)
        existing = self._event_by_id(event_id)
        event = NodeOutcomeEvent(
            event_id=event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="action_snapshot_runner",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=outcome,
            reason=reason,
            graph_revision=command.graph_revision,
            **({"timestamp": existing.timestamp} if existing is not None else {}),
        )
        runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))

    def _dispatch_graph_arena(
        self, runtime: _WorkflowRuntime, scheduled: ScheduledInvocation
    ) -> None:
        """Execute a trusted Arena binding without crossing AgentProvider as a node.

        Candidate workers still use ActorRegistry internally, but the graph
        activation, risk reduction, shadow execution, ranking, promotion and
        publication are all runtime-owned.
        """

        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        try:
            self._validate_arena_node(node)
            binding = self.arena_bindings.resolve(node.runner_ref)
            admitted = scheduled.admission.refs()
            snapshot_ref = admitted["snapshot"]
            risk_ref = admitted["risk"]
            context_refs = {
                name: admitted[name] for name in sorted(binding.context_input_schemas)
            }
            snapshot_artifact = self.data_plane.resolve(snapshot_ref)
            risk_artifact = self.data_plane.resolve(risk_ref)
            if snapshot_artifact.schema != "robomex.admission_snapshot.v1":
                raise EpisodeRuntimeError("Arena snapshot input has the wrong schema")
            if risk_artifact.schema != "robomex.risk_report.v1":
                raise EpisodeRuntimeError("Arena risk input has the wrong schema")
            for name, schema_id in binding.context_input_schemas.items():
                if self.data_plane.resolve(context_refs[name]).schema != schema_id:
                    raise EpisodeRuntimeError(
                        f"Arena context input {name!r} has the wrong schema"
                    )
            snapshot = AdmissionSnapshot.model_validate(snapshot_artifact.payload)
            if snapshot.world_kind is not WorldKind.AUTHORITATIVE:
                raise EpisodeRuntimeError(
                    "Arena promotion input must be an authoritative-world snapshot"
                )
            supplied_risk = RiskReport.model_validate(risk_artifact.payload)
            canonical_risk = RiskReport.assess(supplied_risk.inputs, binding.risk_policy)
            if supplied_risk != canonical_risk:
                raise EpisodeRuntimeError(
                    "Arena risk report differs from the trusted runtime policy reduction"
                )
            candidates = binding.candidates_for(
                snapshot_ref=snapshot_ref,
                risk_ref=risk_ref,
                context_refs=context_refs,
            )
            context = ArenaContext(
                arena_run_id=command.command_id,
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                graph_id=command.graph_id,
                graph_revision=command.graph_revision,
                graph_digest=f"sha256:{runtime.scheduler.graph.digest}",
                command_attempt=command.attempt,
                slot_id=command.activation_id,
                snapshot_ref=snapshot_ref,
                expected_frame=binding.expected_frame,
                world_id=snapshot.world_id,
                resource_id=snapshot.resource_id,
                robot_model_digest=binding.robot_model_digest,
                config_digest=snapshot.config_digest,
                candidate_budget_id=binding.candidate_budget_id,
                candidate_budget_limit=binding.candidate_budget_limit,
                source="graph_arena",
            )
            run_binding_digest = self._graph_arena_binding_digest(
                runtime=runtime,
                command=command,
                binding=binding,
                snapshot_ref=snapshot_ref,
                risk_ref=risk_ref,
                context_refs=context_refs,
                risk_report=canonical_risk,
            )
            execution = self.run_arena(
                command.workflow_id,
                arena=self.create_swarm_arena(
                    gates=binding.gates,
                    adapters=binding.adapters,
                    policy=binding.policy,
                ),
                context=context,
                risk_report=canonical_risk,
                candidates=candidates,
                candidate_budget_remaining=binding.candidate_budget_limit,
                recovery_safe=True,
                run_binding_digest=run_binding_digest,
                publish_runtime_artifacts=False,
            )
            result = execution.result
            artifact_ids = self._publish_graph_arena_outputs(
                runtime=runtime,
                scheduled=scheduled,
                binding=binding,
                snapshot=snapshot,
                snapshot_ref=snapshot_ref,
                risk_ref=risk_ref,
                context_refs=context_refs,
                result=result,
            )
            decision_id = f"arena-decision:{command.command_id}"
            existing_decision = self._event_by_id(decision_id)
            decision = ArenaDecision(
                event_id=decision_id,
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="graph_arena",
                arena_run_id=result.arena_run_id,
                slot_id=command.activation_id,
                graph_revision=command.graph_revision,
                selected_candidate_id=result.selected_candidate_id,
                considered_candidate_ids=tuple(
                    item.candidate_id for item in result.candidate_results
                ),
                rejected_candidate_ids=tuple(
                    item.candidate_id
                    for item in result.candidate_results
                    if item.status is not CandidateStatus.ACCEPTED
                ),
                selection_reason=result.selection_reason,
                selected_hypothesis_ref=artifact_ids[1],
                evidence_refs=artifact_ids,
                **(
                    {"timestamp": existing_decision.timestamp}
                    if existing_decision is not None
                    else {}
                ),
            )
            runtime.scheduler.on_event(
                self._require_same_persisted_event(decision, existing_decision)
            )
            outcome = (
                ControlOutcome.SUCCESS
                if result.selected_candidate_id is not None
                else (
                    ControlOutcome.EXHAUSTED
                    if result.quota_exhausted
                    else ControlOutcome.INFEASIBLE
                )
            )
            node_event_id = self._node_outcome_event_id(command.command_id)
            existing_node = self._event_by_id(node_event_id)
            node_event = NodeOutcomeEvent(
                event_id=node_event_id,
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="graph_arena",
                activation_id=command.activation_id,
                node_id=command.activation_id,
                command_id=command.command_id,
                attempt=command.attempt,
                outcome=outcome,
                reason=(None if outcome is ControlOutcome.SUCCESS else result.selection_reason),
                artifact_ids=artifact_ids,
                graph_revision=command.graph_revision,
                **({"timestamp": existing_node.timestamp} if existing_node is not None else {}),
            )
            runtime.scheduler.on_event(
                self._require_same_persisted_event(node_event, existing_node)
            )
        except RunBudgetExceededError as exc:
            self._complete_graph_arena_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.EXHAUSTED,
                reason=f"run_budget_exhausted:{exc}",
            )
        except (ArenaStaleContextError, ArenaLedgerIntegrityError) as exc:
            self._complete_graph_arena_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.STALE_INPUT,
                reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - runtime fail-closed boundary
            self._complete_graph_arena_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def _publish_graph_arena_outputs(
        self,
        *,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        binding: ArenaBinding,
        snapshot: AdmissionSnapshot,
        snapshot_ref: ResolvedArtifactRef,
        risk_ref: ResolvedArtifactRef,
        context_refs: Mapping[str, ResolvedArtifactRef],
        result: ArenaResult,
    ) -> tuple[str, ...]:
        hypotheses = ArenaHypothesisSet(
            arena_run_id=result.arena_run_id,
            graph_id=result.graph_id,
            graph_revision=result.graph_revision,
            hypotheses=tuple(
                item.hypothesis for item in result.candidate_results if item.hypothesis is not None
            ),
        )
        hypothesis_lineage = tuple(
            dict.fromkeys(
                ref
                for hypothesis in hypotheses.hypotheses
                for ref in (
                    hypothesis.plan_ref,
                    hypothesis.snapshot_ref,
                    *hypothesis.render_refs,
                    *hypothesis.evidence_refs,
                )
            )
        )
        admitted_context_lineage = tuple(
            context_refs[name] for name in sorted(context_refs)
        )
        common_lineage = tuple(
            dict.fromkeys(
                (
                    snapshot_ref,
                    risk_ref,
                    *admitted_context_lineage,
                    *hypothesis_lineage,
                )
            )
        )
        emissions: list[ArtifactEmission] = [
            ArtifactEmission(
                port="result",
                schema_id=result.schema_version,
                payload=result.model_dump(mode="json"),
                lineage=common_lineage,
            ),
            ArtifactEmission(
                port="hypotheses",
                schema_id=hypotheses.schema_version,
                payload=hypotheses.model_dump(mode="json"),
                lineage=common_lineage,
            ),
        ]
        if result.selected_candidate_id is not None:
            promotion = result.promotion_receipt
            selected_ref = result.selected_action_spec_ref
            if promotion is None or selected_ref is None:
                raise EpisodeRuntimeError(
                    "Selected Arena result lacks its runtime promotion binding"
                )
            selected_artifact = self.data_plane.resolve(selected_ref)
            if selected_artifact.schema != "robomex.motion_plan.v2":
                raise EpisodeRuntimeError(
                    "Selected Arena action is not a sealed MotionPlan artifact"
                )
            selected = validate_action_spec(selected_artifact.payload)
            if not isinstance(selected, MotionPlan):
                raise EpisodeRuntimeError("Selected Arena action is not a MotionPlan")
            if (
                selected.expected_snapshot != snapshot
                or selected.world_id != snapshot.world_id
                or selected.resource_id != snapshot.resource_id
                or selected.expected_snapshot.config_digest != snapshot.config_digest
                or selected.robot_model_digest != binding.robot_model_digest
                or promotion.snapshot_ref != snapshot_ref
                or promotion.action_spec_ref != selected_ref
                or promotion.action_spec_digest != selected.content_digest
                or promotion.config_digest != snapshot.config_digest
                or promotion.robot_model_digest != binding.robot_model_digest
            ):
                raise EpisodeRuntimeError(
                    "Selected action snapshot/config/robot binding changed after promotion"
                )
            selected_lineage = tuple(
                dict.fromkeys(
                    (
                        selected_ref,
                        snapshot_ref,
                        risk_ref,
                        *admitted_context_lineage,
                    )
                )
            )
            emissions.extend(
                (
                    ArtifactEmission(
                        port="promotion_receipt",
                        schema_id=promotion.schema_version,
                        payload=promotion.model_dump(mode="json"),
                        lineage=selected_lineage,
                    ),
                    ArtifactEmission(
                        port="selected_action_spec",
                        schema_id=selected.schema_version,
                        payload=selected.model_dump(mode="json"),
                        lineage=selected_lineage,
                    ),
                )
            )
        return self._publish_outputs(
            runtime,
            scheduled,
            ActivationExecutionResult(
                outcome=(
                    ControlOutcome.SUCCESS
                    if result.selected_candidate_id is not None
                    else (
                        ControlOutcome.EXHAUSTED
                        if result.quota_exhausted
                        else ControlOutcome.INFEASIBLE
                    )
                ),
                artifacts=tuple(emissions),
            ),
        )

    @staticmethod
    def _graph_arena_binding_digest(
        *,
        runtime: _WorkflowRuntime,
        command: ActivationCommand,
        binding: ArenaBinding,
        snapshot_ref: ResolvedArtifactRef,
        risk_ref: ResolvedArtifactRef,
        context_refs: Mapping[str, ResolvedArtifactRef],
        risk_report: RiskReport,
    ) -> str:
        payload = {
            "binding_digest": binding.content_digest,
            "graph_digest": runtime.scheduler.graph.digest,
            "graph_id": command.graph_id,
            "graph_revision": command.graph_revision,
            "workflow_id": command.workflow_id,
            "activation_id": command.activation_id,
            "command_id": command.command_id,
            "command_attempt": command.attempt,
            "snapshot_ref": snapshot_ref.to_mapping(),
            "risk_ref": risk_ref.to_mapping(),
            "context_refs": {
                name: context_refs[name].to_mapping() for name in sorted(context_refs)
            },
            "risk_report": risk_report.model_dump(mode="json"),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _complete_graph_arena_failure(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        outcome: ControlOutcome,
        reason: str,
    ) -> None:
        command = scheduled.command
        event_id = self._node_outcome_event_id(command.command_id)
        existing = self._event_by_id(event_id)
        event = NodeOutcomeEvent(
            event_id=event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="graph_arena",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=outcome,
            reason=reason,
            graph_revision=command.graph_revision,
            **({"timestamp": existing.timestamp} if existing is not None else {}),
        )
        runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))

    def _dispatch_embodied_state_reducer(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
    ) -> None:
        """Commit one closed proposal without crossing an AgentProvider boundary."""

        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        # Open/patch validation already established this contract.  Rechecking
        # at dispatch protects recovered scheduler state and future callers
        # that might bypass the normal workflow construction path.
        self._validate_embodied_state_reducer_node(node)
        proposal_ref = scheduled.admission.refs().get("proposal")
        if proposal_ref is None:
            self._complete_reducer_rejection(
                runtime,
                scheduled,
                reason="reducer lacks admitted proposal input",
            )
            return

        try:
            artifact = self.data_plane.resolve(proposal_ref)
            if artifact.schema != _STATE_PROPOSAL_SCHEMA:
                raise EpisodeRuntimeError("reducer proposal input resolved to the wrong schema")
            wire = StateTransitionProposalWire.model_validate(artifact.payload)
            proposal = wire.to_domain()
            if proposal.episode_id != self.episode_id:
                raise StateTransitionRejected("Cross-episode state proposal rejected.")
            commit_event = self._commit_or_reuse_state_proposal(wire, proposal)
        except (StateTransitionRejected, EpisodeRuntimeError, TypeError, ValueError) as exc:
            self._complete_reducer_rejection(
                runtime,
                scheduled,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return

        receipt = StateCommitReceipt(
            episode_id=self.episode_id,
            effect_id=proposal.effect_id,
            proposal_ref=StateArtifactRef.from_domain(proposal_ref),
            proposal_digest=wire.proposal_digest,
            before_revision=proposal.before_revision,
            after_revision=int(commit_event["after_revision"]),
            state_event_id=str(commit_event["event_id"]),
            state_digest=str(commit_event["state_digest"]),
        )
        result = ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="receipt",
                    schema_id=_STATE_RECEIPT_SCHEMA,
                    payload=receipt.model_dump(mode="json"),
                    lineage=(proposal_ref, *proposal.evidence_refs),
                ),
            ),
        )
        # Publication is deliberately after the reducer fsync.  If the process
        # dies anywhere in this block, recovery receives the same command and
        # _commit_or_reuse_state_proposal reconstructs this exact receipt from
        # the already durable state event without advancing state again.
        artifact_ids = self._publish_outputs(runtime, scheduled, result)
        node_event_id = self._node_outcome_event_id(command.command_id)
        existing_node_event = self._event_by_id(node_event_id)
        node_event = NodeOutcomeEvent(
            event_id=node_event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="embodied_state_reducer",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=ControlOutcome.SUCCESS,
            artifact_ids=artifact_ids,
            graph_revision=command.graph_revision,
            **(
                {"timestamp": existing_node_event.timestamp}
                if existing_node_event is not None
                else {}
            ),
        )
        runtime.scheduler.on_event(
            self._require_same_persisted_event(node_event, existing_node_event)
        )

    def _commit_or_reuse_state_proposal(
        self,
        wire: StateTransitionProposalWire,
        proposal: StateTransitionProposal,
    ) -> Mapping[str, Any]:
        with self._state_commit_lock:
            return self._commit_or_reuse_state_proposal_locked(wire, proposal)

    def _commit_or_reuse_state_proposal_locked(
        self,
        wire: StateTransitionProposalWire,
        proposal: StateTransitionProposal,
    ) -> Mapping[str, Any]:
        matching = tuple(
            event
            for event in self.state_reducer.events()
            if isinstance(event.get("proposal"), Mapping)
            and event["proposal"].get("effect_id") == proposal.effect_id
        )
        if len(matching) > 1:
            raise EpisodeRuntimeError(
                f"state effect {proposal.effect_id!r} has multiple durable commits"
            )
        if matching:
            event = matching[0]
            persisted = StateTransitionProposalWire.model_validate(event["proposal"])
            if persisted != wire:
                raise StateTransitionRejected(
                    f"Effect id {proposal.effect_id!r} is already bound to a different proposal."
                )
            return event

        self.state_reducer.commit(proposal)
        committed = tuple(
            event
            for event in self.state_reducer.events()
            if isinstance(event.get("proposal"), Mapping)
            and event["proposal"].get("effect_id") == proposal.effect_id
        )
        if len(committed) != 1:
            raise EpisodeRuntimeError(
                "reducer commit did not produce exactly one durable state event"
            )
        persisted = StateTransitionProposalWire.model_validate(committed[0]["proposal"])
        if persisted != wire:
            raise EpisodeRuntimeError("durable reducer event rebound proposal content")
        return committed[0]

    def _complete_reducer_rejection(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        reason: str,
    ) -> None:
        command = scheduled.command
        node_event_id = self._node_outcome_event_id(command.command_id)
        existing = self._event_by_id(node_event_id)
        event = NodeOutcomeEvent(
            event_id=node_event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="embodied_state_reducer",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=ControlOutcome.STALE_INPUT,
            reason=reason,
            graph_revision=command.graph_revision,
            **({"timestamp": existing.timestamp} if existing is not None else {}),
        )
        runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))

    def _dispatch_actor(self, runtime: _WorkflowRuntime, scheduled: ScheduledInvocation) -> None:
        command = scheduled.command
        graph_node = self._graph_node(runtime, command.activation_id)
        profile = self._profiles[graph_node.runner_ref]
        self._validate_profile(graph_node, profile)
        invocation = InvocationSpec(
            invocation_id=command.command_id,
            idempotency_key=command.command_id,
            objective=str(graph_node.params.get("objective") or graph_node.runner_ref),
            inputs={name: ref.to_mapping() for name, ref in scheduled.admission.refs().items()},
            output_contract={port.name: port.schema_id for port in graph_node.outputs},
            # The graph contract is the only authority request channel.  Keeping
            # this exact (instead of accepting an untyped params override) makes
            # graph review, patch ceilings, and provider admission agree on the
            # same capability set.
            requested_capabilities=frozenset(graph_node.required_capabilities),
            requested_effects=self._requested_effects(graph_node.effect_scope),
            budget=self._invocation_budget(graph_node),
            deadline_monotonic_s=self._provider_deadline_monotonic_s(graph_node),
            metadata={
                "episode_id": self.episode_id,
                "workflow_id": command.workflow_id,
                "activation_id": command.activation_id,
                "attempt": command.attempt,
                "graph_id": command.graph_id,
                "graph_revision": command.graph_revision,
                "graph_digest": runtime.scheduler.graph.digest,
                "effect_scope": command.effect_scope.value,
                "authority_lease_id": command.lease.lease_id if command.lease else None,
                "node_params": dict(graph_node.params),
            },
        )
        try:
            budget_reservation = self._reserve_provider_budget(
                operation_id=f"actor:{command.command_id}",
                node=graph_node,
                invocation=invocation,
            )
        except RunBudgetExceededError as exc:
            self._complete_provider_budget_exhaustion(runtime, scheduled, reason=str(exc))
            return

        provider_entered = False
        raw_result: Any = None
        try:
            try:
                handle = self.actors.get(scheduled.actor_id)
                was_new = False
            except ActorNotFoundError:
                provider_entered = True
                handle = self.actors.spawn(profile, actor_id=scheduled.actor_id)
                was_new = True
            runtime.actor_ids[command.activation_id] = handle.actor_id
            if was_new:
                self._emit_lifecycle(runtime, handle.actor_id, LifecycleTransition.SPAWNED)

            provider_entered = True
            raw_result = handle.invoke(invocation)
        finally:
            self._settle_provider_budget(
                budget_reservation,
                provider_entered=provider_entered,
                invocation=invocation,
                usage=(
                    raw_result.usage
                    if isinstance(raw_result, ActivationExecutionResult)
                    else None
                ),
            )
        self._emit_lifecycle(runtime, handle.actor_id, LifecycleTransition.INVOKED)
        if handle.state is ActorState.RETIRED:
            self._emit_lifecycle(runtime, handle.actor_id, LifecycleTransition.RETIRED)
        if command.lane == ActivationLane.SERVICE:
            if raw_result not in (None, True) and not isinstance(
                raw_result, ActivationExecutionResult
            ):
                raise EpisodeRuntimeError(
                    "A service start must return None/True or ActivationExecutionResult."
                )
            self._register_service_subscription(
                runtime,
                scheduled,
                actor_id=handle.actor_id,
            )
            runtime.scheduler.on_event(
                ServiceOutcome(
                    episode_id=self.episode_id,
                    workflow_id=command.workflow_id,
                    source=f"actor:{handle.actor_id}",
                    activation_id=command.activation_id,
                    command_id=command.command_id,
                    attempt=command.attempt,
                    status=ServiceStatus.STARTED,
                    graph_revision=command.graph_revision,
                )
            )
            return
        if not isinstance(raw_result, ActivationExecutionResult):
            raise EpisodeRuntimeError(
                f"Actor {handle.actor_id!r} returned {type(raw_result).__name__}; "
                "primary activations require ActivationExecutionResult."
            )
        if raw_result.outcome is None:
            raise EpisodeRuntimeError("A primary activation result requires a ControlOutcome.")
        artifact_ids = self._publish_outputs(runtime, scheduled, raw_result)
        event = NodeOutcomeEvent(
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source=f"actor:{handle.actor_id}",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=raw_result.outcome,
            reason=raw_result.reason or None,
            artifact_ids=artifact_ids,
            graph_revision=command.graph_revision,
        )
        runtime.scheduler.on_event(event)

    def _complete_actor_failure(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        exc: Exception,
    ) -> None:
        """Turn provider/contract failures into durable, releasable outcomes."""

        command = scheduled.command
        actor_id = runtime.actor_ids.get(command.activation_id, scheduled.actor_id)
        reason = f"{type(exc).__name__}: {exc}"
        self._emit_lifecycle(
            runtime,
            actor_id,
            LifecycleTransition.FAILED,
            reason=reason,
        )
        try:
            handle = self.actors.get(actor_id)
        except ActorNotFoundError:
            handle = None
        if handle is not None and handle.state is ActorState.RETIRED:
            self._emit_lifecycle(runtime, actor_id, LifecycleTransition.RETIRED)
        if command.lane is ActivationLane.SERVICE:
            runtime.scheduler.on_event(
                ServiceOutcome(
                    episode_id=self.episode_id,
                    workflow_id=command.workflow_id,
                    source="episode_runtime",
                    activation_id=command.activation_id,
                    command_id=command.command_id,
                    attempt=command.attempt,
                    status=ServiceStatus.FAILED,
                    reason=reason,
                    graph_revision=command.graph_revision,
                )
            )
            return
        runtime.scheduler.on_event(
            NodeOutcomeEvent(
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="episode_runtime",
                activation_id=command.activation_id,
                node_id=command.activation_id,
                command_id=command.command_id,
                attempt=command.attempt,
                outcome=ControlOutcome.EXECUTION_FAULT,
                reason=reason,
                graph_revision=command.graph_revision,
            )
        )

    def _dispatch_system_action(
        self, runtime: _WorkflowRuntime, scheduled: ScheduledInvocation
    ) -> None:
        """Execute a sealed artifact through the runtime-owned physical lane."""

        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        continuation_id = (
            f"{self.episode_id}:{command.workflow_id}:{command.graph_id}@"
            f"{command.graph_revision}:{command.activation_id}:a{command.attempt}"
        )
        input_port = str(node.params.get("action_input_port") or "action_spec")
        ref = scheduled.admission.refs().get(input_port)
        if ref is None:
            self._complete_action_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.STALE_INPUT,
                status=ActionStatus.REJECTED,
                reason=f"system action lacks admitted input port {input_port!r}",
            )
            return
        receipt: ExecutionReceipt | None = None
        admitted = None
        backend = None
        evidence_recorder: ActionEvidenceRecorder | None = None
        evidence_lineage: tuple[ResolvedArtifactRef, ...] = ()
        default_evidence = False
        monitor_runtime: MonitorRuntime | None = None
        physical_budget_reservation: RunBudgetReservation | None = None
        # Deterministic event ids require deterministic envelopes across an
        # in-process retry and a post-crash WAL recovery.  Recovery provenance
        # lives in the action WAL; changing ``source`` would rebind the same
        # ActionOutcome id to different bytes and wedge NodeOutcome recovery.
        receipt_source = "sealed_action_runner"
        try:
            resolved = self.data_plane.resolve(ref)
            spec = validate_action_spec(resolved.payload)
            monitor_ref = None
            compiled_monitor = None
            monitor_port = node.params.get("monitor_input_port")
            if monitor_port is not None:
                if not isinstance(monitor_port, str) or not monitor_port.strip():
                    raise EpisodeRuntimeError("monitor_input_port must be a non-empty string")
                monitor_ref = scheduled.admission.refs().get(monitor_port)
                if monitor_ref is None:
                    raise EpisodeRuntimeError(
                        f"system action lacks admitted monitor port {monitor_port!r}"
                    )
                monitor_artifact = self.data_plane.resolve(monitor_ref)
                compiled_monitor = MonitorCompiler().compile(monitor_artifact.payload)
            default_evidence = self._action_evidence_recorder is None
            evidence_recorder = self._action_evidence_recorder or EpisodeActionEvidenceRecorder(
                self.data_plane,
                workflow_id=command.workflow_id,
                publication_attempt=command.attempt,
                base_lineage=tuple(item for item in (ref, monitor_ref) if item is not None),
            )
            self._validate_action_binding(node, command, spec)
            recovered_receipt = self.action_supervisor.receipt_for_continuation(
                continuation_id
            )
            if recovered_receipt is None:
                backend = self._action_backends[(spec.world_id, spec.resource_id)]
                checker = self._feasibility_checkers[(spec.world_id, spec.resource_id)]
                if compiled_monitor is not None:
                    monitor_runtime = MonitorRuntime(
                        episode_id=self.episode_id,
                        workflow_id=command.workflow_id,
                        action_id=command.command_id,
                        plan_digest=spec.content_digest,
                        program=compiled_monitor,
                    )
                    # Compatibility is a read-only admission check.  Run it
                    # before reserving a physical-action unit so an
                    # unsupported/stale telemetry bridge cannot consume the
                    # run's action budget without crossing the WAL or a
                    # physical primitive.  SealedActionRunner repeats the
                    # check immediately before WAL to close the TOCTOU gap.
                    validate_monitor_backend_compatibility(
                        backend=backend,
                        spec=spec,
                        monitor=monitor_runtime,
                    )
            if self.run_budget_authority is not None:
                try:
                    physical_budget_reservation = self.run_budget_authority.reserve(
                        operation_id=f"physical-action:{continuation_id}",
                        requested=RunBudgetVector(physical_actions=1),
                        binding={
                            "kind": "physical_action",
                            "continuation_id": continuation_id,
                            "action_spec_digest": spec.content_digest,
                            "world_id": spec.world_id,
                            "resource_id": spec.resource_id,
                        },
                    )
                except RunBudgetExceededError as exc:
                    self._complete_action_failure(
                        runtime,
                        scheduled,
                        outcome=ControlOutcome.EXHAUSTED,
                        status=ActionStatus.REJECTED,
                        reason=f"run_budget_exhausted:{exc}",
                        plan_digest=spec.content_digest,
                    )
                    return
            if recovered_receipt is not None:
                if (
                    recovered_receipt.spec_digest != spec.content_digest
                    or recovered_receipt.world_id != spec.world_id
                    or recovered_receipt.resource_id != spec.resource_id
                ):
                    raise EpisodeRuntimeError(
                        "persisted continuation receipt does not match the sealed action input"
                    )
                receipt = recovered_receipt
                self._complete_physical_budget(physical_budget_reservation)
            else:
                assert backend is not None
                snapshot = backend.snapshot(spec.world_id, spec.resource_id)
                certificate = checker.certify(spec, snapshot)
                admitted = self.action_supervisor.admit(
                    spec,
                    snapshot,
                    action_id=command.command_id,
                    feasibility_certificate=certificate,
                    monitor_digest=(compiled_monitor.digest if compiled_monitor else None),
                    continuation_id=continuation_id,
                    scheduler_reservation_id=(command.lease.lease_id if command.lease else None),
                    scheduler_reserved_world_id=(command.lease.world_id if command.lease else None),
                    scheduler_reserved_resource_id=(
                        command.lease.resource_id if command.lease else None
                    ),
                )
        except KeyError:
            self._release_physical_budget(physical_budget_reservation)
            self._complete_action_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.INFEASIBLE,
                status=ActionStatus.REJECTED,
                reason=(
                    "no trusted backend or feasibility checker is registered for "
                    "the action resource"
                ),
            )
            return
        except (AdmissionRejectedError, EpisodeRuntimeError, TypeError, ValueError) as exc:
            self._release_physical_budget(physical_budget_reservation)
            self._complete_action_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.STALE_INPUT,
                status=ActionStatus.STALE_PLAN,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return

        if receipt is None:
            assert admitted is not None and backend is not None
            try:
                if isinstance(spec, GripperCommand) and spec.mode is GripperMode.OPEN:
                    self._invalidate_attachment_for_open(ref, admitted.action_id)
                receipt = SealedActionRunner(self.action_supervisor, backend).run(
                    admitted,
                    monitor=monitor_runtime,
                    finding_sink=lambda finding: self._record_monitor_finding(runtime, finding),
                    evidence_recorder=evidence_recorder,
                )
                self._complete_physical_budget(physical_budget_reservation)
            except Exception as exc:
                # The sealed backend boundary was entered.  Failures and
                # partial effects consume the physical-action grant.
                self._complete_physical_budget(physical_budget_reservation)
                self.action_supervisor.release(admitted)
                recovered = self.action_supervisor.receipt_for_continuation(continuation_id)
                if recovered is None:
                    self._complete_action_failure(
                        runtime,
                        scheduled,
                        outcome=ControlOutcome.STALE_INPUT,
                        status=ActionStatus.STALE_PLAN,
                        reason=f"pre_execution_failure:{type(exc).__name__}:{exc}",
                    )
                    return
                receipt = recovered

        assert receipt is not None
        try:
            if default_evidence:
                assert evidence_recorder is not None
                finalized_frames, finalized_video = evidence_recorder.finalize(
                    action_id=receipt.action_id
                )
                if receipt.frame_refs and receipt.frame_refs != finalized_frames:
                    raise EpisodeRuntimeError(
                        "execution receipt frame refs differ from durable evidence"
                    )
                if receipt.video_ref is not None and receipt.video_ref != finalized_video:
                    raise EpisodeRuntimeError(
                        "execution receipt video ref differs from durable evidence"
                    )
                evidence_lineage = self._resolve_action_evidence_refs(
                    finalized_frames, finalized_video
                )
            receipt_lineage: list[ResolvedArtifactRef] = []
            seen_lineage: set[tuple[str, str]] = set()
            for item in (
                *(candidate for candidate in (ref, monitor_ref) if candidate is not None),
                *evidence_lineage,
            ):
                identity = (item.artifact_id, item.content_digest)
                if identity not in seen_lineage:
                    seen_lineage.add(identity)
                    receipt_lineage.append(item)
        except Exception as exc:
            self._complete_action_failure(
                runtime,
                scheduled,
                outcome=ControlOutcome.EXECUTION_FAULT,
                status=ActionStatus.EXECUTION_FAULT,
                reason=f"action_evidence_failure:{type(exc).__name__}:{exc}",
                action_id=receipt.action_id,
                plan_digest=receipt.spec_digest,
            )
            return
        try:
            self._commit_action_receipt(
                runtime,
                scheduled,
                receipt,
                lineage=tuple(receipt_lineage),
                source=receipt_source,
            )
        except Exception:
            if self._command_is_active(runtime, command):
                try:
                    self._commit_action_receipt(
                        runtime,
                        scheduled,
                        receipt,
                        # A retry must preserve the exact artifact binding.  If
                        # the first commit reached ``publish_once`` before a
                        # later scheduler failure, weakening lineage here
                        # would turn a recoverable partial commit into an
                        # immutable artifact-integrity conflict.
                        lineage=tuple(receipt_lineage),
                        source=receipt_source,
                    )
                    return
                except Exception as commit_exc:
                    commit_failure_reason = f"{type(commit_exc).__name__}:{commit_exc}"
                status, _ = self._map_execution_status(receipt.runtime_status)
                self._complete_action_failure(
                    runtime,
                    scheduled,
                    outcome=ControlOutcome.EXECUTION_FAULT,
                    status=status,
                    reason=(f"receipt_commit_failure:{commit_failure_reason}"),
                    action_id=receipt.action_id,
                    plan_digest=receipt.spec_digest,
                )
                return
            raise
        self._drain_manager_wakeups(runtime)

    def _record_monitor_finding(self, runtime: _WorkflowRuntime, finding: MonitorFinding) -> None:
        """Persist a finding immediately and queue only exceptional Manager wakes."""

        already_recorded = self.event_bus.contains(finding.event_id)
        runtime.scheduler.on_event(finding)
        if already_recorded or runtime.manager_session is None:
            return
        signal = self._manager_signal_for_finding(finding)
        if signal is None:
            return
        runtime.pending_manager_signals.append((signal, finding.model_dump(mode="json")))

    def _record_patch_request(self, runtime: _WorkflowRuntime, request: PatchRequest) -> None:
        """Persist a service-authored request and wake only a bounded Manager."""

        already_recorded = self.event_bus.contains(request.event_id)
        runtime.scheduler.on_event(request)
        if already_recorded or runtime.manager_session is None:
            return
        runtime.pending_manager_signals.append(
            (ManagerSignal.RECOVERY, request.model_dump(mode="json"))
        )

    @staticmethod
    def _manager_signal_for_finding(
        finding: MonitorFinding,
    ) -> ManagerSignal | None:
        if finding.finding in {
            MonitorFindingKind.UNOBSERVABLE,
            MonitorFindingKind.TARGET_MOTION,
        }:
            return ManagerSignal.RISK_EXPANSION
        if finding.severity is FindingSeverity.CRITICAL or finding.finding in {
            MonitorFindingKind.ATTACHMENT_ANOMALY,
            MonitorFindingKind.UNSAFE_DEVIATION,
        }:
            return ManagerSignal.RECOVERY
        return None

    def _recover_pending_manager_signals(
        self, runtime: _WorkflowRuntime
    ) -> tuple[tuple[ManagerSignal, Mapping[str, Any]], ...]:
        """Rebuild exceptional wake-ups not covered by a durable call receipt."""

        if runtime.manager_session is None:
            return ()
        processed = {
            receipt.triggering_event_id
            for receipt in runtime.manager_session.receipts
            if receipt.triggering_event_id is not None
        }
        processed.update(
            event.triggering_event_id
            for event in self.event_bus.history
            if isinstance(event, ManagerInvocationOutcome)
            and event.workflow_id == runtime.scheduler.workflow_id
            and event.triggering_event_id is not None
        )
        pending: list[tuple[ManagerSignal, Mapping[str, Any]]] = []
        for event in self.event_bus.history:
            if event.workflow_id != runtime.scheduler.workflow_id or event.event_id in processed:
                continue
            if isinstance(event, MonitorFinding):
                signal = self._manager_signal_for_finding(event)
                if signal is not None:
                    pending.append((signal, event.model_dump(mode="json")))
            elif isinstance(event, PatchRequest):
                pending.append((ManagerSignal.RECOVERY, event.model_dump(mode="json")))
        return tuple(pending)

    @staticmethod
    def _triggering_event_id(
        triggering_event: Mapping[str, Any] | None,
    ) -> str | None:
        if triggering_event is None:
            return None
        value = triggering_event.get("event_id")
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    def _drain_manager_wakeups(self, runtime: _WorkflowRuntime) -> None:
        if runtime.manager_session is None or runtime.manager_invoker is None:
            return
        while runtime.pending_manager_signals:
            signal, triggering_event = runtime.pending_manager_signals.pop(0)
            self.invoke_manager(
                runtime.scheduler.workflow_id,
                signal=signal,
                triggering_event=triggering_event,
            )

    def _resolve_action_evidence_refs(
        self,
        frame_refs: tuple[str, ...],
        video_ref: str | None,
    ) -> tuple[ResolvedArtifactRef, ...]:
        values = (*frame_refs, *((video_ref,) if video_ref is not None else ()))
        resolved: list[ResolvedArtifactRef] = []
        for value in values:
            ref = decode_evidence_ref(value)
            self.data_plane.resolve(ref)
            resolved.append(ref)
        return tuple(resolved)

    def _commit_action_receipt(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        receipt: ExecutionReceipt,
        *,
        lineage: tuple[ResolvedArtifactRef, ...],
        source: str,
    ) -> None:
        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        action_status, control_outcome = self._map_execution_status(receipt.runtime_status)
        output_port = str(node.params.get("receipt_output_port") or "receipt")
        result = ActivationExecutionResult(
            outcome=control_outcome,
            artifacts=(
                ArtifactEmission(
                    port=output_port,
                    schema_id=receipt.schema_version,
                    payload=receipt.model_dump(mode="json"),
                    lineage=lineage,
                ),
            ),
            reason=receipt.abort_reason or "",
        )
        artifact_ids = self._publish_outputs(runtime, scheduled, result)
        receipt_ref = artifact_ids[0] if artifact_ids else None
        if receipt_ref is not None:
            self._invalidate_attachment_after_uncertain_action(
                receipt,
                self.data_plane.artifact_record(receipt_ref).ref,
            )
        action_event_id = self._action_outcome_event_id(command.workflow_id, receipt.action_id)
        existing_action_event = self._event_by_id(action_event_id)
        action_event = ActionOutcome(
            event_id=action_event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source=source,
            action_id=receipt.action_id,
            status=action_status,
            reason=receipt.abort_reason,
            plan_digest=receipt.spec_digest,
            receipt_ref=receipt_ref,
            triggering_finding_id=receipt.triggering_finding_id,
            **(
                {"timestamp": existing_action_event.timestamp}
                if existing_action_event is not None
                else {}
            ),
        )
        runtime.scheduler.on_event(
            self._require_same_persisted_event(action_event, existing_action_event)
        )

        node_event_id = self._node_outcome_event_id(command.command_id)
        existing_node_event = self._event_by_id(node_event_id)
        node_event = NodeOutcomeEvent(
            event_id=node_event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source=source,
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=control_outcome,
            reason=receipt.abort_reason,
            artifact_ids=artifact_ids,
            graph_revision=command.graph_revision,
            **(
                {"timestamp": existing_node_event.timestamp}
                if existing_node_event is not None
                else {}
            ),
        )
        runtime.scheduler.on_event(
            self._require_same_persisted_event(node_event, existing_node_event)
        )

    def _complete_action_failure(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        outcome: ControlOutcome,
        status: ActionStatus,
        reason: str,
        action_id: str | None = None,
        plan_digest: str | None = None,
    ) -> None:
        command = scheduled.command
        resolved_action_id = action_id or command.command_id
        action_event_id = self._action_outcome_event_id(command.workflow_id, resolved_action_id)
        if not self.event_bus.contains(action_event_id):
            runtime.scheduler.on_event(
                ActionOutcome(
                    event_id=action_event_id,
                    episode_id=self.episode_id,
                    workflow_id=command.workflow_id,
                    source="action_supervisor",
                    action_id=resolved_action_id,
                    status=status,
                    reason=reason,
                    plan_digest=plan_digest,
                )
            )
        runtime.scheduler.on_event(
            NodeOutcomeEvent(
                event_id=self._node_outcome_event_id(command.command_id),
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="action_supervisor",
                activation_id=command.activation_id,
                node_id=command.activation_id,
                command_id=command.command_id,
                attempt=command.attempt,
                outcome=outcome,
                reason=reason,
                graph_revision=command.graph_revision,
            )
        )

    def _action_outcome_event_id(self, workflow_id: str, action_id: str) -> str:
        return f"{self.episode_id}:{workflow_id}:action:{action_id}"

    def _event_by_id(self, event_id: str):
        return next(
            (event for event in self.event_bus.history if event.event_id == event_id),
            None,
        )

    @staticmethod
    def _require_same_persisted_event(candidate, existing):
        if existing is None:
            return candidate
        if existing != candidate:
            raise EpisodeRuntimeError(
                f"Persisted event {candidate.event_id!r} conflicts with recovery output."
            )
        return existing

    @staticmethod
    def _node_outcome_event_id(command_id: str) -> str:
        return f"node-outcome:{command_id}"

    @staticmethod
    def _command_is_active(runtime: _WorkflowRuntime, command: ActivationCommand) -> bool:
        return any(
            item.activation_id == command.activation_id
            and item.active_command_id == command.command_id
            for item in runtime.scheduler.snapshot().activations
        )

    @staticmethod
    def _validate_action_binding(node, command: ActivationCommand, spec: ActionSpec) -> None:
        if spec.resource_id != node.authoritative_resource:
            raise EpisodeRuntimeError(
                "sealed action resource differs from the graph authority declaration"
            )
        declared_world = node.authority_world_id
        if spec.world_id != declared_world:
            raise EpisodeRuntimeError(
                "sealed action world differs from the graph authority declaration"
            )
        if command.lease is None:
            raise EpisodeRuntimeError("system action has no scheduler reservation")
        if command.lease.world_id != spec.world_id or command.lease.resource_id != spec.resource_id:
            raise EpisodeRuntimeError("scheduler reservation does not match sealed action")

    def _invalidate_attachment_for_open(
        self, evidence_ref: ResolvedArtifactRef, action_id: str
    ) -> None:
        attachment = self.state_reducer.state.attachment
        if attachment.status is not AttachmentStatus.VERIFIED_HELD or attachment.entity_id is None:
            return
        self.state_reducer.commit(
            StateTransitionProposal.mark_open_admitted(
                episode_id=self.episode_id,
                effect_id=f"open_{action_id}",
                before_revision=self.state_reducer.state.revision,
                source="action_supervisor.open_admission",
                evidence_refs=(evidence_ref,),
                entity_id=attachment.entity_id,
                action_id=action_id,
            )
        )

    def _invalidate_attachment_after_uncertain_action(
        self,
        receipt,
        evidence_ref: ResolvedArtifactRef,
    ) -> None:
        """Invalidate held belief when a primitive may have partially changed the world."""

        if receipt.runtime_status is ExecutionStatus.COMPLETED:
            return
        if not receipt.primitive_receipts and receipt.runtime_status not in {
            ExecutionStatus.INDETERMINATE_AFTER_CRASH,
            ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT,
        }:
            return
        attachment = self.state_reducer.state.attachment
        if attachment.status is not AttachmentStatus.VERIFIED_HELD or attachment.entity_id is None:
            return
        self.state_reducer.commit(
            StateTransitionProposal.mark_interrupted(
                episode_id=self.episode_id,
                effect_id=f"interrupt_{receipt.action_id}",
                before_revision=self.state_reducer.state.revision,
                source="sealed_action_runner.uncertain_terminal",
                evidence_refs=(evidence_ref,),
                entity_id=attachment.entity_id,
                action_id=receipt.action_id,
            )
        )

    @staticmethod
    def _map_execution_status(
        status: ExecutionStatus,
    ) -> tuple[ActionStatus, ControlOutcome]:
        return {
            ExecutionStatus.COMPLETED: (ActionStatus.SUCCEEDED, ControlOutcome.SUCCESS),
            ExecutionStatus.PARTIAL: (
                ActionStatus.PARTIAL,
                ControlOutcome.EXECUTION_FAULT,
            ),
            ExecutionStatus.INTERRUPTED: (
                ActionStatus.INTERRUPTED,
                ControlOutcome.INTERRUPTED,
            ),
            ExecutionStatus.REJECTED: (ActionStatus.REJECTED, ControlOutcome.STALE_INPUT),
            ExecutionStatus.UNKNOWN: (
                ActionStatus.EXECUTION_FAULT,
                ControlOutcome.EXECUTION_FAULT,
            ),
            ExecutionStatus.INDETERMINATE_AFTER_CRASH: (
                ActionStatus.INDETERMINATE_AFTER_CRASH,
                ControlOutcome.INTERRUPTED,
            ),
            ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT: (
                ActionStatus.INDETERMINATE_AFTER_TIMEOUT,
                ControlOutcome.INTERRUPTED,
            ),
        }[status]

    def _publish_outputs(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        result: ActivationExecutionResult,
    ) -> tuple[str, ...]:
        command = scheduled.command
        node = self._graph_node(runtime, command.activation_id)
        declared = {port.name: port for port in node.outputs}
        emitted = {artifact.port: artifact for artifact in result.artifacts}
        if len(emitted) != len(result.artifacts):
            raise EpisodeRuntimeError("An activation emitted the same output port twice.")
        unknown = set(emitted) - set(declared)
        if unknown:
            raise EpisodeRuntimeError(
                f"Activation {node.activation_id!r} emitted undeclared ports: "
                f"{', '.join(sorted(unknown))}."
            )
        if result.outcome == ControlOutcome.SUCCESS:
            missing = [
                port.name for port in node.outputs if port.required and port.name not in emitted
            ]
            if missing:
                raise EpisodeRuntimeError(
                    f"Activation {node.activation_id!r} omitted required outputs: "
                    f"{', '.join(missing)}."
                )
        default_lineage = tuple(scheduled.admission.refs().values())
        for port_name, emission in emitted.items():
            expected = declared[port_name]
            if emission.schema_id != expected.schema_id:
                raise EpisodeRuntimeError(
                    f"Output {node.activation_id}.{port_name} expected {expected.schema_id!r}, "
                    f"got {emission.schema_id!r}."
                )
            self.data_plane.validate_payload(emission.schema_id, emission.payload)
            for ref in emission.lineage or default_lineage:
                self.data_plane.resolve(ref)
        artifact_ids: list[str] = []
        for port_name, emission in emitted.items():
            record = self.data_plane.publish_once(
                workflow_id=command.workflow_id,
                activation_id=command.activation_id,
                attempt=command.attempt,
                port=port_name,
                schema=emission.schema_id,
                payload=emission.payload,
                lineage=emission.lineage or default_lineage,
            )
            artifact_ids.append(record.artifact_id)
            if not any(
                isinstance(event, ArtifactPublished) and event.artifact_id == record.artifact_id
                for event in self.event_bus.history
            ):
                runtime.scheduler.on_event(
                    ArtifactPublished(
                        episode_id=self.episode_id,
                        workflow_id=command.workflow_id,
                        source="episode_data_plane",
                        artifact_id=record.artifact_id,
                        schema_id=record.schema,
                        digest=record.content_digest,
                        producer_activation_id=command.activation_id,
                        port=port_name,
                    )
                )
        return tuple(artifact_ids)

    def _publish_system_artifact(
        self,
        runtime: _WorkflowRuntime,
        *,
        activation_id: str,
        attempt: int,
        port: str,
        schema: str,
        payload: Mapping[str, Any],
        lineage: tuple[ResolvedArtifactRef, ...] = (),
    ):
        record = self.data_plane.publish(
            workflow_id=runtime.scheduler.workflow_id,
            activation_id=activation_id,
            attempt=max(1, attempt),
            port=port,
            schema=schema,
            payload=payload,
            lineage=lineage,
        )
        runtime.scheduler.on_event(
            ArtifactPublished(
                episode_id=self.episode_id,
                workflow_id=runtime.scheduler.workflow_id,
                source="episode_runtime",
                artifact_id=record.artifact_id,
                schema_id=record.schema,
                digest=record.content_digest,
                producer_activation_id=activation_id,
                port=port,
            )
        )
        return record

    def _validate_fragment_profiles(self, proposal: GraphPatchProposal) -> None:
        for node in proposal.fragment.activations:
            if node.runner_kind is RunnerKind.ACTION_SNAPSHOT:
                self._validate_action_snapshot_node(node)
            elif node.runner_kind is RunnerKind.SYSTEM_ACTION:
                self._validate_system_action_node(node)
            elif node.runner_kind is RunnerKind.REDUCER:
                self._validate_embodied_state_reducer_node(node)
            elif node.runner_kind is RunnerKind.ARENA:
                self._validate_arena_node(node)
        missing = {
            node.runner_ref
            for node in proposal.fragment.activations
            if node.runner_kind
            not in {
                RunnerKind.ACTION_SNAPSHOT,
                RunnerKind.SYSTEM_ACTION,
                RunnerKind.REDUCER,
                RunnerKind.ARENA,
            }
            and node.runner_ref not in self._profiles
        }
        if missing:
            raise EpisodeRuntimeError(
                "Graph patch references unregistered actor profiles: " + ", ".join(sorted(missing))
            )
        for node in proposal.fragment.activations:
            if node.runner_kind in {
                RunnerKind.ACTION_SNAPSHOT,
                RunnerKind.SYSTEM_ACTION,
                RunnerKind.REDUCER,
                RunnerKind.ARENA,
            }:
                continue
            self._validate_model_call_declaration(node)
            self._validate_profile(node, self._profiles[node.runner_ref])

    def _validate_graph_profiles(self, graph: CompiledElasticGraph) -> None:
        missing = {
            node.runner_ref
            for node in graph.spec.activations
            if node.runner_kind
            not in {
                RunnerKind.ACTION_SNAPSHOT,
                RunnerKind.SYSTEM_ACTION,
                RunnerKind.REDUCER,
                RunnerKind.ARENA,
            }
            and node.runner_ref not in self._profiles
        }
        if missing:
            raise EpisodeRuntimeError(
                "Compiled graph references unregistered actor profiles: "
                + ", ".join(sorted(missing))
            )
        for node in graph.spec.activations:
            if node.runner_kind not in {
                RunnerKind.ACTION_SNAPSHOT,
                RunnerKind.SYSTEM_ACTION,
                RunnerKind.REDUCER,
                RunnerKind.ARENA,
            }:
                self._validate_model_call_declaration(node)
                self._validate_profile(node, self._profiles[node.runner_ref])

    def _validate_model_call_declaration(self, node) -> None:
        if (
            self.run_budget_authority is not None
            and node.runner_kind is RunnerKind.CODING_WORKER
            and (
                node.estimated_budget.model_calls < 1
                or node.estimated_budget.tokens < 1
                or node.estimated_budget.wall_time_ms < 1
            )
        ):
            raise EpisodeRuntimeError(
                f"Coding worker {node.activation_id!r} must declare positive typed "
                "estimated_budget model_calls, tokens, and wall_time_ms under "
                "manifest budget authority."
            )

    def _validate_arena_contracts(self, graph: CompiledElasticGraph) -> None:
        for node in graph.spec.activations:
            if node.runner_kind is RunnerKind.ARENA:
                self._validate_arena_node(node)

    def _validate_arena_node(self, node) -> None:
        """Close every authority and data port of a graph-native Arena node."""

        binding = self.arena_bindings.resolve(node.runner_ref)
        if node.lane is not ActivationLane.PRIMARY:
            raise EpisodeRuntimeError("Arena nodes must run on the primary lane")
        if node.lifecycle is not LifecycleScope.INVOCATION:
            raise EpisodeRuntimeError("Arena nodes must use invocation lifecycle")
        if node.effect_scope not in {EffectScope.READ_ONLY, EffectScope.SHADOW_WORLD}:
            raise EpisodeRuntimeError("Arena nodes may use only read_only or shadow_world effects")
        if node.authoritative_resource is not None:
            raise EpisodeRuntimeError("Arena nodes can never declare an authoritative resource")
        expected_effect = (
            EffectScope.SHADOW_WORLD
            if any(
                candidate.effect_scope is EffectScope.SHADOW_WORLD
                for candidate in binding.candidates
            )
            else EffectScope.READ_ONLY
        )
        if node.effect_scope is not expected_effect:
            raise EpisodeRuntimeError(
                "Arena graph effect_scope does not match its trusted candidate binding"
            )
        if node.required_capabilities:
            raise EpisodeRuntimeError(
                "Arena authority belongs to its trusted binding, not graph capabilities"
            )
        if node.params:
            raise EpisodeRuntimeError(
                "Arena nodes do not accept params; candidates and budgets are runtime-bound"
            )
        if node.subscriptions:
            raise EpisodeRuntimeError("Arena nodes cannot subscribe as services")
        inputs = {port.name: port for port in node.inputs}
        expected_inputs = {
            "snapshot": "robomex.admission_snapshot.v1",
            "risk": "robomex.risk_report.v1",
            **dict(binding.context_input_schemas),
        }
        if set(inputs) != set(expected_inputs) or any(
            not inputs[name].required or inputs[name].schema_id != schema
            for name, schema in expected_inputs.items()
        ):
            raise EpisodeRuntimeError(
                "Arena requires exact manifest-pinned typed snapshot, risk, and context inputs"
            )
        outputs = {port.name: port for port in node.outputs}
        expected_outputs = {
            "result": ("robomex.arena_result.v1", True),
            "hypotheses": ("robomex.arena_hypotheses.v1", True),
            "promotion_receipt": (
                "robomex.arena_promotion_receipt.v1",
                False,
            ),
            "selected_action_spec": ("robomex.motion_plan.v2", False),
        }
        if set(outputs) != set(expected_outputs) or any(
            outputs[name].schema_id != contract[0] or outputs[name].required is not contract[1]
            for name, contract in expected_outputs.items()
        ):
            raise EpisodeRuntimeError(
                "Arena requires result/hypotheses and optional promotion/action outputs"
            )
        max_candidates = binding.policy.max_candidates
        if node.estimated_budget.actor_spawns < max_candidates:
            raise EpisodeRuntimeError(
                "Arena estimated actor_spawns is below its trusted candidate maximum"
            )
        shadow_bound = sum(
            candidate.effect_scope is EffectScope.SHADOW_WORLD
            for candidate in binding.candidates[:max_candidates]
        )
        if node.estimated_budget.shadow_rollouts < shadow_bound:
            raise EpisodeRuntimeError(
                "Arena estimated shadow_rollouts is below its trusted binding"
            )
        candidate_budget = binding.candidate_provider_budget(max_candidates)
        if self.run_budget_authority is not None:
            undeclared = [
                candidate.candidate_id
                for candidate in binding.candidates[:max_candidates]
                if candidate.profile.runner_kind == RunnerKind.CODING_WORKER.value
                and (
                    candidate.estimated_budget.model_calls < 1
                    or candidate.estimated_budget.tokens < 1
                    or candidate.estimated_budget.wall_time_ms < 1
                )
            ]
            if undeclared:
                raise EpisodeRuntimeError(
                    "Arena coding candidates require trusted positive model_calls, "
                    "tokens, and wall_time_ms: " + ", ".join(undeclared)
                )
        if node.estimated_budget.model_calls < candidate_budget.model_calls:
            raise EpisodeRuntimeError(
                "Arena estimated model_calls is below its trusted candidate aggregate"
            )
        if node.estimated_budget.tokens < candidate_budget.tokens:
            raise EpisodeRuntimeError(
                "Arena estimated tokens is below its trusted candidate aggregate"
            )
        if node.estimated_budget.wall_time_ms < candidate_budget.wall_time_ms:
            raise EpisodeRuntimeError(
                "Arena estimated wall_time_ms is below its trusted candidate aggregate"
            )
        if node.estimated_budget.authoritative_actions:
            raise EpisodeRuntimeError("Arena estimated budget cannot contain authoritative actions")

    def _validate_initial_graph_digest(self, raw_digest: str) -> None:
        """Enforce the manifest allowlist at the runtime boundary.

        The authoring wrapper is defense in depth only: callers may invoke
        ``open_workflow`` or recovery directly, so the Episode must bind every
        durable initial descriptor to a manifest-pinned digest itself.
        """

        if self.allowed_initial_graph_digests is None:
            return
        digest = raw_digest if raw_digest.startswith("sha256:") else f"sha256:{raw_digest}"
        if digest not in self.allowed_initial_graph_digests:
            raise EpisodeRuntimeError(
                f"Initial workflow graph {digest} is not admitted by the run manifest."
            )

    @classmethod
    def _validate_system_action_contracts(cls, graph: CompiledElasticGraph) -> None:
        for node in graph.spec.activations:
            if node.runner_kind is RunnerKind.ACTION_SNAPSHOT:
                cls._validate_action_snapshot_node(node)
            elif node.runner_kind is RunnerKind.SYSTEM_ACTION:
                cls._validate_system_action_node(node)
            elif node.runner_kind is RunnerKind.REDUCER:
                cls._validate_embodied_state_reducer_node(node)

    @staticmethod
    def _validate_system_action_node(node) -> None:
        """Reject malformed physical nodes before a backend can be called."""

        if node.effect_scope is not EffectScope.AUTHORITATIVE_WORLD:
            raise EpisodeRuntimeError(
                "system_action nodes must declare authoritative_world effects"
            )
        if not node.authority_world_id or not node.authoritative_resource:
            raise EpisodeRuntimeError(
                "system_action nodes require typed authority_world_id and authoritative_resource"
            )
        inputs = {port.name: port for port in node.inputs}
        outputs = {port.name: port for port in node.outputs}
        action_port = str(node.params.get("action_input_port") or "action_spec")
        action_contract = inputs.get(action_port)
        allowed_actions = {
            "robomex.motion_plan.v2",
            "robomex.gripper_command.v1",
            "robomex.wait_spec.v1",
        }
        if action_contract is None or action_contract.schema_id not in allowed_actions:
            raise EpisodeRuntimeError("system_action input must be a declared sealed action schema")
        runner_allowlist = {
            "robomex.motion_plan.v2": {
                "runtime.sealed_action",
                "robomex.runtime.execute_motion_plan",
            },
            "robomex.gripper_command.v1": {
                "runtime.sealed_action",
                "robomex.runtime.execute_gripper_command",
            },
            "robomex.wait_spec.v1": {
                "runtime.sealed_action",
                "robomex.runtime.execute_wait",
            },
        }
        if node.runner_ref not in runner_allowlist[action_contract.schema_id]:
            raise EpisodeRuntimeError(
                "system_action runner_ref is not compatible with its sealed schema"
            )
        receipt_port = str(node.params.get("receipt_output_port") or "receipt")
        receipt_contract = outputs.get(receipt_port)
        if (
            receipt_contract is None
            or receipt_contract.schema_id != "robomex.execution_receipt.v2"
            or not receipt_contract.required
        ):
            raise EpisodeRuntimeError(
                "system_action requires a required robomex.execution_receipt.v2 output"
            )
        monitor_port = node.params.get("monitor_input_port")
        if monitor_port is not None:
            if not isinstance(monitor_port, str) or not monitor_port.strip():
                raise EpisodeRuntimeError("monitor_input_port must be a non-empty string")
            monitor_contract = inputs.get(monitor_port)
            if (
                monitor_contract is None
                or monitor_contract.schema_id != "robomex.monitor_program.v1"
            ):
                raise EpisodeRuntimeError(
                    "system_action monitor input must use robomex.monitor_program.v1"
                )

    @staticmethod
    def _validate_action_snapshot_node(node) -> None:
        """Reserve fresh admission capture for one fixed runtime-owned contract."""

        if node.runner_ref != ACTION_SNAPSHOT_RUNNER_REF:
            raise EpisodeRuntimeError(
                "action_snapshot runner_ref must name the runtime-owned capture runner"
            )
        if node.lane is not ActivationLane.PRIMARY:
            raise EpisodeRuntimeError("action_snapshot must run on the primary lane")
        if node.lifecycle is not LifecycleScope.INVOCATION:
            raise EpisodeRuntimeError("action_snapshot must use invocation lifecycle")
        if node.effect_scope is not EffectScope.READ_ONLY:
            raise EpisodeRuntimeError("action_snapshot must be read_only")
        if not node.authority_world_id or not node.authoritative_resource:
            raise EpisodeRuntimeError("action_snapshot requires typed world/resource selection")
        if node.inputs or node.bindings:
            raise EpisodeRuntimeError("action_snapshot cannot admit graph inputs")
        if tuple((port.name, port.schema_id, port.required) for port in node.outputs) != (
            ("snapshot", ACTION_SNAPSHOT_SCHEMA_ID, True),
        ):
            raise EpisodeRuntimeError("action_snapshot requires exactly one typed snapshot output")
        if node.subscriptions or node.required_capabilities or node.verifier_tags or node.params:
            raise EpisodeRuntimeError("action_snapshot has no provider-controlled contract fields")
        if any(node.estimated_budget.model_dump(mode="python").values()):
            raise EpisodeRuntimeError("action_snapshot must have zero execution budget")

    @staticmethod
    def _validate_embodied_state_reducer_node(node) -> None:
        """Reserve the reducer kind for the sole runtime-owned state commit path."""

        if node.runner_ref != _EMBODIED_STATE_REDUCER_RUNNER:
            raise EpisodeRuntimeError(
                "reducer runner_ref must be the runtime-owned embodied-state reducer"
            )
        if node.lane is not ActivationLane.PRIMARY:
            raise EpisodeRuntimeError("embodied-state reducer must run on the primary lane")
        if node.lifecycle is not LifecycleScope.INVOCATION:
            raise EpisodeRuntimeError("embodied-state reducer must use invocation lifecycle")
        if node.effect_scope is not EffectScope.READ_ONLY:
            raise EpisodeRuntimeError(
                "embodied-state reducer cannot request physical/shadow action authority"
            )
        if node.authoritative_resource is not None:
            raise EpisodeRuntimeError(
                "embodied-state reducer cannot name an authoritative resource"
            )
        if node.required_capabilities:
            raise EpisodeRuntimeError("runtime-owned reducer does not accept provider capabilities")
        if node.subscriptions:
            raise EpisodeRuntimeError(
                "embodied-state reducer is an invoked commit, not an event service"
            )
        if node.params:
            raise EpisodeRuntimeError(
                "embodied-state reducer has no configurable provider parameters"
            )
        inputs = tuple((port.name, port.schema_id, port.required) for port in node.inputs)
        outputs = tuple((port.name, port.schema_id, port.required) for port in node.outputs)
        if inputs != (("proposal", _STATE_PROPOSAL_SCHEMA, True),):
            raise EpisodeRuntimeError(
                "embodied-state reducer requires exactly one required proposal input"
            )
        if outputs != (("receipt", _STATE_RECEIPT_SCHEMA, True),):
            raise EpisodeRuntimeError(
                "embodied-state reducer requires exactly one required receipt output"
            )

    def _admit_command_inputs(
        self, runtime: _WorkflowRuntime, command: ActivationCommand
    ) -> InputAdmission:
        node = self._graph_node(runtime, command.activation_id)
        bindings: dict[str, Any] = {}
        for binding in node.bindings:
            if isinstance(binding, ArtifactBinding):
                bindings[binding.input_port] = {
                    "$ref": f"{binding.source_activation}.{binding.source_port}"
                }
            elif isinstance(binding, ExternalBinding):
                ref = runtime.external_refs.get(binding.ref)
                if ref is None:
                    raise EpisodeRuntimeError(
                        f"Workflow lacks external artifact ref {binding.ref!r}."
                    )
                resolved = self.data_plane.resolve(ref)
                if resolved.schema != binding.schema_id:
                    raise EpisodeRuntimeError(
                        f"External ref {binding.ref!r} expected {binding.schema_id!r}, "
                        f"got {resolved.schema!r}."
                    )
                bindings[binding.input_port] = ref
        purpose = (
            AdmissionPurpose.PHYSICAL_ACTION
            if node.runner_kind is RunnerKind.SYSTEM_ACTION
            else AdmissionPurpose.VERIFICATION
        )
        freshness_context = None
        if self._freshness_context_provider is not None:
            freshness_context = self._freshness_context_provider(purpose)
            if not isinstance(freshness_context, AdmissionFreshnessContext):
                raise EpisodeRuntimeError(
                    "freshness_context_provider must return AdmissionFreshnessContext."
                )
            if freshness_context.purpose is not purpose:
                raise EpisodeRuntimeError(
                    "freshness context purpose does not match activation admission."
                )
            if freshness_context.current_state_revision != self.state_reducer.state.revision:
                raise EpisodeRuntimeError(
                    "freshness context carries a stale embodied-state revision."
                )
        return self.data_plane.admit_inputs(
            admission_id=f"adm_{command.command_id}",
            workflow_id=command.workflow_id,
            activation_id=command.activation_id,
            bindings=bindings,
            freshness_context=freshness_context,
        )

    @staticmethod
    def _install_runtime_schemas(registry: SchemaRegistry) -> None:
        """Compatibility wrapper; new code should call install_runtime_schemas."""

        install_runtime_schemas(registry)

    def _actor_id(self, runtime: _WorkflowRuntime, command: ActivationCommand) -> str:
        node = self._graph_node(runtime, command.activation_id)
        if node.lifecycle == LifecycleScope.INVOCATION:
            return f"{command.workflow_id}_{command.activation_id}_a{command.attempt}"
        if node.lifecycle == LifecycleScope.WORKFLOW:
            return f"{command.workflow_id}_{command.activation_id}"
        return f"{self.episode_id}_{command.activation_id}"

    def _emit_lifecycle(
        self,
        runtime: _WorkflowRuntime,
        actor_id: str,
        transition: LifecycleTransition,
        *,
        reason: str | None = None,
    ) -> None:
        runtime.scheduler.on_event(
            LifecycleEvent(
                episode_id=self.episode_id,
                workflow_id=runtime.scheduler.workflow_id,
                source="episode_runtime",
                actor_id=actor_id,
                transition=transition,
                reason=reason,
            )
        )

    @staticmethod
    def _requested_effects(effect_scope: EffectScope) -> frozenset[str]:
        if effect_scope == EffectScope.READ_ONLY:
            return frozenset()
        return frozenset({effect_scope.value})

    @staticmethod
    def _invocation_budget(node) -> Mapping[str, float]:
        """Project the graph's typed estimate into the provider grant."""

        return {
            name: float(value)
            for name, value in node.estimated_budget.model_dump(mode="python").items()
        }

    def _provider_deadline_monotonic_s(self, node) -> float | None:
        """Derive a process-local deadline from typed and run-wide limits.

        The dynamic monotonic value is intentionally not durable identity.  A
        retry binds the same ``wall_time_ms`` grant and run/graph metadata, then
        derives an equal or tighter deadline in its current process.
        """

        local_remaining_s = node.estimated_budget.wall_time_ms / 1000.0
        authority = self.run_budget_authority
        if authority is None:
            return time.monotonic() + local_remaining_s if local_remaining_s > 0 else None
        snapshot = authority.snapshot()
        run_remaining_s = max(snapshot.deadline_s - snapshot.observed_at_s, 0.0)
        return time.monotonic() + min(local_remaining_s, run_remaining_s)

    def _run_deadline_monotonic_s(self) -> float | None:
        """Translate the durable run wall deadline for nested provider calls."""

        authority = self.run_budget_authority
        if authority is None:
            return None
        snapshot = authority.snapshot()
        run_remaining_s = max(snapshot.deadline_s - snapshot.observed_at_s, 0.0)
        return time.monotonic() + run_remaining_s

    def _reserve_provider_budget(
        self,
        *,
        operation_id: str,
        node,
        invocation: InvocationSpec,
    ) -> RunBudgetReservation | None:
        """Reserve the authoritative grant before any provider boundary."""

        if self.run_budget_authority is None:
            return None
        return self.run_budget_authority.reserve(
            operation_id=operation_id,
            requested=RunBudgetVector(
                model_calls=node.estimated_budget.model_calls,
                tokens=node.estimated_budget.tokens,
                wall_time_s=node.estimated_budget.wall_time_ms / 1000.0,
            ),
            binding={
                "kind": "provider_invocation",
                "invocation_fingerprint": invocation.fingerprint(),
            },
        )

    def _settle_provider_budget(
        self,
        reservation: RunBudgetReservation | None,
        *,
        provider_entered: bool,
        invocation: InvocationSpec,
        usage: InvocationUsage | None,
    ) -> None:
        if reservation is None or self.run_budget_authority is None:
            return
        if reservation.status is RunBudgetOperationStatus.COMPLETED:
            return
        if provider_entered:
            if usage is None:
                # Unknown/crashed/opaque providers are conservatively charged
                # the complete reservation.
                self.run_budget_authority.complete(reservation)
                return
            requested = reservation.requested
            measured = RunBudgetVector(
                model_calls=usage.model_calls,
                tokens=usage.tokens,
                wall_time_s=usage.wall_time_s,
            )
            if usage.invocation_fingerprint != f"sha256:{invocation.fingerprint()}":
                self.run_budget_authority.complete(reservation)
                raise EpisodeRuntimeError(
                    "Provider usage is bound to another invocation fingerprint"
                )
            exceeded = [
                name
                for name in ("model_calls", "tokens", "wall_time_s")
                if getattr(measured, name) > getattr(requested, name)
            ]
            if exceeded:
                self.run_budget_authority.complete(reservation)
                raise EpisodeRuntimeError(
                    "Provider reported usage above its reserved grant: "
                    + ", ".join(exceeded)
                )
            self.run_budget_authority.complete(reservation, settlement=measured)
        else:
            self.run_budget_authority.release(reservation)

    def _complete_physical_budget(self, reservation: RunBudgetReservation | None) -> None:
        if reservation is None or self.run_budget_authority is None:
            return
        if reservation.status is not RunBudgetOperationStatus.COMPLETED:
            self.run_budget_authority.complete(reservation)

    def _release_physical_budget(self, reservation: RunBudgetReservation | None) -> None:
        if reservation is None or self.run_budget_authority is None:
            return
        if reservation.status is RunBudgetOperationStatus.RESERVED:
            self.run_budget_authority.release(reservation)

    def _complete_provider_budget_exhaustion(
        self,
        runtime: _WorkflowRuntime,
        scheduled: ScheduledInvocation,
        *,
        reason: str,
    ) -> None:
        """Close a denied provider activation without crossing the provider."""

        command = scheduled.command
        closed_reason = f"run_budget_exhausted:{reason}"
        if command.lane is ActivationLane.SERVICE:
            event_id = f"budget-service:{command.command_id}"
            existing = self._event_by_id(event_id)
            event = ServiceOutcome(
                event_id=event_id,
                episode_id=self.episode_id,
                workflow_id=command.workflow_id,
                source="run_budget_authority",
                activation_id=command.activation_id,
                command_id=command.command_id,
                attempt=command.attempt,
                status=ServiceStatus.FAILED,
                reason=closed_reason,
                graph_revision=command.graph_revision,
                **({"timestamp": existing.timestamp} if existing is not None else {}),
            )
            runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))
            return
        event_id = self._node_outcome_event_id(command.command_id)
        existing = self._event_by_id(event_id)
        event = NodeOutcomeEvent(
            event_id=event_id,
            episode_id=self.episode_id,
            workflow_id=command.workflow_id,
            source="run_budget_authority",
            activation_id=command.activation_id,
            node_id=command.activation_id,
            command_id=command.command_id,
            attempt=command.attempt,
            outcome=ControlOutcome.EXHAUSTED,
            reason=closed_reason,
            graph_revision=command.graph_revision,
            **({"timestamp": existing.timestamp} if existing is not None else {}),
        )
        runtime.scheduler.on_event(self._require_same_persisted_event(event, existing))

    @staticmethod
    def _validate_profile(node, profile: ActorProfile) -> None:
        if profile.runner_kind != node.runner_kind.value:
            raise EpisodeRuntimeError(
                f"ActorProfile {profile.profile_id!r} runner_kind {profile.runner_kind!r} "
                f"does not match graph runner {node.runner_kind.value!r}."
            )
        expected_lifecycle = (
            ActorLifecycle.EPHEMERAL
            if node.lifecycle == LifecycleScope.INVOCATION
            else ActorLifecycle.SERVICE
        )
        if profile.lifecycle != expected_lifecycle:
            raise EpisodeRuntimeError(
                f"ActorProfile {profile.profile_id!r} lifecycle does not match graph node."
            )
        capability_excess = frozenset(node.required_capabilities) - profile.capability_ceiling
        effect_excess = (
            EpisodeRuntime._requested_effects(node.effect_scope) - profile.effect_ceiling
        )
        if capability_excess or effect_excess:
            details: list[str] = []
            if capability_excess:
                details.append(f"capabilities={sorted(capability_excess)!r}")
            if effect_excess:
                details.append(f"effects={sorted(effect_excess)!r}")
            raise EpisodeRuntimeError(
                f"Graph node {node.activation_id!r} exceeds ActorProfile "
                f"{profile.profile_id!r}: {', '.join(details)}."
            )

    @staticmethod
    def _intent_status(status: WorkflowStatus) -> IntentStatus:
        return {
            WorkflowStatus.SUCCEEDED: IntentStatus.SUCCEEDED,
            WorkflowStatus.FAILED: IntentStatus.FAILED,
            WorkflowStatus.UNCERTAIN: IntentStatus.UNCERTAIN,
            WorkflowStatus.EXHAUSTED: IntentStatus.EXHAUSTED,
            WorkflowStatus.CANCELLED: IntentStatus.CANCELLED,
        }[status]

    @staticmethod
    def _graph_node(runtime: _WorkflowRuntime, activation_id: str):
        return next(
            node
            for node in runtime.scheduler.graph.spec.activations
            if node.activation_id == activation_id
        )

    def _arena_graph_revision(self, context: ArenaContext) -> tuple[str, int]:
        runtime = self._workflow(context.workflow_id)
        spec = runtime.scheduler.graph.spec
        return spec.graph_id, spec.revision

    def _workflow(self, workflow_id: str) -> _WorkflowRuntime:
        try:
            return self._workflows[workflow_id]
        except KeyError as exc:
            raise EpisodeRuntimeError(f"Unknown workflow {workflow_id!r}.") from exc


__all__ = [
    "ActivationExecutionResult",
    "ArtifactEmission",
    "EpisodeRuntime",
    "EpisodeRuntimeError",
    "InvocationUsage",
    "ScheduledInvocation",
    "ServiceDeliveryReport",
    "ServiceEventResult",
    "install_runtime_schemas",
]
