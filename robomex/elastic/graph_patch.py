"""Transactional elastic-frontier patching for RoboMEx v2.

The graph is the control/data contract of an Agent Swarm, not a mutable bag of
nodes.  A manager may only compose at a pre-authorized :class:`ClosedSlot`.
Each slot freezes the typed cut and the maximum effects, capabilities, budget,
and verifier duties of the replacement.  The coordinator materializes a full
successor graph, compiles it, and only then atomically hands it to the running
activation scheduler.

Actor roster changes are intentionally outside this protocol.  Spawning or
retiring an implementation actor does not alter graph topology or revision.
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.elastic.compiler import CompiledElasticGraph, ElasticGraphCompiler
from robomex.elastic.graph_spec import (
    ActivationLane,
    ActivationSpec,
    ArtifactBinding,
    BoundedLoopSpec,
    EffectScope,
    ElasticGraphSpec,
    ExecutionBudget,
    ExternalBinding,
    TransitionSpec,
)
from robomex.runtime.activation import (
    ActivationScheduler,
    ActivationStatus,
    SchedulerError,
)
from robomex.runtime.events import ControlOutcome, RosterUpdate


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PatchOperation(str, Enum):  # noqa: UP042 - Python 3.10 support
    FILL_SLOT = "fill_slot"
    REPLACE_UNEXECUTED_FRAGMENT = "replace_unexecuted_fragment"


class BarrierProfile(str, Enum):  # noqa: UP042 - Python 3.10 support
    GLOBAL_QUIESCENCE = "global_quiescence"
    AFFECTED_SCOPE = "affected_scope"


class PatchRejectCode(str, Enum):  # noqa: UP042 - Python 3.10 support
    STALE_BASE_REVISION = "stale_base_revision"
    UNKNOWN_SLOT = "unknown_slot"
    OPERATION_NOT_ALLOWED = "operation_not_allowed"
    INVALID_FRONTIER = "invalid_frontier"
    BARRIER_NOT_SATISFIED = "barrier_not_satisfied"
    TARGET_IMMUTABLE = "target_immutable"
    EFFECT_CEILING_EXCEEDED = "effect_ceiling_exceeded"
    CAPABILITY_CEILING_EXCEEDED = "capability_ceiling_exceeded"
    BUDGET_CEILING_EXCEEDED = "budget_ceiling_exceeded"
    MISSING_VERIFIER = "missing_verifier"
    VERIFIER_ORDER_INVALID = "verifier_order_invalid"
    INVALID_FRAGMENT = "invalid_fragment"
    COMPILE_FAILED = "compile_failed"
    COMMIT_CONFLICT = "commit_conflict"


class TypedInputCut(_StrictModel):
    """One typed data dependency entering the closed fragment."""

    cut_id: str = Field(min_length=1, max_length=256)
    schema_id: str = Field(min_length=1, max_length=192)
    target_activation: str = Field(min_length=1, max_length=128)
    target_port: str = Field(min_length=1, max_length=128)


class TypedOutputCut(_StrictModel):
    """One typed artifact source exported by the closed fragment."""

    cut_id: str = Field(min_length=1, max_length=256)
    schema_id: str = Field(min_length=1, max_length=192)
    source_activation: str = Field(min_length=1, max_length=128)
    source_port: str = Field(min_length=1, max_length=128)


class ControlExitCut(_StrictModel):
    """One primary-control edge leaving the closed fragment."""

    cut_id: str = Field(min_length=1, max_length=256)
    source_activation: str = Field(min_length=1, max_length=128)
    outcome: ControlOutcome
    target_activation: str = Field(min_length=1, max_length=128)


class VerifierObligation(_StrictModel):
    """Evidence gate that a replacement must carry before risky effects."""

    obligation_id: str = Field(min_length=1, max_length=128)
    verifier_tag: str = Field(min_length=1, max_length=128)
    required_output_schema_id: str | None = Field(default=None, max_length=192)
    before_effect_scope: EffectScope = EffectScope.AUTHORITATIVE_WORLD
    require_control_dominance: bool = True


class ClosedSlot(_StrictModel):
    """A typed, bounded and currently closed composition point."""

    slot_id: str = Field(min_length=1, max_length=128)
    target_activation_ids: tuple[str, ...] = Field(min_length=1)
    control_entry_activation: str = Field(min_length=1, max_length=128)
    input_cut: tuple[TypedInputCut, ...] = ()
    output_cut: tuple[TypedOutputCut, ...] = ()
    control_exit_cut: tuple[ControlExitCut, ...] = ()
    effect_ceiling: EffectScope = EffectScope.READ_ONLY
    capability_ceiling: tuple[str, ...] = ()
    budget_ceiling: ExecutionBudget = Field(default_factory=ExecutionBudget)
    verifier_obligations: tuple[VerifierObligation, ...] = ()
    allowed_operations: tuple[PatchOperation, ...] = (
        PatchOperation.FILL_SLOT,
        PatchOperation.REPLACE_UNEXECUTED_FRAGMENT,
    )
    barrier_profile: BarrierProfile = BarrierProfile.AFFECTED_SCOPE

    @model_validator(mode="after")
    def _validate_slot(self) -> ClosedSlot:
        if any(not value.strip() for value in self.target_activation_ids):
            raise ValueError("ClosedSlot target ids must be non-empty strings.")
        if len(self.target_activation_ids) != len(set(self.target_activation_ids)):
            raise ValueError("A ClosedSlot cannot repeat target activations.")
        if self.control_entry_activation not in self.target_activation_ids:
            raise ValueError("ClosedSlot control entry must be one of its targets.")
        for label, values in (
            ("input cut", tuple(item.cut_id for item in self.input_cut)),
            ("output cut", tuple(item.cut_id for item in self.output_cut)),
            ("control exit cut", tuple(item.cut_id for item in self.control_exit_cut)),
            (
                "verifier obligation",
                tuple(item.obligation_id for item in self.verifier_obligations),
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"A ClosedSlot cannot repeat a {label} id.")
        if len(self.capability_ceiling) != len(set(self.capability_ceiling)):
            raise ValueError("A ClosedSlot cannot repeat a capability ceiling.")
        if any(not value.strip() for value in self.capability_ceiling):
            raise ValueError("Capability ceilings must be non-empty strings.")
        if len(self.allowed_operations) != len(set(self.allowed_operations)):
            raise ValueError("A ClosedSlot cannot repeat an allowed operation.")
        return self


class ComposableFrontier(_StrictModel):
    """Revision-bound set of the only legal graph composition points."""

    graph_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    graph_digest: str = Field(min_length=64, max_length=64)
    slots: tuple[ClosedSlot, ...] = ()

    @model_validator(mode="after")
    def _unique_slots(self) -> ComposableFrontier:
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("ComposableFrontier contains duplicate slot ids.")
        return self


class FragmentInputRoute(_StrictModel):
    cut_id: str = Field(min_length=1, max_length=256)
    target_activation: str = Field(min_length=1, max_length=128)
    target_port: str = Field(min_length=1, max_length=128)


class FragmentOutputRoute(_StrictModel):
    cut_id: str = Field(min_length=1, max_length=256)
    source_activation: str = Field(min_length=1, max_length=128)
    source_port: str = Field(min_length=1, max_length=128)


class FragmentExitRoute(_StrictModel):
    cut_id: str = Field(min_length=1, max_length=256)
    source_activation: str = Field(min_length=1, max_length=128)
    outcome: ControlOutcome


class OuterLoopReplacement(_StrictModel):
    """Membership substitution when a slot cuts only part of an old loop."""

    loop_id: str = Field(min_length=1, max_length=128)
    activation_ids: tuple[str, ...] = Field(min_length=1)
    entry_activation: str | None = Field(default=None, max_length=128)


class GraphFragment(_StrictModel):
    """A fully declared replacement, with explicit routes across every cut."""

    entry_activation: str = Field(min_length=1, max_length=128)
    activations: tuple[ActivationSpec, ...] = Field(min_length=1)
    transitions: tuple[TransitionSpec, ...] = ()
    bounded_loops: tuple[BoundedLoopSpec, ...] = ()
    input_routes: tuple[FragmentInputRoute, ...] = ()
    output_routes: tuple[FragmentOutputRoute, ...] = ()
    exit_routes: tuple[FragmentExitRoute, ...] = ()
    terminal_activations: tuple[str, ...] = ()
    outer_loop_replacements: tuple[OuterLoopReplacement, ...] = ()

    @model_validator(mode="after")
    def _validate_fragment_local_ids(self) -> GraphFragment:
        activation_ids = [item.activation_id for item in self.activations]
        activation_set = set(activation_ids)
        if len(activation_ids) != len(activation_set):
            raise ValueError("GraphFragment contains duplicate activation ids.")
        if self.entry_activation not in activation_set:
            raise ValueError("GraphFragment entry is absent from its activations.")
        if not set(self.terminal_activations).issubset(activation_set):
            raise ValueError("GraphFragment terminal is absent from its activations.")
        if len(self.terminal_activations) != len(set(self.terminal_activations)):
            raise ValueError("GraphFragment contains duplicate terminals.")
        for transition in self.transitions:
            if transition.source not in activation_set or transition.target not in activation_set:
                raise ValueError("Fragment transitions must be internal; use explicit exit routes.")
        for route in self.input_routes:
            if route.target_activation not in activation_set:
                raise ValueError("Fragment input route targets an unknown activation.")
        for route in self.output_routes:
            if route.source_activation not in activation_set:
                raise ValueError("Fragment output route names an unknown activation.")
        for route in self.exit_routes:
            if route.source_activation not in activation_set:
                raise ValueError("Fragment exit route names an unknown activation.")
        loop_replacements = [item.loop_id for item in self.outer_loop_replacements]
        if len(loop_replacements) != len(set(loop_replacements)):
            raise ValueError("GraphFragment repeats an outer-loop replacement.")
        return self


class GraphPatchProposal(_StrictModel):
    patch_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    operation: PatchOperation
    slot_id: str = Field(min_length=1, max_length=128)
    base_revision: int = Field(ge=1)
    fragment: GraphFragment
    opened_slots: tuple[ClosedSlot, ...] = ()
    requested_by: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _unique_opened_slots(self) -> GraphPatchProposal:
        ids = [slot.slot_id for slot in self.opened_slots]
        if len(ids) != len(set(ids)):
            raise ValueError("A patch cannot open duplicate slot ids.")
        return self


class PatchReceipt(_StrictModel):
    receipt_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    patch_id: str
    operation: PatchOperation
    slot_id: str
    accepted: bool
    reason_codes: tuple[PatchRejectCode, ...] = ()
    reasons: tuple[str, ...] = ()
    before_revision: int = Field(ge=1)
    after_revision: int = Field(ge=1)
    before_digest: str = Field(min_length=64, max_length=64)
    after_digest: str = Field(min_length=64, max_length=64)
    proposed_digest: str | None = Field(default=None, min_length=64, max_length=64)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # noqa: UP017

    @model_validator(mode="after")
    def _decision_shape(self) -> PatchReceipt:
        if self.accepted and (self.reason_codes or self.reasons):
            raise ValueError("An accepted PatchReceipt cannot contain rejection reasons.")
        if not self.accepted and (not self.reason_codes or not self.reasons):
            raise ValueError("A rejected PatchReceipt requires a code and a reason.")
        return self


class GraphPatchCommit(_StrictModel):
    """Recovery-complete record for one accepted dynamic-graph mutation."""

    schema_version: Literal["robomex.graph_patch_commit.v1"] = (
        "robomex.graph_patch_commit.v1"
    )
    episode_id: str = Field(min_length=1, max_length=256)
    workflow_id: str = Field(min_length=1, max_length=256)
    proposal: GraphPatchProposal
    receipt: PatchReceipt
    predecessor_graph: ElasticGraphSpec
    predecessor_digest: str = Field(min_length=64, max_length=64)
    successor_graph: ElasticGraphSpec
    successor_digest: str = Field(min_length=64, max_length=64)
    next_frontier: ComposableFrontier

    @model_validator(mode="after")
    def _causal_binding(self) -> GraphPatchCommit:
        if not self.receipt.accepted:
            raise ValueError("GraphPatchCommit requires an accepted receipt.")
        if self.proposal.patch_id != self.receipt.patch_id:
            raise ValueError("Graph patch proposal and receipt ids differ.")
        if self.proposal.base_revision != self.receipt.before_revision:
            raise ValueError("Graph patch proposal is not bound to the receipt base.")
        if (
            self.predecessor_graph.revision != self.receipt.before_revision
            or self.predecessor_digest != self.receipt.before_digest
        ):
            raise ValueError("Predecessor graph is not bound to the accepted receipt.")
        if (
            self.successor_graph.revision != self.receipt.after_revision
            or self.successor_digest != self.receipt.after_digest
        ):
            raise ValueError("Successor graph is not bound to the accepted receipt.")
        if self.predecessor_graph.graph_id != self.successor_graph.graph_id:
            raise ValueError("A graph patch commit cannot change graph identity.")
        if (
            self.next_frontier.graph_id != self.successor_graph.graph_id
            or self.next_frontier.revision != self.successor_graph.revision
            or self.next_frontier.graph_digest != self.successor_digest
        ):
            raise ValueError("Next frontier is not bound to the successor graph.")
        return self


@dataclass(frozen=True)
class GraphVersionRecord:
    revision: int
    digest: str
    spec: ElasticGraphSpec
    accepted_receipt: PatchReceipt | None


class _PatchValidationError(ValueError):
    def __init__(self, code: PatchRejectCode, message: str) -> None:
        super().__init__(message)
        self.code = code


_EFFECT_RANK = {
    EffectScope.READ_ONLY: 0,
    EffectScope.SHADOW_WORLD: 1,
    EffectScope.AUTHORITATIVE_WORLD: 2,
}


def _input_cut_id(activation_id: str, port: str) -> str:
    return f"input:{activation_id}:{port}"


def _output_cut_id(activation_id: str, port: str) -> str:
    return f"output:{activation_id}:{port}"


def _exit_cut_id(source: str, outcome: ControlOutcome, target: str) -> str:
    return f"exit:{source}:{outcome.value}:{target}"


def derive_closed_slot(
    graph: CompiledElasticGraph,
    *,
    slot_id: str,
    target_activation_ids: tuple[str, ...],
    control_entry_activation: str | None = None,
    effect_ceiling: EffectScope = EffectScope.READ_ONLY,
    capability_ceiling: tuple[str, ...] = (),
    budget_ceiling: ExecutionBudget | None = None,
    verifier_obligations: tuple[VerifierObligation, ...] = (),
    allowed_operations: tuple[PatchOperation, ...] = (
        PatchOperation.FILL_SLOT,
        PatchOperation.REPLACE_UNEXECUTED_FRAGMENT,
    ),
    barrier_profile: BarrierProfile = BarrierProfile.AFFECTED_SCOPE,
) -> ClosedSlot:
    """Derive a canonical typed cut from one already compiled graph."""

    target_set = set(target_activation_ids)
    nodes = {node.activation_id: node for node in graph.spec.activations}
    unknown = target_set - set(nodes)
    if unknown:
        raise ValueError(f"Cannot derive slot over unknown activations: {sorted(unknown)!r}.")
    incoming_control = {
        edge.target
        for edge in graph.spec.transitions
        if edge.source not in target_set and edge.target in target_set
    }
    if graph.spec.entry_activation in target_set:
        incoming_control.add(graph.spec.entry_activation)
    if control_entry_activation is None:
        if len(incoming_control) != 1:
            raise ValueError("A ClosedSlot requires one explicit primary control entry.")
        control_entry_activation = next(iter(incoming_control))

    input_cut: list[TypedInputCut] = []
    for activation_id in target_activation_ids:
        node = nodes[activation_id]
        ports = {port.name: port for port in node.inputs}
        for binding in node.bindings:
            is_boundary = isinstance(binding, ExternalBinding) or (
                isinstance(binding, ArtifactBinding)
                and binding.source_activation not in target_set
            )
            if is_boundary:
                input_cut.append(
                    TypedInputCut(
                        cut_id=_input_cut_id(activation_id, binding.input_port),
                        schema_id=ports[binding.input_port].schema_id,
                        target_activation=activation_id,
                        target_port=binding.input_port,
                    )
                )

    output_keys: set[tuple[str, str]] = set()
    for node in graph.spec.activations:
        if node.activation_id in target_set:
            continue
        for binding in node.bindings:
            if isinstance(binding, ArtifactBinding) and binding.source_activation in target_set:
                output_keys.add((binding.source_activation, binding.source_port))
    output_cut: list[TypedOutputCut] = []
    for activation_id, port_name in sorted(output_keys):
        port = next(
            port for port in nodes[activation_id].outputs if port.name == port_name
        )
        output_cut.append(
            TypedOutputCut(
                cut_id=_output_cut_id(activation_id, port_name),
                schema_id=port.schema_id,
                source_activation=activation_id,
                source_port=port_name,
            )
        )
    exits = tuple(
        ControlExitCut(
            cut_id=_exit_cut_id(edge.source, edge.outcome, edge.target),
            source_activation=edge.source,
            outcome=edge.outcome,
            target_activation=edge.target,
        )
        for edge in graph.spec.transitions
        if edge.source in target_set and edge.target not in target_set
    )
    return ClosedSlot(
        slot_id=slot_id,
        target_activation_ids=target_activation_ids,
        control_entry_activation=control_entry_activation,
        input_cut=tuple(input_cut),
        output_cut=tuple(output_cut),
        control_exit_cut=exits,
        effect_ceiling=effect_ceiling,
        capability_ceiling=capability_ceiling,
        budget_ceiling=budget_ceiling or ExecutionBudget(),
        verifier_obligations=verifier_obligations,
        allowed_operations=allowed_operations,
        barrier_profile=barrier_profile,
    )


class GraphPatchCoordinator:
    """Own optimistic, compile-before-commit updates for one live scheduler."""

    def __init__(
        self,
        *,
        scheduler: ActivationScheduler,
        frontier: ComposableFrontier,
        compiler: ElasticGraphCompiler | None = None,
        persist_frontier: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.compiler = compiler or ElasticGraphCompiler()
        self._lock = threading.RLock()
        graph = scheduler.graph
        if (
            frontier.graph_id != graph.spec.graph_id
            or frontier.revision != graph.spec.revision
            or frontier.graph_digest != graph.digest
        ):
            raise ValueError("ComposableFrontier is not bound to the scheduler graph revision.")
        self._validate_slots(graph, frontier.slots)
        self._slots = {slot.slot_id: slot for slot in frontier.slots}
        self._receipts: list[PatchReceipt] = []
        self._versions: list[GraphVersionRecord] = [
            GraphVersionRecord(
                revision=graph.spec.revision,
                digest=graph.digest,
                spec=graph.spec,
                accepted_receipt=None,
            )
        ]
        self._roster_updates: list[RosterUpdate] = []
        self._commits: list[GraphPatchCommit] = []
        if persist_frontier and scheduler.durable_state_enabled:
            scheduler.checkpoint(
                reason="graph_frontier",
                metadata={
                    "composable_frontier": frontier.model_dump(mode="json")
                },
            )

    @classmethod
    def recover_from_commits(
        cls,
        *,
        scheduler: ActivationScheduler,
        commits: tuple[GraphPatchCommit | dict, ...],
        compiler: ElasticGraphCompiler | None = None,
    ) -> GraphPatchCoordinator:
        """Restore the elastic frontier and audit chain after scheduler restart."""

        if not commits:
            raise ValueError("At least one GraphPatchCommit is required for recovery.")
        parsed = tuple(
            item if isinstance(item, GraphPatchCommit) else GraphPatchCommit.model_validate(item)
            for item in commits
        )
        graph_compiler = compiler or ElasticGraphCompiler()
        previous: GraphPatchCommit | None = None
        versions: list[GraphVersionRecord] = []
        for item in parsed:
            predecessor = graph_compiler.compile(item.predecessor_graph)
            if predecessor.digest != item.predecessor_digest:
                raise ValueError("GraphPatchCommit predecessor digest failed recompilation.")
            compiled = graph_compiler.compile(item.successor_graph)
            if compiled.digest != item.successor_digest:
                raise ValueError("GraphPatchCommit successor digest failed recompilation.")
            if (
                item.episode_id != scheduler.episode_id
                or item.workflow_id != scheduler.workflow_id
            ):
                raise ValueError("GraphPatchCommit belongs to another scheduler.")
            if previous is not None and (
                item.receipt.before_revision != previous.receipt.after_revision
                or item.receipt.before_digest != previous.receipt.after_digest
                or item.predecessor_graph != previous.successor_graph
            ):
                raise ValueError("GraphPatchCommit chain has a revision gap or fork.")
            if previous is None:
                versions.append(
                    GraphVersionRecord(
                        revision=predecessor.spec.revision,
                        digest=predecessor.digest,
                        spec=predecessor.spec,
                        accepted_receipt=None,
                    )
                )
            versions.append(
                GraphVersionRecord(
                    revision=compiled.spec.revision,
                    digest=compiled.digest,
                    spec=compiled.spec,
                    accepted_receipt=item.receipt,
                )
            )
            previous = item
        latest = parsed[-1]
        if (
            scheduler.graph.spec.revision != latest.receipt.after_revision
            or scheduler.graph.digest != latest.receipt.after_digest
        ):
            raise ValueError("Recovered scheduler does not match the latest patch commit.")
        coordinator = cls(
            scheduler=scheduler,
            frontier=latest.next_frontier,
            compiler=graph_compiler,
            persist_frontier=False,
        )
        coordinator._receipts = [item.receipt for item in parsed]
        coordinator._versions = versions
        coordinator._commits = list(parsed)
        return coordinator

    @property
    def frontier(self) -> ComposableFrontier:
        graph = self.scheduler.graph
        return ComposableFrontier(
            graph_id=graph.spec.graph_id,
            revision=graph.spec.revision,
            graph_digest=graph.digest,
            slots=tuple(self._slots.values()),
        )

    @property
    def receipts(self) -> tuple[PatchReceipt, ...]:
        return tuple(self._receipts)

    @property
    def versions(self) -> tuple[GraphVersionRecord, ...]:
        return tuple(self._versions)

    @property
    def roster_updates(self) -> tuple[RosterUpdate, ...]:
        return tuple(self._roster_updates)

    @property
    def commits(self) -> tuple[GraphPatchCommit, ...]:
        return tuple(self._commits)

    def record_roster_update(self, update: RosterUpdate) -> ComposableFrontier:
        """Audit a roster event without changing topology or graph revision."""

        with self._lock:
            if update.episode_id != self.scheduler.episode_id:
                raise ValueError("RosterUpdate belongs to a different episode.")
            if update.workflow_id != self.scheduler.workflow_id:
                raise ValueError("RosterUpdate belongs to a different workflow.")
            already_recorded = any(
                existing.event_id == update.event_id for existing in self._roster_updates
            )
            self.scheduler.on_event(update)
            if not already_recorded:
                self._roster_updates.append(update)
            return self.frontier

    def fill_slot(
        self,
        *,
        slot_id: str,
        base_revision: int,
        fragment: GraphFragment,
        opened_slots: tuple[ClosedSlot, ...] = (),
        patch_id: str | None = None,
        requested_by: str | None = None,
    ) -> PatchReceipt:
        return self.apply(
            GraphPatchProposal(
                patch_id=patch_id or uuid.uuid4().hex,
                operation=PatchOperation.FILL_SLOT,
                slot_id=slot_id,
                base_revision=base_revision,
                fragment=fragment,
                opened_slots=opened_slots,
                requested_by=requested_by,
            )
        )

    def replace_unexecuted_fragment(
        self,
        *,
        slot_id: str,
        base_revision: int,
        fragment: GraphFragment,
        opened_slots: tuple[ClosedSlot, ...] = (),
        patch_id: str | None = None,
        requested_by: str | None = None,
    ) -> PatchReceipt:
        return self.apply(
            GraphPatchProposal(
                patch_id=patch_id or uuid.uuid4().hex,
                operation=PatchOperation.REPLACE_UNEXECUTED_FRAGMENT,
                slot_id=slot_id,
                base_revision=base_revision,
                fragment=fragment,
                opened_slots=opened_slots,
                requested_by=requested_by,
            )
        )

    def apply(self, proposal: GraphPatchProposal) -> PatchReceipt:
        with self._lock:
            before = self.scheduler.graph
            if proposal.base_revision != before.spec.revision:
                return self._reject(
                    proposal,
                    PatchRejectCode.STALE_BASE_REVISION,
                    f"Patch base {proposal.base_revision} is stale; current revision is "
                    f"{before.spec.revision}.",
                )
            slot = self._slots.get(proposal.slot_id)
            if slot is None:
                return self._reject(
                    proposal,
                    PatchRejectCode.UNKNOWN_SLOT,
                    f"Slot {proposal.slot_id!r} is not open at this frontier.",
                )
            if proposal.operation not in slot.allowed_operations:
                return self._reject(
                    proposal,
                    PatchRejectCode.OPERATION_NOT_ALLOWED,
                    f"Slot {slot.slot_id!r} does not allow {proposal.operation.value!r}.",
                )
            if (
                proposal.operation == PatchOperation.FILL_SLOT
                and len(slot.target_activation_ids) != 1
            ):
                return self._reject(
                    proposal,
                    PatchRejectCode.OPERATION_NOT_ALLOWED,
                    "fill_slot is only valid for a single closed placeholder.",
                )

            try:
                self._validate_slot(before, slot)
                self._validate_target_history(slot)
                self._validate_barrier(slot)
                self._validate_fragment(slot, proposal.fragment, before)
                candidate_spec = self._materialize(before, slot, proposal)
                compiled = self.compiler.compile(candidate_spec)
                next_slot_map = {
                    existing_id: existing
                    for existing_id, existing in self._slots.items()
                    if existing_id != slot.slot_id
                }
                # Supplying an already-open id is an explicit frontier-contract
                # refresh, useful when this patch changes that slot's typed
                # boundary.  Proposal-local duplicate ids remain forbidden.
                for opened in proposal.opened_slots:
                    next_slot_map[opened.slot_id] = opened
                next_slots = tuple(next_slot_map.values())
                self._validate_slots(compiled, next_slots)
            except _PatchValidationError as exc:
                return self._reject(proposal, exc.code, str(exc))
            except (ValueError, TypeError) as exc:
                return self._reject(
                    proposal,
                    PatchRejectCode.COMPILE_FAILED,
                    f"Successor graph failed compile-before-commit: {exc}",
                )

            accepted = PatchReceipt(
                patch_id=proposal.patch_id,
                operation=proposal.operation,
                slot_id=proposal.slot_id,
                accepted=True,
                before_revision=before.spec.revision,
                after_revision=compiled.spec.revision,
                before_digest=before.digest,
                after_digest=compiled.digest,
                proposed_digest=compiled.digest,
            )
            patch_commit = GraphPatchCommit(
                episode_id=self.scheduler.episode_id,
                workflow_id=self.scheduler.workflow_id,
                proposal=proposal,
                receipt=accepted,
                predecessor_graph=before.spec,
                predecessor_digest=before.digest,
                successor_graph=compiled.spec,
                successor_digest=compiled.digest,
                next_frontier=ComposableFrontier(
                    graph_id=compiled.spec.graph_id,
                    revision=compiled.spec.revision,
                    graph_digest=compiled.digest,
                    slots=next_slots,
                ),
            )
            try:
                self.scheduler.replace_compiled_graph(
                    compiled,
                    expected_digest=before.digest,
                    removed_activation_ids=slot.target_activation_ids,
                    replacement_entry_activation=proposal.fragment.entry_activation,
                    commit_metadata={
                        "graph_patch_commit": patch_commit.model_dump(mode="json"),
                        "composable_frontier": patch_commit.next_frontier.model_dump(
                            mode="json"
                        ),
                    },
                )
            except SchedulerError as exc:
                return self._reject(
                    proposal,
                    PatchRejectCode.COMMIT_CONFLICT,
                    f"Scheduler rejected the atomic commit: {exc}",
                    proposed_digest=compiled.digest,
                )

            # No validation remains after scheduler commit.
            self._slots = {item.slot_id: item for item in next_slots}
            self._receipts.append(accepted)
            self._commits.append(patch_commit)
            self._versions.append(
                GraphVersionRecord(
                    revision=compiled.spec.revision,
                    digest=compiled.digest,
                    spec=compiled.spec,
                    accepted_receipt=accepted,
                )
            )
            return accepted

    def reject_preflight(
        self,
        proposal: GraphPatchProposal,
        *,
        reason: str,
        code: PatchRejectCode = PatchRejectCode.INVALID_FRAGMENT,
    ) -> PatchReceipt:
        """Record an EpisodeRuntime/catalog preflight rejection without mutation."""

        if not reason.strip():
            raise ValueError("preflight rejection reason must not be empty")
        with self._lock:
            return self._reject(proposal, code, reason)

    def _reject(
        self,
        proposal: GraphPatchProposal,
        code: PatchRejectCode,
        reason: str,
        *,
        proposed_digest: str | None = None,
    ) -> PatchReceipt:
        graph = self.scheduler.graph
        receipt = PatchReceipt(
            patch_id=proposal.patch_id,
            operation=proposal.operation,
            slot_id=proposal.slot_id,
            accepted=False,
            reason_codes=(code,),
            reasons=(reason,),
            before_revision=graph.spec.revision,
            after_revision=graph.spec.revision,
            before_digest=graph.digest,
            after_digest=graph.digest,
            proposed_digest=proposed_digest,
        )
        self._receipts.append(receipt)
        return receipt

    def _validate_slots(
        self, graph: CompiledElasticGraph, slots: tuple[ClosedSlot, ...]
    ) -> None:
        ids = [slot.slot_id for slot in slots]
        if len(ids) != len(set(ids)):
            raise ValueError("The next frontier contains duplicate slot ids.")
        covered: set[str] = set()
        for slot in slots:
            overlap = covered.intersection(slot.target_activation_ids)
            if overlap:
                raise ValueError(
                    f"Composable slots cannot overlap targets: {', '.join(sorted(overlap))}."
                )
            covered.update(slot.target_activation_ids)
            self._validate_slot(graph, slot)

    def _validate_slot(self, graph: CompiledElasticGraph, slot: ClosedSlot) -> None:
        node_map = {node.activation_id: node for node in graph.spec.activations}
        targets = set(slot.target_activation_ids)
        unknown = targets - set(node_map)
        if unknown:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                f"Slot {slot.slot_id!r} targets unknown activations: {sorted(unknown)!r}.",
            )
        if any(node_map[item].lane != ActivationLane.PRIMARY for item in targets):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                "Graph patches replace primary-control fragments; services use roster updates.",
            )
        incoming_targets = {
            edge.target
            for edge in graph.spec.transitions
            if edge.source not in targets and edge.target in targets
        }
        if graph.spec.entry_activation in targets:
            incoming_targets.add(graph.spec.entry_activation)
        if incoming_targets != {slot.control_entry_activation}:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                "ClosedSlot does not describe a single primary-control entry cut.",
            )

        expected_inputs: dict[tuple[str, str], str] = {}
        for activation_id in slot.target_activation_ids:
            node = node_map[activation_id]
            ports = {port.name: port for port in node.inputs}
            for binding in node.bindings:
                if isinstance(binding, ExternalBinding) or (
                    isinstance(binding, ArtifactBinding)
                    and binding.source_activation not in targets
                ):
                    expected_inputs[(activation_id, binding.input_port)] = ports[
                        binding.input_port
                    ].schema_id
        actual_inputs = {
            (item.target_activation, item.target_port): item.schema_id
            for item in slot.input_cut
        }
        if len(actual_inputs) != len(slot.input_cut) or actual_inputs != expected_inputs:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                "ClosedSlot input cut is incomplete or has a stale schema.",
            )

        expected_output_keys: set[tuple[str, str]] = set()
        for node in graph.spec.activations:
            if node.activation_id in targets:
                continue
            for binding in node.bindings:
                if isinstance(binding, ArtifactBinding) and binding.source_activation in targets:
                    expected_output_keys.add(
                        (binding.source_activation, binding.source_port)
                    )
        expected_outputs = {
            key: next(
                port.schema_id
                for port in node_map[key[0]].outputs
                if port.name == key[1]
            )
            for key in expected_output_keys
        }
        actual_outputs = {
            (item.source_activation, item.source_port): item.schema_id
            for item in slot.output_cut
        }
        if len(actual_outputs) != len(slot.output_cut) or actual_outputs != expected_outputs:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                "ClosedSlot output cut is incomplete or has a stale schema.",
            )

        expected_exits = {
            (edge.source, edge.outcome, edge.target)
            for edge in graph.spec.transitions
            if edge.source in targets and edge.target not in targets
        }
        actual_exits = {
            (item.source_activation, item.outcome, item.target_activation)
            for item in slot.control_exit_cut
        }
        if len(actual_exits) != len(slot.control_exit_cut) or actual_exits != expected_exits:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRONTIER,
                "ClosedSlot control-exit cut no longer matches the graph.",
            )

    def _validate_barrier(self, slot: ClosedSlot) -> None:
        snapshot = self.scheduler.snapshot()
        if slot.barrier_profile == BarrierProfile.GLOBAL_QUIESCENCE:
            running = [
                item.activation_id
                for item in snapshot.activations
                if item.status == ActivationStatus.RUNNING
            ]
            if running:
                raise _PatchValidationError(
                    PatchRejectCode.BARRIER_NOT_SATISFIED,
                    "Global-quiescence barrier has running activations: "
                    + ", ".join(sorted(running)),
                )
        else:
            target_set = set(slot.target_activation_ids)
            affected = set(target_set)
            for edge in self.scheduler.graph.spec.transitions:
                if edge.target in target_set:
                    # Its continuation target is rewritten by the patch; an
                    # in-flight command from this predecessor still carries
                    # the old graph revision.
                    affected.add(edge.source)
                if edge.source in target_set:
                    affected.add(edge.target)
            for node in self.scheduler.graph.spec.activations:
                if any(
                    isinstance(binding, ArtifactBinding)
                    and binding.source_activation in target_set
                    for binding in node.bindings
                ):
                    # The consumer binding is rewritten to a fragment export.
                    affected.add(node.activation_id)
            running = [
                item.activation_id
                for item in snapshot.activations
                if item.activation_id in affected
                and item.status == ActivationStatus.RUNNING
            ]
            if running:
                raise _PatchValidationError(
                    PatchRejectCode.BARRIER_NOT_SATISFIED,
                    "Affected-scope barrier has running targets: "
                    + ", ".join(sorted(running)),
                )

    def _validate_target_history(self, slot: ClosedSlot) -> None:
        by_id = {
            item.activation_id: item for item in self.scheduler.snapshot().activations
        }
        target_set = set(slot.target_activation_ids)
        rebound_consumers = {
            node.activation_id
            for node in self.scheduler.graph.spec.activations
            if any(
                isinstance(binding, ArtifactBinding)
                and binding.source_activation in target_set
                for binding in node.bindings
            )
        }
        immutable = [
            activation_id
            for activation_id in (*slot.target_activation_ids, *sorted(rebound_consumers))
            if by_id[activation_id].attempts > 0
            or by_id[activation_id].active_command_id is not None
            or (
                activation_id in target_set
                and by_id[activation_id].status
                not in {ActivationStatus.PENDING, ActivationStatus.READY}
            )
        ]
        if immutable:
            raise _PatchValidationError(
                PatchRejectCode.TARGET_IMMUTABLE,
                "Committed, running, or previously admitted affected nodes are immutable: "
                + ", ".join(sorted(immutable)),
            )

    def _validate_fragment(
        self,
        slot: ClosedSlot,
        fragment: GraphFragment,
        graph: CompiledElasticGraph,
    ) -> None:
        target_set = set(slot.target_activation_ids)
        outside_ids = {
            node.activation_id for node in graph.spec.activations
        } - target_set
        fragment_ids = {node.activation_id for node in fragment.activations}
        collisions = outside_ids.intersection(fragment_ids)
        if collisions:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment activation ids collide with surviving graph nodes: "
                + ", ".join(sorted(collisions)),
            )

        input_cuts = {item.cut_id: item for item in slot.input_cut}
        input_route_ids = {route.cut_id for route in fragment.input_routes}
        if input_route_ids != set(input_cuts):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment must consume every typed input cut and no undeclared cut.",
            )
        routed_ports = {
            (route.target_activation, route.target_port): route.cut_id
            for route in fragment.input_routes
        }
        if len(routed_ports) != len(fragment.input_routes):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "A fragment input port cannot be routed more than once.",
            )
        fragment_nodes = {node.activation_id: node for node in fragment.activations}
        for route in fragment.input_routes:
            cut = input_cuts[route.cut_id]
            node = fragment_nodes[route.target_activation]
            port = next(
                (item for item in node.inputs if item.name == route.target_port), None
            )
            if port is None or port.schema_id != cut.schema_id:
                raise _PatchValidationError(
                    PatchRejectCode.INVALID_FRAGMENT,
                    f"Input route {route.cut_id!r} does not preserve schema {cut.schema_id!r}.",
                )

        output_cuts = {item.cut_id: item for item in slot.output_cut}
        output_routes = {route.cut_id: route for route in fragment.output_routes}
        if len(output_routes) != len(fragment.output_routes) or set(output_routes) != set(
            output_cuts
        ):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment must produce every typed output cut and no undeclared cut.",
            )
        for cut_id, route in output_routes.items():
            output = next(
                (
                    item
                    for item in fragment_nodes[route.source_activation].outputs
                    if item.name == route.source_port
                ),
                None,
            )
            if output is None or output.schema_id != output_cuts[cut_id].schema_id:
                raise _PatchValidationError(
                    PatchRejectCode.INVALID_FRAGMENT,
                    f"Output route {cut_id!r} does not preserve its typed cut.",
                )

        exit_cuts = {item.cut_id: item for item in slot.control_exit_cut}
        exit_routes = {route.cut_id: route for route in fragment.exit_routes}
        if len(exit_routes) != len(fragment.exit_routes) or set(exit_routes) != set(exit_cuts):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment must map every control exit exactly once.",
            )

        replaces_terminal = bool(
            target_set.intersection(graph.spec.terminal_activations)
        )
        if replaces_terminal != bool(fragment.terminal_activations):
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment terminal declarations do not match the replaced terminal cut.",
            )

        max_effect = max(
            (_EFFECT_RANK[node.effect_scope] for node in fragment.activations),
            default=0,
        )
        if max_effect > _EFFECT_RANK[slot.effect_ceiling]:
            raise _PatchValidationError(
                PatchRejectCode.EFFECT_CEILING_EXCEEDED,
                f"Fragment effects exceed slot ceiling {slot.effect_ceiling.value!r}.",
            )

        capabilities: set[str] = set()
        for node in fragment.activations:
            capabilities.update(node.required_capabilities)
        excess = capabilities - set(slot.capability_ceiling)
        if excess:
            raise _PatchValidationError(
                PatchRejectCode.CAPABILITY_CEILING_EXCEEDED,
                "Fragment requests capabilities outside the ceiling: "
                + ", ".join(sorted(excess)),
            )

        consumed = self._fragment_budget(fragment)
        exceeded = [
            field
            for field in ExecutionBudget.model_fields
            if getattr(consumed, field) > getattr(slot.budget_ceiling, field)
        ]
        if exceeded:
            details = ", ".join(
                f"{field}={getattr(consumed, field)}>{getattr(slot.budget_ceiling, field)}"
                for field in exceeded
            )
            raise _PatchValidationError(
                PatchRejectCode.BUDGET_CEILING_EXCEEDED,
                "Fragment exceeds slot budget: " + details,
            )
        self._validate_verifiers(slot, fragment)

    @staticmethod
    def _fragment_budget(fragment: GraphFragment) -> ExecutionBudget:
        totals = dict.fromkeys(ExecutionBudget.model_fields, 0)
        for node in fragment.activations:
            raw_param_budget = node.params.get("budget", {})
            if not isinstance(raw_param_budget, dict):
                raise _PatchValidationError(
                    PatchRejectCode.INVALID_FRAGMENT,
                    "params.budget must be an object.",
                )
            for field in totals:
                declared = getattr(node.estimated_budget, field)
                raw = raw_param_budget.get(field, 0)
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(raw)
                    or raw < 0
                ):
                    raise _PatchValidationError(
                        PatchRejectCode.INVALID_FRAGMENT,
                        f"Invalid non-negative budget value for {field!r}.",
                    )
                value = max(declared, math.ceil(raw))
                if field == "authoritative_actions" and (
                    node.effect_scope == EffectScope.AUTHORITATIVE_WORLD
                ):
                    value = max(value, 1)
                if field == "shadow_rollouts" and node.effect_scope == EffectScope.SHADOW_WORLD:
                    value = max(value, 1)
                totals[field] += value
        return ExecutionBudget(**totals)

    @staticmethod
    def _validate_verifiers(slot: ClosedSlot, fragment: GraphFragment) -> None:
        nodes = {node.activation_id: node for node in fragment.activations}
        predecessors: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for edge in fragment.transitions:
            predecessors[edge.target].add(edge.source)
        dominators = {
            node_id: ({fragment.entry_activation} if node_id == fragment.entry_activation else set(nodes))
            for node_id in nodes
        }
        changed = True
        while changed:
            changed = False
            for node_id in sorted(set(nodes) - {fragment.entry_activation}):
                preds = predecessors[node_id]
                common = (
                    set.intersection(*(dominators[pred] for pred in preds))
                    if preds
                    else set()
                )
                updated = {node_id} | common
                if updated != dominators[node_id]:
                    dominators[node_id] = updated
                    changed = True

        for obligation in slot.verifier_obligations:
            candidates = []
            for node in fragment.activations:
                if obligation.verifier_tag not in node.verifier_tags:
                    continue
                if obligation.required_output_schema_id is not None and not any(
                    output.schema_id == obligation.required_output_schema_id
                    for output in node.outputs
                ):
                    continue
                candidates.append(node.activation_id)
            if not candidates:
                raise _PatchValidationError(
                    PatchRejectCode.MISSING_VERIFIER,
                    f"Fragment does not satisfy verifier obligation {obligation.obligation_id!r}.",
                )
            if not obligation.require_control_dominance:
                continue
            protected = [
                node.activation_id
                for node in fragment.activations
                if _EFFECT_RANK[node.effect_scope]
                >= _EFFECT_RANK[obligation.before_effect_scope]
            ]
            invalid = [
                node_id
                for node_id in protected
                if not any(
                    verifier != node_id and verifier in dominators[node_id]
                    for verifier in candidates
                )
            ]
            if invalid:
                raise _PatchValidationError(
                    PatchRejectCode.VERIFIER_ORDER_INVALID,
                    f"Verifier {obligation.obligation_id!r} does not dominate effects: "
                    + ", ".join(sorted(invalid)),
                )

    def _materialize(
        self,
        graph: CompiledElasticGraph,
        slot: ClosedSlot,
        proposal: GraphPatchProposal,
    ) -> ElasticGraphSpec:
        fragment = proposal.fragment
        targets = set(slot.target_activation_ids)
        old_nodes = {node.activation_id: node for node in graph.spec.activations}
        fragment_nodes = {node.activation_id: node for node in fragment.activations}
        input_cuts = {item.cut_id: item for item in slot.input_cut}

        # Resolve typed input routes by copying the immutable old boundary
        # binding.  The fragment cannot smuggle a new undeclared external ref.
        routed_ports: set[tuple[str, str]] = set()
        for route in fragment.input_routes:
            cut = input_cuts[route.cut_id]
            old_node = old_nodes[cut.target_activation]
            old_binding = next(
                binding for binding in old_node.bindings if binding.input_port == cut.target_port
            )
            node = fragment_nodes[route.target_activation]
            replacement = old_binding.model_copy(update={"input_port": route.target_port})
            bindings = tuple(
                binding
                for binding in node.bindings
                if binding.input_port != route.target_port
            ) + (replacement,)
            fragment_nodes[route.target_activation] = node.model_copy(
                update={"bindings": bindings}
            )
            routed_ports.add((route.target_activation, route.target_port))

        for node in fragment_nodes.values():
            for binding in node.bindings:
                if isinstance(binding, ArtifactBinding):
                    is_boundary = binding.source_activation not in fragment_nodes
                else:
                    is_boundary = True
                if is_boundary and (node.activation_id, binding.input_port) not in routed_ports:
                    raise _PatchValidationError(
                        PatchRejectCode.INVALID_FRAGMENT,
                        f"Fragment input {node.activation_id}.{binding.input_port} crosses an "
                        "undeclared typed cut.",
                    )

        output_routes = {item.cut_id: item for item in fragment.output_routes}
        output_cut_by_source = {
            (item.source_activation, item.source_port): item
            for item in slot.output_cut
        }
        rewritten_survivors: list[ActivationSpec] = []
        for node in graph.spec.activations:
            if node.activation_id in targets:
                continue
            bindings = []
            for binding in node.bindings:
                if isinstance(binding, ArtifactBinding):
                    cut = output_cut_by_source.get(
                        (binding.source_activation, binding.source_port)
                    )
                    if cut is not None:
                        route = output_routes[cut.cut_id]
                        binding = binding.model_copy(
                            update={
                                "source_activation": route.source_activation,
                                "source_port": route.source_port,
                            }
                        )
                bindings.append(binding)
            rewritten_survivors.append(node.model_copy(update={"bindings": tuple(bindings)}))

        # Preserve stable ordering by inserting the new fragment where the
        # first removed node appeared.
        activation_order: list[ActivationSpec] = []
        inserted = False
        survivor_map = {item.activation_id: item for item in rewritten_survivors}
        for old in graph.spec.activations:
            if old.activation_id in targets:
                if not inserted:
                    activation_order.extend(fragment_nodes[item.activation_id] for item in fragment.activations)
                    inserted = True
                continue
            activation_order.append(survivor_map[old.activation_id])

        transitions: list[TransitionSpec] = []
        for edge in graph.spec.transitions:
            source_in = edge.source in targets
            target_in = edge.target in targets
            if source_in:
                continue
            if target_in:
                transitions.append(edge.model_copy(update={"target": fragment.entry_activation}))
            else:
                transitions.append(edge)
        transitions.extend(fragment.transitions)
        exit_cuts = {item.cut_id: item for item in slot.control_exit_cut}
        for route in fragment.exit_routes:
            cut = exit_cuts[route.cut_id]
            transitions.append(
                TransitionSpec(
                    source=route.source_activation,
                    outcome=route.outcome,
                    target=cut.target_activation,
                )
            )

        loop_replacements = {
            item.loop_id: item for item in fragment.outer_loop_replacements
        }
        loops: list[BoundedLoopSpec] = []
        for loop in graph.spec.bounded_loops:
            members = set(loop.activation_ids)
            overlap = members.intersection(targets)
            if not overlap:
                loops.append(loop)
                continue
            if members.issubset(targets):
                continue
            replacement = loop_replacements.get(loop.loop_id)
            if replacement is None:
                raise _PatchValidationError(
                    PatchRejectCode.INVALID_FRAGMENT,
                    f"Partial outer loop {loop.loop_id!r} requires explicit membership routing.",
                )
            if not set(replacement.activation_ids).issubset(fragment_nodes):
                raise _PatchValidationError(
                    PatchRejectCode.INVALID_FRAGMENT,
                    f"Outer loop replacement {loop.loop_id!r} names non-fragment nodes.",
                )
            entry = loop.entry_activation
            if entry in targets:
                entry = replacement.entry_activation or fragment.entry_activation
            loops.append(
                loop.model_copy(
                    update={
                        "activation_ids": tuple(
                            item for item in loop.activation_ids if item not in targets
                        )
                        + replacement.activation_ids,
                        "entry_activation": entry,
                    }
                )
            )
        unused_loop_routes = set(loop_replacements) - {
            loop.loop_id
            for loop in graph.spec.bounded_loops
            if set(loop.activation_ids).intersection(targets)
            and not set(loop.activation_ids).issubset(targets)
        }
        if unused_loop_routes:
            raise _PatchValidationError(
                PatchRejectCode.INVALID_FRAGMENT,
                "Fragment supplies unused outer-loop replacements: "
                + ", ".join(sorted(unused_loop_routes)),
            )
        loops.extend(fragment.bounded_loops)

        terminal_order = tuple(
            item for item in graph.spec.terminal_activations if item not in targets
        ) + fragment.terminal_activations
        terminal_order = tuple(dict.fromkeys(terminal_order))

        metadata = dict(graph.spec.metadata)
        audit = list(metadata.get("elastic_patch_history", ()))
        audit.append(
            {
                "patch_id": proposal.patch_id,
                "operation": proposal.operation.value,
                "slot_id": slot.slot_id,
                "base_revision": graph.spec.revision,
                "requested_by": proposal.requested_by,
            }
        )
        metadata["elastic_patch_history"] = audit
        return ElasticGraphSpec(
            graph_id=graph.spec.graph_id,
            revision=graph.spec.revision + 1,
            entry_activation=(
                fragment.entry_activation
                if graph.spec.entry_activation in targets
                else graph.spec.entry_activation
            ),
            terminal_activations=terminal_order,
            activations=tuple(activation_order),
            transitions=tuple(transitions),
            bounded_loops=tuple(loops),
            metadata=metadata,
        )


__all__ = [
    "BarrierProfile",
    "ClosedSlot",
    "ComposableFrontier",
    "ControlExitCut",
    "FragmentExitRoute",
    "FragmentInputRoute",
    "FragmentOutputRoute",
    "GraphFragment",
    "GraphPatchCommit",
    "GraphPatchCoordinator",
    "GraphPatchProposal",
    "GraphVersionRecord",
    "OuterLoopReplacement",
    "PatchOperation",
    "PatchReceipt",
    "PatchRejectCode",
    "TypedInputCut",
    "TypedOutputCut",
    "VerifierObligation",
    "derive_closed_slot",
]
