"""Risk-adaptive Agent-Swarm protocol for bowl-on-plate correction.

This module upgrades only the correction segment of the production bowl-place
protocol.  Transport, release, verification, state reducers, tracking
services, and the recovery frontier remain the audited baseline.  Whenever
alignment needs adjustment, however, there is exactly one legal path::

    fresh action snapshot -> deterministic risk -> graph Arena
        -> runtime-promoted sealed MotionPlan -> authoritative execution

The graph contains no candidate list or risk threshold.  Those are trusted,
manifest-pinned production dependencies supplied by the orchestration layer.
Low risk therefore invokes one candidate and high risk invokes K without
allowing graph authors or Agents to change K, candidate profiles, physical
context refs, or hard-gate authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from robomex.authoring.monitoring import CompiledMonitorProgram
from robomex.elastic import (
    ActivationSpec,
    ArtifactBinding,
    BoundedLoopSpec,
    ClosedSlot,
    CompiledElasticGraph,
    ComposableFrontier,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    PatchOperation,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
    VerifierObligation,
    derive_closed_slot,
)
from robomex.orchestration.risk_provider import (
    DETERMINISTIC_MOTION_RISK_RUNNER_REF,
    MOTION_RISK_CAPABILITIES,
)
from robomex.protocols.bowl_place import (
    ADMISSION_SNAPSHOT,
    ALIGNMENT_ERROR,
    ATTACHMENT_EVIDENCE,
    MOTION_PLAN,
    OBSERVATION,
    RECOVERY_SAFETY_DECISION,
    SERVO_DECISION,
    BowlPlaceProtocolConfig,
    CodingPhaseBudgetConfig,
    build_bowl_attachment_monitor_program,
    build_bowl_place_graph,
)
from robomex.runtime.events import ControlOutcome

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

RISK_REPORT = "robomex.risk_report.v1"
ARENA_RESULT = "robomex.arena_result.v1"
ARENA_HYPOTHESES = "robomex.arena_hypotheses.v1"
ARENA_PROMOTION_RECEIPT = "robomex.arena_promotion_receipt.v1"

CORRECTION_RISK_ACTIVATION_ID = "assess_correction_risk"
CORRECTION_ARENA_ACTIVATION_ID = "correction_arena"
CORRECTION_SNAPSHOT_ACTIVATION_ID = "snapshot_correction"
CORRECTION_EXECUTE_ACTIVATION_ID = "execute_correction"
LEGACY_CORRECTION_PLANNER_ACTIVATION_ID = "plan_correction"

CORRECTION_CONTEXT_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "observation": OBSERVATION,
        "attachment_evidence": ATTACHMENT_EVIDENCE,
        "alignment_error": ALIGNMENT_ERROR,
        "servo_decision": SERVO_DECISION,
    }
)

_RISK_FAILURE_OUTCOMES = (
    ControlOutcome.FAILED,
    ControlOutcome.STALE_INPUT,
    ControlOutcome.STALE_OBSERVATION,
    ControlOutcome.UNCERTAIN,
    ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
    ControlOutcome.WRONG_GROUNDING,
    ControlOutcome.INFEASIBLE,
    ControlOutcome.EXHAUSTED,
)
_ARENA_FAILURE_OUTCOMES = (
    ControlOutcome.INFEASIBLE,
    ControlOutcome.EXHAUSTED,
    ControlOutcome.STALE_INPUT,
    ControlOutcome.FAILED,
)


class RiskAdaptiveBowlPlaceProtocolError(RuntimeError):
    """The graph or its production dependencies violate the swarm contract."""


class RiskAdaptiveBowlPlaceProtocolConfig(BowlPlaceProtocolConfig):
    """Compile-time graph and Arena budget contract.

    Candidate identities and risk thresholds are intentionally absent: they
    enter through the manifest-pinned production assembly, whose policy must
    exactly match ``max_arena_candidates``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["robomex.risk_adaptive_bowl_place_protocol_config.v1"] = (
        "robomex.risk_adaptive_bowl_place_protocol_config.v1"
    )
    graph_id: str = Field(
        default="bowl_on_plate_risk_adaptive_swarm",
        min_length=1,
    )
    correction_arena_binding_id: str = Field(
        default="robomex.bowl_place.correction_arena",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    max_arena_candidates: int = Field(default=3, ge=1, le=8)
    arena_candidate_budget: CodingPhaseBudgetConfig = Field(
        default_factory=CodingPhaseBudgetConfig,
        description="Per-candidate budget for one Arena activation.",
    )

    @property
    def correction_candidate_budget_limit(self) -> int:
        """Durable ledger quota for all possible correction rounds."""

        return self.max_alignment_iterations * self.max_arena_candidates


class BowlCorrectionCandidateStrategy(BaseModel):
    """Manifest-visible semantic role and hypothesis prior for one worker.

    The default candidates differ only in their planning objective. Until a
    trusted per-plan physical scorer supplies measured values, defaults must
    not preselect a winner through utility, risk, or claimed clearance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["robomex.bowl_correction_candidate_strategy.v1"] = (
        "robomex.bowl_correction_candidate_strategy.v1"
    )
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    strategy: NonEmptyStr
    objective: NonEmptyStr
    utility: float = Field(default=0.0, allow_inf_nan=False)
    estimated_risk: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    clearance_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    preconditions: tuple[NonEmptyStr, ...] = (
        "verified-held attachment",
        "fresh synchronized bowl and plate geometry",
        "bounded servo correction",
    )

    @field_validator("preconditions")
    @classmethod
    def _unique_preconditions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("candidate preconditions must be unique")
        return value


DEFAULT_CORRECTION_STRATEGIES: tuple[BowlCorrectionCandidateStrategy, ...] = (
    BowlCorrectionCandidateStrategy(
        candidate_id="direct",
        strategy="direct_bounded_servo",
        objective=(
            "Author one sealed joint-space plan for the exact bounded correction; "
            "prefer the shortest feasible path while preserving the admitted TCP pose."
        ),
    ),
    BowlCorrectionCandidateStrategy(
        candidate_id="clearance",
        strategy="clearance_biased_servo",
        objective=(
            "Author one sealed bounded-correction plan that maximizes obstacle and "
            "support clearance before optimizing path length."
        ),
    ),
    BowlCorrectionCandidateStrategy(
        candidate_id="conservative",
        strategy="conservative_waypoint_servo",
        objective=(
            "Author one conservative sealed bounded-correction plan using smooth "
            "joint motion and a low-risk waypoint geometry."
        ),
    ),
)


@dataclass(frozen=True)
class RiskAdaptiveBowlPlaceProtocol:
    spec: ElasticGraphSpec
    compiled: CompiledElasticGraph
    recovery_slot: ClosedSlot
    recovery_frontier: ComposableFrontier
    attachment_monitor_program: CompiledMonitorProgram


def _port(name: str, schema_id: str, *, required: bool = True) -> PortSpecV2:
    return PortSpecV2(name=name, schema_id=schema_id, required=required)


def _artifact(
    input_port: str,
    source_activation: str,
    source_port: str,
) -> ArtifactBinding:
    return ArtifactBinding(
        input_port=input_port,
        source_activation=source_activation,
        source_port=source_port,
    )


def _transition(source: str, outcome: ControlOutcome, target: str) -> TransitionSpec:
    return TransitionSpec(source=source, outcome=outcome, target=target)


def _risk_activation() -> ActivationSpec:
    return ActivationSpec(
        activation_id=CORRECTION_RISK_ACTIVATION_ID,
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref=DETERMINISTIC_MOTION_RISK_RUNNER_REF,
        inputs=tuple(_port(name, schema) for name, schema in CORRECTION_CONTEXT_SCHEMAS.items()),
        outputs=(_port("risk_report", RISK_REPORT),),
        bindings=(
            _artifact("observation", "capture_alignment", "observation"),
            _artifact(
                "attachment_evidence",
                "verify_alignment_attachment",
                "attachment_evidence",
            ),
            _artifact("alignment_error", "estimate_alignment", "alignment_error"),
            _artifact("servo_decision", "alignment_gate", "servo_decision"),
        ),
        required_capabilities=tuple(sorted(MOTION_RISK_CAPABILITIES)),
        verifier_tags=(
            "canonical-motion-risk",
            "same-snapshot-entity-and-revision",
            "complete-physical-evidence-lineage",
        ),
        estimated_budget=ExecutionBudget(),
    )


def _arena_activation(config: RiskAdaptiveBowlPlaceProtocolConfig) -> ActivationSpec:
    candidate_budget = config.arena_candidate_budget
    count = config.max_arena_candidates
    context_ports = tuple(
        _port(name, schema) for name, schema in CORRECTION_CONTEXT_SCHEMAS.items()
    )
    context_bindings = (
        _artifact("observation", "capture_alignment", "observation"),
        _artifact(
            "attachment_evidence",
            "verify_alignment_attachment",
            "attachment_evidence",
        ),
        _artifact("alignment_error", "estimate_alignment", "alignment_error"),
        _artifact("servo_decision", "alignment_gate", "servo_decision"),
    )
    return ActivationSpec(
        activation_id=CORRECTION_ARENA_ACTIVATION_ID,
        runner_kind=RunnerKind.ARENA,
        runner_ref=config.correction_arena_binding_id,
        effect_scope=EffectScope.READ_ONLY,
        inputs=(
            _port("snapshot", ADMISSION_SNAPSHOT),
            _port("risk", RISK_REPORT),
            *context_ports,
        ),
        outputs=(
            _port("result", ARENA_RESULT),
            _port("hypotheses", ARENA_HYPOTHESES),
            _port("promotion_receipt", ARENA_PROMOTION_RECEIPT, required=False),
            _port("selected_action_spec", MOTION_PLAN, required=False),
        ),
        bindings=(
            _artifact("snapshot", CORRECTION_SNAPSHOT_ACTIVATION_ID, "snapshot"),
            _artifact("risk", CORRECTION_RISK_ACTIVATION_ID, "risk_report"),
            *context_bindings,
        ),
        verifier_tags=(
            "runtime-motion-promotion",
            "fresh-admission-snapshot",
            "manifest-pinned-risk-adaptive-swarm",
        ),
        estimated_budget=ExecutionBudget(
            model_calls=count * candidate_budget.model_calls,
            tokens=count * candidate_budget.tokens,
            wall_time_ms=count * candidate_budget.wall_time_ms,
            actor_spawns=count,
        ),
    )


def _replace_correction_execution(node: ActivationSpec) -> ActivationSpec:
    bindings = tuple(
        _artifact(
            "action_spec",
            CORRECTION_ARENA_ACTIVATION_ID,
            "selected_action_spec",
        )
        if binding.input_port == "action_spec"
        else binding
        for binding in node.bindings
    )
    if not any(
        binding.input_port == "action_spec"
        and binding.source_activation == CORRECTION_ARENA_ACTIVATION_ID
        and binding.source_port == "selected_action_spec"
        for binding in bindings
    ):
        raise RiskAdaptiveBowlPlaceProtocolError(
            "execute_correction has no replaceable sealed action binding"
        )
    return node.model_copy(
        update={
            "bindings": bindings,
            "params": {
                **dict(node.params),
                "selected_by": CORRECTION_ARENA_ACTIVATION_ID,
            },
        }
    )


def build_risk_adaptive_bowl_place_graph(
    config: RiskAdaptiveBowlPlaceProtocolConfig | None = None,
) -> ElasticGraphSpec:
    """Replace the baseline singleton correction author with risk + Arena."""

    cfg = config or RiskAdaptiveBowlPlaceProtocolConfig()
    baseline = build_bowl_place_graph(cfg)
    risk_node = _risk_activation()
    arena_node = _arena_activation(cfg)

    activations: list[ActivationSpec] = []
    for node in baseline.activations:
        if node.activation_id == LEGACY_CORRECTION_PLANNER_ACTIVATION_ID:
            continue
        if node.activation_id == CORRECTION_EXECUTE_ACTIVATION_ID:
            activations.extend((risk_node, arena_node))
            node = _replace_correction_execution(node)
        activations.append(node)

    transitions = [
        edge
        for edge in baseline.transitions
        if edge.source != LEGACY_CORRECTION_PLANNER_ACTIVATION_ID
        and edge.target != LEGACY_CORRECTION_PLANNER_ACTIVATION_ID
    ]
    transitions.extend(
        (
            _transition(
                CORRECTION_SNAPSHOT_ACTIVATION_ID,
                ControlOutcome.SUCCESS,
                CORRECTION_RISK_ACTIVATION_ID,
            ),
            _transition(
                CORRECTION_RISK_ACTIVATION_ID,
                ControlOutcome.SUCCESS,
                CORRECTION_ARENA_ACTIVATION_ID,
            ),
            _transition(
                CORRECTION_ARENA_ACTIVATION_ID,
                ControlOutcome.SUCCESS,
                CORRECTION_EXECUTE_ACTIVATION_ID,
            ),
        )
    )
    transitions.extend(
        _transition(CORRECTION_RISK_ACTIVATION_ID, outcome, "recovery_frontier")
        for outcome in _RISK_FAILURE_OUTCOMES
    )
    transitions.extend(
        _transition(CORRECTION_ARENA_ACTIVATION_ID, outcome, "recovery_frontier")
        for outcome in _ARENA_FAILURE_OUTCOMES
    )

    loops: list[BoundedLoopSpec] = []
    for loop in baseline.bounded_loops:
        if loop.loop_id != "alignment_visual_servo":
            loops.append(loop)
            continue
        members: list[str] = []
        for activation_id in loop.activation_ids:
            if activation_id == LEGACY_CORRECTION_PLANNER_ACTIVATION_ID:
                members.extend((CORRECTION_RISK_ACTIVATION_ID, CORRECTION_ARENA_ACTIVATION_ID))
            else:
                members.append(activation_id)
        loops.append(loop.model_copy(update={"activation_ids": tuple(members)}))

    spec = ElasticGraphSpec(
        graph_id=baseline.graph_id,
        revision=baseline.revision,
        entry_activation=baseline.entry_activation,
        terminal_activations=baseline.terminal_activations,
        activations=tuple(activations),
        transitions=tuple(transitions),
        bounded_loops=tuple(loops),
        metadata={
            **dict(baseline.metadata),
            "profile": "risk_adaptive_agent_swarm_v1",
            "correction_control_path": (
                CORRECTION_SNAPSHOT_ACTIVATION_ID,
                CORRECTION_RISK_ACTIVATION_ID,
                CORRECTION_ARENA_ACTIVATION_ID,
                CORRECTION_EXECUTE_ACTIVATION_ID,
            ),
            "correction_arena": {
                "binding_id": cfg.correction_arena_binding_id,
                "max_candidates": cfg.max_arena_candidates,
                "context_schemas": dict(CORRECTION_CONTEXT_SCHEMAS),
                "selection_authority": "manifest_pinned_arena_binding",
                "physical_writer": CORRECTION_EXECUTE_ACTIVATION_ID,
            },
            "risk_authority": {
                "runner_ref": DETERMINISTIC_MOTION_RISK_RUNNER_REF,
                "policy_source": "manifest_pinned_provider_and_arena_binding",
                "low_risk_candidates": 1,
                "high_risk_candidates": cfg.max_arena_candidates,
            },
        },
    )
    _validate_risk_adaptive_graph(spec, cfg)
    return spec


def _validate_risk_adaptive_graph(
    spec: ElasticGraphSpec,
    config: RiskAdaptiveBowlPlaceProtocolConfig,
) -> None:
    nodes = {node.activation_id: node for node in spec.activations}
    required = {
        CORRECTION_SNAPSHOT_ACTIVATION_ID,
        CORRECTION_RISK_ACTIVATION_ID,
        CORRECTION_ARENA_ACTIVATION_ID,
        CORRECTION_EXECUTE_ACTIVATION_ID,
        "capture_alignment",
        "verify_alignment_attachment",
        "estimate_alignment",
        "alignment_gate",
        "recovery_frontier",
    }
    if not required.issubset(nodes) or LEGACY_CORRECTION_PLANNER_ACTIVATION_ID in nodes:
        raise RiskAdaptiveBowlPlaceProtocolError(
            "risk-adaptive correction nodes do not exactly replace the legacy planner"
        )
    risk = nodes[CORRECTION_RISK_ACTIVATION_ID]
    if (
        risk.runner_ref != DETERMINISTIC_MOTION_RISK_RUNNER_REF
        or {port.name: port.schema_id for port in risk.inputs} != dict(CORRECTION_CONTEXT_SCHEMAS)
        or {port.name: port.schema_id for port in risk.outputs} != {"risk_report": RISK_REPORT}
    ):
        raise RiskAdaptiveBowlPlaceProtocolError("deterministic risk node contract drifted")
    arena = nodes[CORRECTION_ARENA_ACTIVATION_ID]
    expected_arena_inputs = {
        "snapshot": ADMISSION_SNAPSHOT,
        "risk": RISK_REPORT,
        **dict(CORRECTION_CONTEXT_SCHEMAS),
    }
    if (
        arena.runner_kind is not RunnerKind.ARENA
        or arena.runner_ref != config.correction_arena_binding_id
        or {port.name: port.schema_id for port in arena.inputs} != expected_arena_inputs
        or arena.params
        or arena.required_capabilities
    ):
        raise RiskAdaptiveBowlPlaceProtocolError("graph-native Arena contract drifted")
    execute = nodes[CORRECTION_EXECUTE_ACTIVATION_ID]
    selected_bindings = [
        binding for binding in execute.bindings if binding.input_port == "action_spec"
    ]
    if len(selected_bindings) != 1 or (
        selected_bindings[0].source_activation != CORRECTION_ARENA_ACTIVATION_ID
        or selected_bindings[0].source_port != "selected_action_spec"
    ):
        raise RiskAdaptiveBowlPlaceProtocolError(
            "authoritative correction does not consume the Arena-promoted plan"
        )
    edges = {(edge.source, edge.outcome, edge.target) for edge in spec.transitions}
    required_success_path = {
        (
            CORRECTION_SNAPSHOT_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_RISK_ACTIVATION_ID,
        ),
        (
            CORRECTION_RISK_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_ARENA_ACTIVATION_ID,
        ),
        (
            CORRECTION_ARENA_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_EXECUTE_ACTIVATION_ID,
        ),
    }
    if not required_success_path.issubset(edges):
        raise RiskAdaptiveBowlPlaceProtocolError("correction success path has a bypass")
    expected_recovery = {
        (CORRECTION_RISK_ACTIVATION_ID, outcome, "recovery_frontier")
        for outcome in _RISK_FAILURE_OUTCOMES
    } | {
        (CORRECTION_ARENA_ACTIVATION_ID, outcome, "recovery_frontier")
        for outcome in _ARENA_FAILURE_OUTCOMES
    }
    if not expected_recovery.issubset(edges):
        raise RiskAdaptiveBowlPlaceProtocolError("risk/Arena failure routing is incomplete")
    loop = next(
        (item for item in spec.bounded_loops if item.loop_id == "alignment_visual_servo"),
        None,
    )
    if loop is None or not {
        CORRECTION_SNAPSHOT_ACTIVATION_ID,
        CORRECTION_RISK_ACTIVATION_ID,
        CORRECTION_ARENA_ACTIVATION_ID,
        CORRECTION_EXECUTE_ACTIVATION_ID,
    }.issubset(loop.activation_ids):
        raise RiskAdaptiveBowlPlaceProtocolError(
            "the complete risk-adaptive correction path must belong to the bounded loop"
        )


def build_risk_adaptive_bowl_place_protocol(
    config: RiskAdaptiveBowlPlaceProtocolConfig | None = None,
    *,
    compiler: ElasticGraphCompiler | None = None,
) -> RiskAdaptiveBowlPlaceProtocol:
    cfg = config or RiskAdaptiveBowlPlaceProtocolConfig()
    spec = build_risk_adaptive_bowl_place_graph(cfg)
    compiled = (compiler or ElasticGraphCompiler()).compile(spec)
    attachment_program = build_bowl_attachment_monitor_program(
        debounce_count=cfg.attachment_monitor_debounce_count
    )
    slot = derive_closed_slot(
        compiled,
        slot_id="bowl_place_recovery",
        target_activation_ids=("recovery_frontier",),
        control_entry_activation="recovery_frontier",
        effect_ceiling=EffectScope.AUTHORITATIVE_WORLD,
        capability_ceiling=(
            "episode.artifact.read",
            "perception.observe",
            "perception.geometry",
            "perception.track",
            "monitor.author",
            "motion.plan",
            "runtime.submit_sealed_action",
        ),
        budget_ceiling=ExecutionBudget(
            tokens=20_000,
            wall_time_ms=180_000,
            shadow_rollouts=4,
            authoritative_actions=5,
            actor_spawns=max(4, cfg.max_arena_candidates),
        ),
        verifier_obligations=(
            VerifierObligation(
                obligation_id="recovery-safety-gate",
                verifier_tag="recovery-safety-gate",
                required_output_schema_id=RECOVERY_SAFETY_DECISION,
                before_effect_scope=EffectScope.AUTHORITATIVE_WORLD,
            ),
        ),
        allowed_operations=(
            PatchOperation.FILL_SLOT,
            PatchOperation.REPLACE_UNEXECUTED_FRAGMENT,
        ),
    )
    frontier = ComposableFrontier(
        graph_id=compiled.spec.graph_id,
        revision=compiled.spec.revision,
        graph_digest=compiled.digest,
        slots=(slot,),
    )
    return RiskAdaptiveBowlPlaceProtocol(
        spec=spec,
        compiled=compiled,
        recovery_slot=slot,
        recovery_frontier=frontier,
        attachment_monitor_program=attachment_program,
    )


__all__ = [
    "ARENA_HYPOTHESES",
    "ARENA_PROMOTION_RECEIPT",
    "ARENA_RESULT",
    "CORRECTION_ARENA_ACTIVATION_ID",
    "CORRECTION_CONTEXT_SCHEMAS",
    "CORRECTION_EXECUTE_ACTIVATION_ID",
    "CORRECTION_RISK_ACTIVATION_ID",
    "CORRECTION_SNAPSHOT_ACTIVATION_ID",
    "DEFAULT_CORRECTION_STRATEGIES",
    "LEGACY_CORRECTION_PLANNER_ACTIVATION_ID",
    "RISK_REPORT",
    "BowlCorrectionCandidateStrategy",
    "RiskAdaptiveBowlPlaceProtocol",
    "RiskAdaptiveBowlPlaceProtocolConfig",
    "RiskAdaptiveBowlPlaceProtocolError",
    "build_risk_adaptive_bowl_place_graph",
    "build_risk_adaptive_bowl_place_protocol",
]
