"""Fixed-topology RoboMEx v2 protocol for bowl-on-plate placement.

The graph makes every physical phase visible and independently receipted.  Its
only cycle is the predeclared bounded visual-servo loop.  Normal correction
does not wake the Manager; exceptional closed outcomes route to a typed
``ComposableFrontier`` whose placeholder may be replaced by a recovery patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from robomex.authoring.monitoring import (
    CompiledMonitorProgram,
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
)
from robomex.elastic import (
    ACTION_SNAPSHOT_RUNNER_REF,
    ACTION_SNAPSHOT_SCHEMA_ID,
    ActivationLane,
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
    ExternalBinding,
    LifecycleScope,
    PatchOperation,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
    VerifierObligation,
    derive_closed_slot,
)
from robomex.runtime.events import ControlOutcome

ATTACHMENT_STATE = "robomex.attachment_state.v1"
ATTACHMENT_GUARD = "robomex.attachment_guard.v1"
ATTACHMENT_EVIDENCE = "robomex.attachment_evidence.v1"
OBSERVATION = "robomex.bowl_place_observation.v1"
HELD_BOWL = "robomex.held_bowl_estimate.v1"
PLATE_TARGET = "robomex.plate_support_target.v1"
ALIGNMENT_ERROR = "robomex.alignment_error.v1"
SERVO_DECISION = "robomex.servo_decision.v1"
MOTION_PLAN = "robomex.motion_plan.v2"
MONITOR_PROGRAM = "robomex.monitor_program.v1"
GRIPPER_COMMAND = "robomex.gripper_command.v1"
WAIT_SPEC = "robomex.wait_spec.v1"
EXECUTION_RECEIPT = "robomex.execution_receipt.v2"
CHECKPOINT = "robomex.phase_checkpoint.v1"
PLACEMENT_VERDICT = "robomex.placement_verdict.v1"
RECOVERY_SAFETY_DECISION = "robomex.recovery_safety_decision.v1"
RELATION_EVIDENCE = "robomex.relation_evidence.v1"
STATE_PROPOSAL = "robomex.state_transition_proposal.v1"
STATE_RECEIPT = "robomex.state_commit_receipt.v1"
ADMISSION_SNAPSHOT = ACTION_SNAPSHOT_SCHEMA_ID

_ATTACHMENT_MONITOR_SOURCE = """
def evaluate(sample):
    if sample["identity_match"] == False:
        return {
            "finding": "unsafe_deviation",
            "severity": "critical",
            "confidence": 1.0,
            "details": {"reason": "held_entity_identity_swap"},
        }
    if sample["held_entity_visible"] == False:
        return {
            "finding": "unobservable",
            "severity": "critical",
            "details": {"reason": "held_entity_occluded"},
        }
    if sample["attachment_status"] != "verified_held":
        return {
            "finding": "attachment_anomaly",
            "severity": "critical",
            "confidence": 1.0,
            "details": {"reason": "attachment_not_verified_held"},
        }
    return None
"""


class CodingPhaseBudgetConfig(BaseModel):
    """Manifest-visible upper grant for one skill-augmented coding activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["robomex.coding_phase_budget.v1"] = "robomex.coding_phase_budget.v1"
    model_calls: int = Field(default=3, ge=1, le=32)
    tokens: int = Field(default=64_000, ge=1, le=1_000_000)
    wall_time_ms: int = Field(default=60_000, ge=1_000, le=600_000)

    def execution_budget(self) -> ExecutionBudget:
        return ExecutionBudget(
            model_calls=self.model_calls,
            tokens=self.tokens,
            wall_time_ms=self.wall_time_ms,
        )


class BowlPlaceProtocolConfig(BaseModel):
    """Compile-time limits; changing one produces a new graph digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["robomex.bowl_place_protocol_config.v1"] = (
        "robomex.bowl_place_protocol_config.v1"
    )
    graph_id: str = Field(default="bowl_on_plate_visual_servo", min_length=1)
    revision: int = Field(default=1, ge=1)
    max_alignment_iterations: int = Field(default=6, ge=1, le=1000)
    max_step_translation_m: float = Field(default=0.015, gt=0, allow_inf_nan=False)
    max_step_yaw_rad: float = Field(default=0.12, gt=0, allow_inf_nan=False)
    max_cumulative_translation_m: float = Field(default=0.06, gt=0, allow_inf_nan=False)
    max_cumulative_yaw_rad: float = Field(default=0.40, gt=0, allow_inf_nan=False)
    tolerance_xy_m: float = Field(default=0.006, gt=0, allow_inf_nan=False)
    tolerance_z_m: float = Field(default=0.006, gt=0, allow_inf_nan=False)
    tolerance_yaw_rad: float = Field(default=0.08, gt=0, allow_inf_nan=False)
    settle_duration_s: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    attachment_monitor_debounce_count: int = Field(default=1, ge=1, le=100)
    coding_budget: CodingPhaseBudgetConfig = Field(default_factory=CodingPhaseBudgetConfig)
    authority_world_id: str = Field(default="authoritative", min_length=1, max_length=256)
    arm_resource_id: str = Field(default="robot.arm", min_length=1, max_length=256)
    gripper_resource_id: str = Field(default="robot.gripper", min_length=1, max_length=256)
    controller_resource_id: str = Field(default="robot.controller", min_length=1, max_length=256)


@dataclass(frozen=True)
class FixedBowlPlaceProtocol:
    """Compiled fixed protocol plus its revision-bound recovery frontier."""

    spec: ElasticGraphSpec
    compiled: CompiledElasticGraph
    recovery_slot: ClosedSlot
    recovery_frontier: ComposableFrontier
    attachment_monitor_program: CompiledMonitorProgram


def build_bowl_attachment_monitor_program(
    *,
    debounce_count: int = 1,
) -> CompiledMonitorProgram:
    """Compile the baseline read-only attachment/identity monitor."""

    return MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id="bowl_attachment_guard",
            source=_ATTACHMENT_MONITOR_SOURCE,
            hook=MonitorHook.CONTROL,
            allowed_signals=(
                "attachment_status",
                "held_entity_visible",
                "identity_match",
            ),
            debounce_count=debounce_count,
            max_runtime_ms=5.0,
        )
    )


def _port(name: str, schema_id: str) -> PortSpecV2:
    return PortSpecV2(name=name, schema_id=schema_id)


def _artifact(input_port: str, source_activation: str, source_port: str) -> ArtifactBinding:
    return ArtifactBinding(
        input_port=input_port,
        source_activation=source_activation,
        source_port=source_port,
    )


def _external(input_port: str, ref: str, schema_id: str) -> ExternalBinding:
    return ExternalBinding(input_port=input_port, ref=ref, schema_id=schema_id)


def _transition(source: str, outcome: ControlOutcome, target: str) -> TransitionSpec:
    return TransitionSpec(source=source, outcome=outcome, target=target)


def _planner(
    activation_id: str,
    *,
    runner_ref: str,
    action_schema: str,
    inputs: tuple[PortSpecV2, ...] = (),
    bindings: tuple[ArtifactBinding | ExternalBinding, ...] = (),
    snapshot_producer: str,
    coding_budget: CodingPhaseBudgetConfig,
    params: dict[str, object] | None = None,
) -> ActivationSpec:
    return ActivationSpec(
        activation_id=activation_id,
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref=runner_ref,
        inputs=(*inputs, _port("snapshot", ADMISSION_SNAPSHOT)),
        outputs=(_port("action_spec", action_schema),),
        bindings=(*bindings, _artifact("snapshot", snapshot_producer, "snapshot")),
        required_capabilities=("motion.plan",),
        estimated_budget=coding_budget.execution_budget(),
        params=params or {},
    )


def _system_action(
    activation_id: str,
    *,
    runner_ref: str,
    action_schema: str,
    producer: str,
    resource: str,
    world_id: str,
    phase: str,
    monitor_producer: str | None = None,
) -> ActivationSpec:
    """Every physical phase consumes one sealed spec and emits one receipt."""

    inputs = [_port("action_spec", action_schema)]
    bindings: list[ArtifactBinding | ExternalBinding] = [
        _artifact("action_spec", producer, "action_spec")
    ]
    params: dict[str, object] = {
        "phase": phase,
        "sealed_input": True,
        "receipt_authority": "runtime",
    }
    if monitor_producer is not None:
        inputs.append(_port("monitor_program", MONITOR_PROGRAM))
        bindings.append(_artifact("monitor_program", monitor_producer, "monitor_program"))
        params["monitor_input_port"] = "monitor_program"

    return ActivationSpec(
        activation_id=activation_id,
        runner_kind=RunnerKind.SYSTEM_ACTION,
        runner_ref=runner_ref,
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        authority_world_id=world_id,
        authoritative_resource=resource,
        inputs=tuple(inputs),
        outputs=(_port("receipt", EXECUTION_RECEIPT),),
        bindings=tuple(bindings),
        required_capabilities=("runtime.submit_sealed_action",),
        estimated_budget=ExecutionBudget(
            wall_time_ms=60_000,
            authoritative_actions=1,
        ),
        params=params,
    )


def _action_snapshot(
    activation_id: str,
    *,
    world_id: str,
    resource: str,
) -> ActivationSpec:
    """Runtime-owned fresh physical precondition for exactly one action author."""

    return ActivationSpec(
        activation_id=activation_id,
        runner_kind=RunnerKind.ACTION_SNAPSHOT,
        runner_ref=ACTION_SNAPSHOT_RUNNER_REF,
        authority_world_id=world_id,
        authoritative_resource=resource,
        outputs=(_port("snapshot", ADMISSION_SNAPSHOT),),
        estimated_budget=ExecutionBudget(),
    )


def _recovery_edges(source: str, outcomes: tuple[ControlOutcome, ...]) -> list[TransitionSpec]:
    return [_transition(source, outcome, "recovery_frontier") for outcome in outcomes]


def _state_reducer(activation_id: str, *, producer: str) -> ActivationSpec:
    """Runtime-owned reducer: an Agent may propose state, never commit it."""

    return ActivationSpec(
        activation_id=activation_id,
        runner_kind=RunnerKind.REDUCER,
        runner_ref="robomex.runtime.embodied_state_reducer",
        inputs=(_port("proposal", STATE_PROPOSAL),),
        outputs=(_port("receipt", STATE_RECEIPT),),
        bindings=(_artifact("proposal", producer, "proposal"),),
    )


def build_bowl_place_graph(
    config: BowlPlaceProtocolConfig | None = None,
) -> ElasticGraphSpec:
    """Build the fresh-evidence, reducer-authoritative placement graph.

    Trackers remain long-lived read-only services.  Every control decision is
    instead based on a primary synchronized checkpoint captured after the
    preceding world-changing action.  No workflow-open ``ExternalBinding`` is
    permitted to stand in for current physical evidence.
    """

    cfg = config or BowlPlaceProtocolConfig()
    author_attachment_monitor = ActivationSpec(
        activation_id="author_attachment_monitor",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.author_attachment_monitor",
        outputs=(_port("monitor_program", MONITOR_PROGRAM),),
        required_capabilities=("monitor.author",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={
            "objective": "Author a restricted online held-object safety guard.",
            "monitor_hook": MonitorHook.CONTROL.value,
            "allowed_signals": (
                "attachment_status",
                "held_entity_visible",
                "identity_match",
            ),
            "debounce_count": cfg.attachment_monitor_debounce_count,
        },
    )

    capture_initial = ActivationSpec(
        activation_id="capture_initial_attachment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.capture_synchronized_checkpoint",
        outputs=(_port("observation", OBSERVATION),),
        required_capabilities=("perception.observe",),
        verifier_tags=("synchronized-bowl-plate-snapshot",),
        params={"phase": "initial_attachment", "source": "observation_registry"},
    )
    verify_initial_attachment = ActivationSpec(
        activation_id="verify_initial_attachment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.verify_attachment_evidence",
        inputs=(_port("observation", OBSERVATION),),
        outputs=(
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
        ),
        bindings=(_artifact("observation", capture_initial.activation_id, "observation"),),
        verifier_tags=("attachment-verified-held", "fresh-observation"),
        params={"required_status": "verified_held", "phase": "transport"},
    )
    propose_initial_attachment = ActivationSpec(
        activation_id="propose_initial_attachment",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.propose_attachment_transition",
        inputs=(
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
        ),
        outputs=(_port("proposal", STATE_PROPOSAL),),
        bindings=(
            _artifact("observation", capture_initial.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_initial_attachment.activation_id,
                "attachment_evidence",
            ),
        ),
        required_capabilities=("state.propose",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={"target_status": "verified_held"},
    )
    commit_initial_attachment = _state_reducer(
        "commit_initial_attachment", producer=propose_initial_attachment.activation_id
    )
    snapshot_transport = _action_snapshot(
        "snapshot_transport",
        world_id=cfg.authority_world_id,
        resource=cfg.arm_resource_id,
    )
    plan_transport = _planner(
        "plan_transport",
        runner_ref="robomex.bowl_place.plan_transport_to_hover",
        action_schema=MOTION_PLAN,
        inputs=(
            _port("observation", OBSERVATION),
            _port("attachment_guard", ATTACHMENT_GUARD),
            _port("state_receipt", STATE_RECEIPT),
        ),
        bindings=(
            _artifact("observation", capture_initial.activation_id, "observation"),
            _artifact(
                "attachment_guard", verify_initial_attachment.activation_id, "attachment_guard"
            ),
            _artifact("state_receipt", commit_initial_attachment.activation_id, "receipt"),
        ),
        snapshot_producer=snapshot_transport.activation_id,
        coding_budget=cfg.coding_budget,
        params={
            "plan_kind": "transport_to_safe_hover",
            "tcp_frame_id": "panda_hand",
            "planner_backend": "curobo",
        },
    )
    execute_transport = _system_action(
        "execute_transport",
        runner_ref="robomex.runtime.execute_motion_plan",
        action_schema=MOTION_PLAN,
        producer=plan_transport.activation_id,
        resource=cfg.arm_resource_id,
        world_id=cfg.authority_world_id,
        phase="transport",
        monitor_producer=author_attachment_monitor.activation_id,
    )

    # This is the single loop entry.  The control edge guarantees that it runs
    # only after transport or correction completed; the emitted artifact binds
    # its lineage to that latest physical receipt.
    capture_alignment = ActivationSpec(
        activation_id="capture_alignment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.capture_synchronized_checkpoint",
        outputs=(_port("observation", OBSERVATION),),
        required_capabilities=("perception.observe",),
        verifier_tags=("synchronized-bowl-plate-snapshot", "post-action-freshness"),
        params={
            "phase": "alignment",
            "source": "observation_registry",
            "causal_predecessors": ("execute_transport", "execute_correction"),
            "require_new_generation_and_revision": True,
        },
    )
    verify_alignment_attachment = ActivationSpec(
        activation_id="verify_alignment_attachment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.verify_attachment_evidence",
        inputs=(_port("observation", OBSERVATION),),
        outputs=(
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
        ),
        bindings=(_artifact("observation", capture_alignment.activation_id, "observation"),),
        verifier_tags=("attachment-verified-held", "post-action-freshness"),
        params={"required_status": "verified_held", "phase": "correction"},
    )
    estimate_alignment = ActivationSpec(
        activation_id="estimate_alignment",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.estimate_support_alignment",
        inputs=(
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
        ),
        outputs=(
            _port("held_bowl", HELD_BOWL),
            _port("plate_target", PLATE_TARGET),
            _port("alignment_error", ALIGNMENT_ERROR),
        ),
        bindings=(
            _artifact("observation", capture_alignment.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_alignment_attachment.activation_id,
                "attachment_evidence",
            ),
        ),
        required_capabilities=("perception.geometry",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={
            "error_space": "translation_plus_yaw",
            "fail_on_mixed_snapshot": True,
            "tolerance_xy_m": cfg.tolerance_xy_m,
            "tolerance_z_m": cfg.tolerance_z_m,
            "tolerance_yaw_rad": cfg.tolerance_yaw_rad,
        },
    )
    alignment_gate = ActivationSpec(
        activation_id="alignment_gate",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.gate_bounded_correction",
        inputs=(
            _port("observation", OBSERVATION),
            _port("alignment_error", ALIGNMENT_ERROR),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
        ),
        outputs=(_port("servo_decision", SERVO_DECISION),),
        bindings=(
            _artifact("observation", capture_alignment.activation_id, "observation"),
            _artifact("alignment_error", estimate_alignment.activation_id, "alignment_error"),
            _artifact(
                "attachment_evidence",
                verify_alignment_attachment.activation_id,
                "attachment_evidence",
            ),
            _artifact(
                "attachment_guard",
                verify_alignment_attachment.activation_id,
                "attachment_guard",
            ),
        ),
        verifier_tags=("alignment-freshness", "attachment-verified-held"),
        params={
            "max_step_translation_m": cfg.max_step_translation_m,
            "max_step_yaw_rad": cfg.max_step_yaw_rad,
            "max_cumulative_translation_m": cfg.max_cumulative_translation_m,
            "max_cumulative_yaw_rad": cfg.max_cumulative_yaw_rad,
            "max_iterations": cfg.max_alignment_iterations,
            "tolerance_xy_m": cfg.tolerance_xy_m,
            "tolerance_z_m": cfg.tolerance_z_m,
            "tolerance_yaw_rad": cfg.tolerance_yaw_rad,
            "requires_fresh_generation_and_revision": True,
        },
    )
    snapshot_correction = _action_snapshot(
        "snapshot_correction",
        world_id=cfg.authority_world_id,
        resource=cfg.arm_resource_id,
    )
    plan_correction = _planner(
        "plan_correction",
        runner_ref="robomex.bowl_place.plan_bounded_correction",
        action_schema=MOTION_PLAN,
        inputs=(
            _port("servo_decision", SERVO_DECISION),
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
        ),
        bindings=(
            _artifact("servo_decision", alignment_gate.activation_id, "servo_decision"),
            _artifact("observation", capture_alignment.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_alignment_attachment.activation_id,
                "attachment_evidence",
            ),
        ),
        snapshot_producer=snapshot_correction.activation_id,
        coding_budget=cfg.coding_budget,
        params={
            "plan_kind": "bounded_correction",
            "tcp_frame_id": "panda_hand",
            "planner_backend": "curobo",
        },
    )
    execute_correction = _system_action(
        "execute_correction",
        runner_ref="robomex.runtime.execute_motion_plan",
        action_schema=MOTION_PLAN,
        producer=plan_correction.activation_id,
        resource=cfg.arm_resource_id,
        world_id=cfg.authority_world_id,
        phase="bounded_correction",
        monitor_producer=author_attachment_monitor.activation_id,
    )
    snapshot_descend = _action_snapshot(
        "snapshot_descend",
        world_id=cfg.authority_world_id,
        resource=cfg.arm_resource_id,
    )
    plan_descend = _planner(
        "plan_descend",
        runner_ref="robomex.bowl_place.plan_bounded_descend",
        action_schema=MOTION_PLAN,
        inputs=(
            _port("servo_decision", SERVO_DECISION),
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
        ),
        bindings=(
            _artifact("servo_decision", alignment_gate.activation_id, "servo_decision"),
            _artifact("observation", capture_alignment.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_alignment_attachment.activation_id,
                "attachment_evidence",
            ),
            _artifact(
                "attachment_guard",
                verify_alignment_attachment.activation_id,
                "attachment_guard",
            ),
        ),
        snapshot_producer=snapshot_descend.activation_id,
        coding_budget=cfg.coding_budget,
        params={
            "required_alignment": "within_tolerance",
            "plan_kind": "descend_to_release",
            "tcp_frame_id": "panda_hand",
            "planner_backend": "curobo",
        },
    )
    execute_descend = _system_action(
        "execute_descend",
        runner_ref="robomex.runtime.execute_motion_plan",
        action_schema=MOTION_PLAN,
        producer=plan_descend.activation_id,
        resource=cfg.arm_resource_id,
        world_id=cfg.authority_world_id,
        phase="descend",
        monitor_producer=author_attachment_monitor.activation_id,
    )

    capture_pre_release = ActivationSpec(
        activation_id="capture_pre_release",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.capture_synchronized_checkpoint",
        inputs=(_port("receipt", EXECUTION_RECEIPT),),
        outputs=(_port("observation", OBSERVATION),),
        bindings=(_artifact("receipt", execute_descend.activation_id, "receipt"),),
        required_capabilities=("perception.observe",),
        verifier_tags=("post-descend-freshness",),
        params={"phase": "pre_release", "source": "observation_registry"},
    )
    verify_pre_release_attachment = ActivationSpec(
        activation_id="verify_pre_release_attachment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.verify_attachment_evidence",
        inputs=(
            _port("observation", OBSERVATION),
            _port("receipt", EXECUTION_RECEIPT),
        ),
        outputs=(
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
        ),
        bindings=(
            _artifact("observation", capture_pre_release.activation_id, "observation"),
            _artifact("receipt", execute_descend.activation_id, "receipt"),
        ),
        verifier_tags=("attachment-verified-held", "post-descend-freshness"),
        params={"required_status": "verified_held", "phase": "open"},
    )
    estimate_pre_release_alignment = ActivationSpec(
        activation_id="estimate_pre_release_alignment",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.estimate_support_alignment",
        inputs=(
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
        ),
        outputs=(_port("alignment_error", ALIGNMENT_ERROR),),
        bindings=(
            _artifact("observation", capture_pre_release.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_pre_release_attachment.activation_id,
                "attachment_evidence",
            ),
        ),
        required_capabilities=("perception.geometry",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={
            "phase": "pre_release",
            "fail_on_mixed_snapshot": True,
            "tolerance_xy_m": cfg.tolerance_xy_m,
            "tolerance_z_m": cfg.tolerance_z_m,
            "tolerance_yaw_rad": cfg.tolerance_yaw_rad,
        },
    )
    checkpoint_pre_release = ActivationSpec(
        activation_id="checkpoint_pre_release",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.pre_release_checkpoint",
        inputs=(
            _port("receipt", EXECUTION_RECEIPT),
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("attachment_guard", ATTACHMENT_GUARD),
            _port("alignment_error", ALIGNMENT_ERROR),
        ),
        outputs=(_port("checkpoint", CHECKPOINT),),
        bindings=(
            _artifact("receipt", execute_descend.activation_id, "receipt"),
            _artifact("observation", capture_pre_release.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_pre_release_attachment.activation_id,
                "attachment_evidence",
            ),
            _artifact(
                "attachment_guard",
                verify_pre_release_attachment.activation_id,
                "attachment_guard",
            ),
            _artifact(
                "alignment_error",
                estimate_pre_release_alignment.activation_id,
                "alignment_error",
            ),
        ),
        verifier_tags=("attachment-verified-held", "alignment-within-tolerance"),
        params={"phase": "pre_release", "requires_fresh_alignment": True},
    )
    snapshot_open = _action_snapshot(
        "snapshot_open",
        world_id=cfg.authority_world_id,
        resource=cfg.gripper_resource_id,
    )
    plan_open = ActivationSpec(
        activation_id="plan_open",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.build_open_command",
        inputs=(
            _port("checkpoint", CHECKPOINT),
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
            _port("alignment_error", ALIGNMENT_ERROR),
            _port("snapshot", ADMISSION_SNAPSHOT),
        ),
        outputs=(_port("action_spec", GRIPPER_COMMAND),),
        bindings=(
            _artifact("checkpoint", checkpoint_pre_release.activation_id, "checkpoint"),
            _artifact("observation", capture_pre_release.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_pre_release_attachment.activation_id,
                "attachment_evidence",
            ),
            _artifact(
                "alignment_error",
                estimate_pre_release_alignment.activation_id,
                "alignment_error",
            ),
            _artifact("snapshot", snapshot_open.activation_id, "snapshot"),
        ),
        verifier_tags=("fresh-open-admission",),
        params={"mode": "open", "required_attachment": "verified_held"},
    )
    execute_open = _system_action(
        "execute_open",
        runner_ref="robomex.runtime.execute_gripper_command",
        action_schema=GRIPPER_COMMAND,
        producer=plan_open.activation_id,
        resource=cfg.gripper_resource_id,
        world_id=cfg.authority_world_id,
        phase="open",
    )
    snapshot_settle = _action_snapshot(
        "snapshot_settle",
        world_id=cfg.authority_world_id,
        resource=cfg.controller_resource_id,
    )
    plan_settle = ActivationSpec(
        activation_id="plan_settle",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.build_settle_wait",
        inputs=(
            _port("receipt", EXECUTION_RECEIPT),
            _port("snapshot", ADMISSION_SNAPSHOT),
        ),
        outputs=(_port("action_spec", WAIT_SPEC),),
        bindings=(
            _artifact("receipt", execute_open.activation_id, "receipt"),
            _artifact("snapshot", snapshot_settle.activation_id, "snapshot"),
        ),
        params={"duration_s": cfg.settle_duration_s, "hold_command": "hold_current"},
    )
    execute_settle = _system_action(
        "execute_settle",
        runner_ref="robomex.runtime.execute_wait",
        action_schema=WAIT_SPEC,
        producer=plan_settle.activation_id,
        resource=cfg.controller_resource_id,
        world_id=cfg.authority_world_id,
        phase="settle",
    )
    capture_release = ActivationSpec(
        activation_id="capture_release",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.capture_synchronized_checkpoint",
        inputs=(
            _port("open_receipt", EXECUTION_RECEIPT),
            _port("settle_receipt", EXECUTION_RECEIPT),
        ),
        outputs=(_port("observation", OBSERVATION),),
        bindings=(
            _artifact("open_receipt", execute_open.activation_id, "receipt"),
            _artifact("settle_receipt", execute_settle.activation_id, "receipt"),
        ),
        required_capabilities=("perception.observe",),
        verifier_tags=("post-open-release-observation",),
        params={"phase": "post_release", "source": "observation_registry"},
    )
    verify_release_attachment = ActivationSpec(
        activation_id="verify_release_attachment",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.verify_attachment_evidence",
        inputs=(_port("observation", OBSERVATION),),
        outputs=(_port("attachment_evidence", ATTACHMENT_EVIDENCE),),
        bindings=(_artifact("observation", capture_release.activation_id, "observation"),),
        verifier_tags=("release-observed", "fresh-observation"),
        params={"required_status": "not_held", "phase": "post_release"},
    )
    propose_release_attachment = ActivationSpec(
        activation_id="propose_release_attachment",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.propose_attachment_transition",
        inputs=(
            _port("observation", OBSERVATION),
            _port("attachment_evidence", ATTACHMENT_EVIDENCE),
        ),
        outputs=(_port("proposal", STATE_PROPOSAL),),
        bindings=(
            _artifact("observation", capture_release.activation_id, "observation"),
            _artifact(
                "attachment_evidence",
                verify_release_attachment.activation_id,
                "attachment_evidence",
            ),
        ),
        required_capabilities=("state.propose",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={"target_status": "not_held"},
    )
    commit_release_attachment = _state_reducer(
        "commit_release_attachment", producer=propose_release_attachment.activation_id
    )
    checkpoint_post_release = ActivationSpec(
        activation_id="checkpoint_post_release",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.post_release_checkpoint",
        inputs=(
            _port("observation", OBSERVATION),
            _port("state_receipt", STATE_RECEIPT),
            _port("open_receipt", EXECUTION_RECEIPT),
            _port("settle_receipt", EXECUTION_RECEIPT),
        ),
        outputs=(_port("checkpoint", CHECKPOINT),),
        bindings=(
            _artifact("observation", capture_release.activation_id, "observation"),
            _artifact("state_receipt", commit_release_attachment.activation_id, "receipt"),
            _artifact("open_receipt", execute_open.activation_id, "receipt"),
            _artifact("settle_receipt", execute_settle.activation_id, "receipt"),
        ),
        verifier_tags=("release-observed", "release-state-committed"),
        params={"phase": "post_release"},
    )
    snapshot_retreat = _action_snapshot(
        "snapshot_retreat",
        world_id=cfg.authority_world_id,
        resource=cfg.arm_resource_id,
    )
    plan_retreat = _planner(
        "plan_retreat",
        runner_ref="robomex.bowl_place.plan_safe_retreat",
        action_schema=MOTION_PLAN,
        inputs=(_port("checkpoint", CHECKPOINT),),
        bindings=(_artifact("checkpoint", checkpoint_post_release.activation_id, "checkpoint"),),
        snapshot_producer=snapshot_retreat.activation_id,
        coding_budget=cfg.coding_budget,
        params={
            "plan_kind": "safe_retreat",
            "tcp_frame_id": "panda_hand",
            "planner_backend": "curobo",
        },
    )
    execute_retreat = _system_action(
        "execute_retreat",
        runner_ref="robomex.runtime.execute_motion_plan",
        action_schema=MOTION_PLAN,
        producer=plan_retreat.activation_id,
        resource=cfg.arm_resource_id,
        world_id=cfg.authority_world_id,
        phase="retreat",
    )
    capture_final = ActivationSpec(
        activation_id="capture_final_relation",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.capture_synchronized_checkpoint",
        inputs=(_port("receipt", EXECUTION_RECEIPT),),
        outputs=(_port("observation", OBSERVATION),),
        bindings=(_artifact("receipt", execute_retreat.activation_id, "receipt"),),
        required_capabilities=("perception.observe",),
        verifier_tags=("independent-final-observation",),
        params={"phase": "final_relation", "source": "observation_registry"},
    )
    verify_placement = ActivationSpec(
        activation_id="verify_placement",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.verify_final_support_relation",
        inputs=(_port("observation", OBSERVATION),),
        outputs=(
            _port("relation_evidence", RELATION_EVIDENCE),
            _port("verdict", PLACEMENT_VERDICT),
        ),
        bindings=(_artifact("observation", capture_final.activation_id, "observation"),),
        verifier_tags=("placement-relation", "independent-final-observation"),
    )
    propose_final_relation = ActivationSpec(
        activation_id="propose_final_relation",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="robomex.bowl_place.propose_relation_transition",
        inputs=(
            _port("observation", OBSERVATION),
            _port("relation_evidence", RELATION_EVIDENCE),
            _port("verdict", PLACEMENT_VERDICT),
        ),
        outputs=(_port("proposal", STATE_PROPOSAL),),
        bindings=(
            _artifact("observation", capture_final.activation_id, "observation"),
            _artifact("relation_evidence", verify_placement.activation_id, "relation_evidence"),
            _artifact("verdict", verify_placement.activation_id, "verdict"),
        ),
        required_capabilities=("state.propose",),
        estimated_budget=cfg.coding_budget.execution_budget(),
        params={"predicate": "supported_by", "value": "asserted"},
    )
    commit_final_relation = _state_reducer(
        "commit_final_relation", producer=propose_final_relation.activation_id
    )
    placement_complete = ActivationSpec(
        activation_id="placement_complete",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.complete",
        inputs=(
            _port("verdict", PLACEMENT_VERDICT),
            _port("state_receipt", STATE_RECEIPT),
        ),
        bindings=(
            _artifact("verdict", verify_placement.activation_id, "verdict"),
            _artifact("state_receipt", commit_final_relation.activation_id, "receipt"),
        ),
    )
    recovery_frontier = ActivationSpec(
        activation_id="recovery_frontier",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="robomex.bowl_place.closed_recovery_frontier",
        params={"closed_slot": True, "manager_wake": "exception_only", "default": "safe_stop"},
    )
    held_tracker = ActivationSpec(
        activation_id="held_bowl_tracker",
        runner_kind=RunnerKind.TRACKING_SERVICE,
        runner_ref="robomex.bowl_place.track_held_bowl",
        lane=ActivationLane.SERVICE,
        lifecycle=LifecycleScope.WORKFLOW,
        subscriptions=("action_outcome",),
        required_capabilities=("perception.track",),
        estimated_budget=ExecutionBudget(),
        params={"role": "continuous_monitor_only"},
    )
    plate_tracker = ActivationSpec(
        activation_id="plate_tracker",
        runner_kind=RunnerKind.TRACKING_SERVICE,
        runner_ref="robomex.bowl_place.track_plate",
        lane=ActivationLane.SERVICE,
        lifecycle=LifecycleScope.WORKFLOW,
        subscriptions=("action_outcome",),
        required_capabilities=("perception.track",),
        estimated_budget=ExecutionBudget(),
        params={"role": "continuous_monitor_only"},
    )

    transitions: list[TransitionSpec] = [
        _transition(
            author_attachment_monitor.activation_id,
            ControlOutcome.SUCCESS,
            capture_initial.activation_id,
        ),
        _transition(
            capture_initial.activation_id,
            ControlOutcome.SUCCESS,
            verify_initial_attachment.activation_id,
        ),
        _transition(
            verify_initial_attachment.activation_id,
            ControlOutcome.SUCCESS,
            propose_initial_attachment.activation_id,
        ),
        _transition(
            propose_initial_attachment.activation_id,
            ControlOutcome.SUCCESS,
            commit_initial_attachment.activation_id,
        ),
        _transition(
            commit_initial_attachment.activation_id,
            ControlOutcome.SUCCESS,
            snapshot_transport.activation_id,
        ),
        _transition(
            snapshot_transport.activation_id, ControlOutcome.SUCCESS, plan_transport.activation_id
        ),
        _transition(
            plan_transport.activation_id, ControlOutcome.SUCCESS, execute_transport.activation_id
        ),
        _transition(
            execute_transport.activation_id, ControlOutcome.SUCCESS, capture_alignment.activation_id
        ),
        _transition(
            capture_alignment.activation_id,
            ControlOutcome.SUCCESS,
            verify_alignment_attachment.activation_id,
        ),
        _transition(
            verify_alignment_attachment.activation_id,
            ControlOutcome.SUCCESS,
            estimate_alignment.activation_id,
        ),
        _transition(
            estimate_alignment.activation_id, ControlOutcome.SUCCESS, alignment_gate.activation_id
        ),
        _transition(
            alignment_gate.activation_id,
            ControlOutcome.NEEDS_ADJUSTMENT,
            snapshot_correction.activation_id,
        ),
        _transition(
            snapshot_correction.activation_id, ControlOutcome.SUCCESS, plan_correction.activation_id
        ),
        _transition(
            plan_correction.activation_id, ControlOutcome.SUCCESS, execute_correction.activation_id
        ),
        _transition(
            execute_correction.activation_id,
            ControlOutcome.SUCCESS,
            capture_alignment.activation_id,
        ),
        _transition(
            alignment_gate.activation_id, ControlOutcome.SUCCESS, snapshot_descend.activation_id
        ),
        _transition(
            snapshot_descend.activation_id, ControlOutcome.SUCCESS, plan_descend.activation_id
        ),
        _transition(
            plan_descend.activation_id, ControlOutcome.SUCCESS, execute_descend.activation_id
        ),
        _transition(
            execute_descend.activation_id, ControlOutcome.SUCCESS, capture_pre_release.activation_id
        ),
        _transition(
            capture_pre_release.activation_id,
            ControlOutcome.SUCCESS,
            verify_pre_release_attachment.activation_id,
        ),
        _transition(
            verify_pre_release_attachment.activation_id,
            ControlOutcome.SUCCESS,
            estimate_pre_release_alignment.activation_id,
        ),
        _transition(
            estimate_pre_release_alignment.activation_id,
            ControlOutcome.SUCCESS,
            checkpoint_pre_release.activation_id,
        ),
        _transition(
            checkpoint_pre_release.activation_id,
            ControlOutcome.SUCCESS,
            snapshot_open.activation_id,
        ),
        _transition(snapshot_open.activation_id, ControlOutcome.SUCCESS, plan_open.activation_id),
        _transition(plan_open.activation_id, ControlOutcome.SUCCESS, execute_open.activation_id),
        _transition(
            execute_open.activation_id, ControlOutcome.SUCCESS, snapshot_settle.activation_id
        ),
        _transition(
            snapshot_settle.activation_id, ControlOutcome.SUCCESS, plan_settle.activation_id
        ),
        _transition(
            plan_settle.activation_id, ControlOutcome.SUCCESS, execute_settle.activation_id
        ),
        _transition(
            execute_settle.activation_id, ControlOutcome.SUCCESS, capture_release.activation_id
        ),
        _transition(
            capture_release.activation_id,
            ControlOutcome.SUCCESS,
            verify_release_attachment.activation_id,
        ),
        _transition(
            verify_release_attachment.activation_id,
            ControlOutcome.SUCCESS,
            propose_release_attachment.activation_id,
        ),
        _transition(
            propose_release_attachment.activation_id,
            ControlOutcome.SUCCESS,
            commit_release_attachment.activation_id,
        ),
        _transition(
            commit_release_attachment.activation_id,
            ControlOutcome.SUCCESS,
            checkpoint_post_release.activation_id,
        ),
        _transition(
            checkpoint_post_release.activation_id,
            ControlOutcome.SUCCESS,
            snapshot_retreat.activation_id,
        ),
        _transition(
            snapshot_retreat.activation_id, ControlOutcome.SUCCESS, plan_retreat.activation_id
        ),
        _transition(
            plan_retreat.activation_id, ControlOutcome.SUCCESS, execute_retreat.activation_id
        ),
        _transition(
            execute_retreat.activation_id, ControlOutcome.SUCCESS, capture_final.activation_id
        ),
        _transition(
            capture_final.activation_id, ControlOutcome.SUCCESS, verify_placement.activation_id
        ),
        _transition(
            verify_placement.activation_id,
            ControlOutcome.SUCCESS,
            propose_final_relation.activation_id,
        ),
        _transition(
            propose_final_relation.activation_id,
            ControlOutcome.SUCCESS,
            commit_final_relation.activation_id,
        ),
        _transition(
            commit_final_relation.activation_id,
            ControlOutcome.SUCCESS,
            placement_complete.activation_id,
        ),
    ]

    for source in (
        author_attachment_monitor.activation_id,
        capture_initial.activation_id,
        verify_initial_attachment.activation_id,
        propose_initial_attachment.activation_id,
        commit_initial_attachment.activation_id,
        capture_alignment.activation_id,
        verify_alignment_attachment.activation_id,
        estimate_alignment.activation_id,
        alignment_gate.activation_id,
        capture_pre_release.activation_id,
        verify_pre_release_attachment.activation_id,
        estimate_pre_release_alignment.activation_id,
        checkpoint_pre_release.activation_id,
        capture_release.activation_id,
        verify_release_attachment.activation_id,
        propose_release_attachment.activation_id,
        commit_release_attachment.activation_id,
        checkpoint_post_release.activation_id,
        capture_final.activation_id,
        verify_placement.activation_id,
        propose_final_relation.activation_id,
        commit_final_relation.activation_id,
    ):
        transitions.extend(
            _recovery_edges(
                source,
                (
                    ControlOutcome.FAILED,
                    ControlOutcome.STALE_INPUT,
                    ControlOutcome.STALE_OBSERVATION,
                    ControlOutcome.UNCERTAIN,
                    ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                    ControlOutcome.WRONG_GROUNDING,
                    ControlOutcome.TARGET_DRIFT,
                    ControlOutcome.EXHAUSTED,
                    ControlOutcome.FAILED_PLACEMENT,
                ),
            )
        )
    for source in (
        plan_transport.activation_id,
        plan_correction.activation_id,
        plan_descend.activation_id,
        plan_retreat.activation_id,
        plan_open.activation_id,
        plan_settle.activation_id,
    ):
        transitions.extend(
            _recovery_edges(
                source,
                (ControlOutcome.INFEASIBLE, ControlOutcome.STALE_INPUT, ControlOutcome.FAILED),
            )
        )
    for source in (
        snapshot_transport.activation_id,
        snapshot_correction.activation_id,
        snapshot_descend.activation_id,
        snapshot_open.activation_id,
        snapshot_settle.activation_id,
        snapshot_retreat.activation_id,
    ):
        transitions.extend(
            _recovery_edges(
                source,
                (ControlOutcome.INFEASIBLE, ControlOutcome.STALE_INPUT),
            )
        )
    for source in (
        execute_transport.activation_id,
        execute_correction.activation_id,
        execute_descend.activation_id,
        execute_open.activation_id,
        execute_settle.activation_id,
        execute_retreat.activation_id,
    ):
        transitions.extend(
            _recovery_edges(
                source,
                (
                    ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                    ControlOutcome.INTERRUPTED,
                    ControlOutcome.EXECUTION_FAULT,
                    ControlOutcome.STALE_INPUT,
                    ControlOutcome.FAILED,
                ),
            )
        )

    activations = (
        author_attachment_monitor,
        capture_initial,
        verify_initial_attachment,
        propose_initial_attachment,
        commit_initial_attachment,
        snapshot_transport,
        plan_transport,
        execute_transport,
        capture_alignment,
        verify_alignment_attachment,
        estimate_alignment,
        alignment_gate,
        snapshot_correction,
        plan_correction,
        execute_correction,
        snapshot_descend,
        plan_descend,
        execute_descend,
        capture_pre_release,
        verify_pre_release_attachment,
        estimate_pre_release_alignment,
        checkpoint_pre_release,
        snapshot_open,
        plan_open,
        execute_open,
        snapshot_settle,
        plan_settle,
        execute_settle,
        capture_release,
        verify_release_attachment,
        propose_release_attachment,
        commit_release_attachment,
        checkpoint_post_release,
        snapshot_retreat,
        plan_retreat,
        execute_retreat,
        capture_final,
        verify_placement,
        propose_final_relation,
        commit_final_relation,
        placement_complete,
        recovery_frontier,
        held_tracker,
        plate_tracker,
    )
    return ElasticGraphSpec(
        graph_id=cfg.graph_id,
        revision=cfg.revision,
        entry_activation=author_attachment_monitor.activation_id,
        terminal_activations=(placement_complete.activation_id, recovery_frontier.activation_id),
        activations=activations,
        transitions=tuple(transitions),
        bounded_loops=(
            BoundedLoopSpec(
                loop_id="alignment_visual_servo",
                activation_ids=(
                    capture_alignment.activation_id,
                    verify_alignment_attachment.activation_id,
                    estimate_alignment.activation_id,
                    alignment_gate.activation_id,
                    snapshot_correction.activation_id,
                    plan_correction.activation_id,
                    execute_correction.activation_id,
                ),
                entry_activation=capture_alignment.activation_id,
                max_iterations=cfg.max_alignment_iterations,
                progress_schema_id=ALIGNMENT_ERROR,
            ),
        ),
        metadata={
            "protocol": "bowl_on_plate_visual_servo",
            "profile": "fresh_evidence_reducer_v2",
            "normal_manager_wake": False,
            "checkpoint_source": "ObservationRegistry",
            "forbid_open_time_attachment_external_binding": True,
            "online_monitor": {
                "author_activation": author_attachment_monitor.activation_id,
                "artifact_schema": MONITOR_PROGRAM,
                "sealed_action_consumers": (
                    execute_transport.activation_id,
                    execute_correction.activation_id,
                    execute_descend.activation_id,
                ),
            },
            "state_authority": {
                "reducers": (
                    commit_initial_attachment.activation_id,
                    commit_release_attachment.activation_id,
                    commit_final_relation.activation_id,
                ),
                "proposal_schema": STATE_PROPOSAL,
                "receipt_schema": STATE_RECEIPT,
            },
            "recovery_frontier": "bowl_place_recovery",
            "closed_servo_outcomes": {
                "within_tolerance": ControlOutcome.SUCCESS.value,
                "correction_required": ControlOutcome.NEEDS_ADJUSTMENT.value,
                "loop_exhausted": ControlOutcome.EXHAUSTED.value,
                "target_drift": ControlOutcome.TARGET_DRIFT.value,
                "occluded": ControlOutcome.UNCERTAIN.value,
                "dropped": ControlOutcome.ATTACHMENT_NOT_CONFIRMED.value,
                "identity_swap": ControlOutcome.WRONG_GROUNDING.value,
                "stale_observation": ControlOutcome.STALE_OBSERVATION.value,
            },
        },
    )


def build_fixed_bowl_place_protocol(
    config: BowlPlaceProtocolConfig | None = None,
    *,
    compiler: ElasticGraphCompiler | None = None,
) -> FixedBowlPlaceProtocol:
    """Compile the fixed graph and bind its authorized recovery slot."""

    cfg = config or BowlPlaceProtocolConfig()
    spec = build_bowl_place_graph(cfg)
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
            actor_spawns=4,
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
    return FixedBowlPlaceProtocol(
        spec=spec,
        compiled=compiled,
        recovery_slot=slot,
        recovery_frontier=frontier,
        attachment_monitor_program=attachment_program,
    )


__all__ = [
    "ALIGNMENT_ERROR",
    "ATTACHMENT_GUARD",
    "ATTACHMENT_STATE",
    "BowlPlaceProtocolConfig",
    "CodingPhaseBudgetConfig",
    "CHECKPOINT",
    "EXECUTION_RECEIPT",
    "FixedBowlPlaceProtocol",
    "GRIPPER_COMMAND",
    "HELD_BOWL",
    "MOTION_PLAN",
    "MONITOR_PROGRAM",
    "OBSERVATION",
    "PLACEMENT_VERDICT",
    "PLATE_TARGET",
    "RECOVERY_SAFETY_DECISION",
    "SERVO_DECISION",
    "WAIT_SPEC",
    "build_bowl_place_graph",
    "build_bowl_attachment_monitor_program",
    "build_fixed_bowl_place_protocol",
]
