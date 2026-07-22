from __future__ import annotations

import pytest

from robomex.elastic import (
    ActivationLane,
    ActivationSpec,
    ArtifactBinding,
    BarrierProfile,
    ComposableFrontier,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    ExternalBinding,
    FragmentExitRoute,
    FragmentInputRoute,
    FragmentOutputRoute,
    GraphFragment,
    GraphPatchCoordinator,
    LifecycleScope,
    PatchOperation,
    PatchRejectCode,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
    VerifierObligation,
    derive_closed_slot,
)
from robomex.runtime.activation import ActivationScheduler, ActivationStatus
from robomex.runtime.events import (
    ControlOutcome,
    NodeOutcomeEvent,
    RosterOperation,
    RosterUpdate,
    ServiceOutcome,
    ServiceStatus,
)

POSE = "test.pose.v1"
PLAN = "test.plan.v1"


def _base_graph():
    prepare = ActivationSpec(
        activation_id="prepare",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="test.prepare",
        outputs=(PortSpecV2(name="pose", schema_id=POSE),),
    )
    slot = ActivationSpec(
        activation_id="closed",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="test.closed",
        inputs=(PortSpecV2(name="pose", schema_id=POSE),),
        outputs=(PortSpecV2(name="plan", schema_id=PLAN),),
        bindings=(
            ArtifactBinding(
                input_port="pose", source_activation="prepare", source_port="pose"
            ),
        ),
    )
    consume = ActivationSpec(
        activation_id="consume",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="test.consume",
        inputs=(PortSpecV2(name="plan", schema_id=PLAN),),
        bindings=(
            ArtifactBinding(
                input_port="plan", source_activation="closed", source_port="plan"
            ),
        ),
    )
    done = ActivationSpec(
        activation_id="done",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="test.done",
    )
    tracker = ActivationSpec(
        activation_id="tracker",
        runner_kind=RunnerKind.TRACKING_SERVICE,
        runner_ref="test.tracker",
        lane=ActivationLane.SERVICE,
        lifecycle=LifecycleScope.WORKFLOW,
        subscriptions=("observation",),
    )
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="patchable",
            entry_activation="prepare",
            terminal_activations=("done",),
            activations=(prepare, slot, consume, done, tracker),
            transitions=(
                TransitionSpec(
                    source="prepare", outcome=ControlOutcome.SUCCESS, target="closed"
                ),
                TransitionSpec(
                    source="closed", outcome=ControlOutcome.SUCCESS, target="consume"
                ),
                TransitionSpec(
                    source="consume", outcome=ControlOutcome.SUCCESS, target="done"
                ),
            ),
        )
    )


def _event(command, event_id: str) -> NodeOutcomeEvent:
    return NodeOutcomeEvent(
        event_id=event_id,
        episode_id="episode",
        workflow_id="workflow",
        source=f"test:{command.activation_id}",
        activation_id=command.activation_id,
        node_id=command.activation_id,
        command_id=command.command_id,
        attempt=command.attempt,
        outcome=ControlOutcome.SUCCESS,
        graph_revision=command.graph_revision,
    )


def _scheduler(*, advance_to_slot: bool = True) -> ActivationScheduler:
    scheduler = ActivationScheduler(
        episode_id="episode", workflow_id="workflow", graph=_base_graph()
    )
    scheduler.start()
    tracker = scheduler.next_commands()[0]
    scheduler.on_event(
        ServiceOutcome(
            episode_id="episode",
            workflow_id="workflow",
            source="test:tracker",
            activation_id=tracker.activation_id,
            command_id=tracker.command_id,
            attempt=tracker.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=tracker.graph_revision,
        )
    )
    prepare = scheduler.next_commands()[0]
    if advance_to_slot:
        scheduler.on_event(_event(prepare, "prepare-complete"))
    return scheduler


def _slot(
    scheduler: ActivationScheduler,
    *,
    targets: tuple[str, ...] = ("closed",),
    effect_ceiling: EffectScope = EffectScope.AUTHORITATIVE_WORLD,
    capabilities: tuple[str, ...] = ("motion.plan",),
    budget: ExecutionBudget | None = None,
    verifier: bool = True,
    barrier: BarrierProfile = BarrierProfile.AFFECTED_SCOPE,
):
    obligations = (
        VerifierObligation(
            obligation_id="safety-gate",
            verifier_tag="safety-gate",
            required_output_schema_id=POSE,
        ),
    ) if verifier else ()
    return derive_closed_slot(
        scheduler.graph,
        slot_id="future-motion",
        target_activation_ids=targets,
        effect_ceiling=effect_ceiling,
        capability_ceiling=capabilities,
        budget_ceiling=budget
        or ExecutionBudget(tokens=100, authoritative_actions=1),
        verifier_obligations=obligations,
        barrier_profile=barrier,
    )


def _coordinator(scheduler: ActivationScheduler, slot) -> GraphPatchCoordinator:
    return GraphPatchCoordinator(
        scheduler=scheduler,
        frontier=ComposableFrontier(
            graph_id=scheduler.graph.spec.graph_id,
            revision=scheduler.graph.spec.revision,
            graph_digest=scheduler.graph.digest,
            slots=(slot,),
        ),
    )


def _motion_fragment(
    slot,
    *,
    capability: str = "motion.plan",
    tokens: int = 20,
    verifier_tag: str | None = "safety-gate",
    extra_unreachable: bool = False,
) -> GraphFragment:
    input_cut = slot.input_cut[0]
    output_cut = slot.output_cut[0]
    exit_cut = slot.control_exit_cut[0]
    verify = ActivationSpec(
        activation_id="verify_pose",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="test.verify_pose",
        inputs=(PortSpecV2(name="pose", schema_id=POSE),),
        outputs=(PortSpecV2(name="verified", schema_id=POSE),),
        bindings=(
            ExternalBinding(
                input_port="pose", ref="slot://future-motion/pose", schema_id=POSE
            ),
        ),
        verifier_tags=((verifier_tag,) if verifier_tag else ()),
        estimated_budget=ExecutionBudget(tokens=5),
    )
    act = ActivationSpec(
        activation_id="execute_motion",
        runner_kind=RunnerKind.SYSTEM_ACTION,
        runner_ref="test.execute_motion",
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        authority_world_id="authoritative",
        authoritative_resource="robot.arm",
        inputs=(PortSpecV2(name="verified", schema_id=POSE),),
        outputs=(PortSpecV2(name="plan", schema_id=PLAN),),
        bindings=(
            ArtifactBinding(
                input_port="verified",
                source_activation="verify_pose",
                source_port="verified",
            ),
        ),
        required_capabilities=(capability,),
        estimated_budget=ExecutionBudget(tokens=tokens, authoritative_actions=1),
    )
    activations = [verify, act]
    if extra_unreachable:
        activations.append(
            ActivationSpec(
                activation_id="orphan",
                runner_kind=RunnerKind.CODING_WORKER,
                runner_ref="test.orphan",
            )
        )
    return GraphFragment(
        entry_activation="verify_pose",
        activations=tuple(activations),
        transitions=(
            TransitionSpec(
                source="verify_pose",
                outcome=ControlOutcome.SUCCESS,
                target="execute_motion",
            ),
        ),
        input_routes=(
            FragmentInputRoute(
                cut_id=input_cut.cut_id,
                target_activation="verify_pose",
                target_port="pose",
            ),
        ),
        output_routes=(
            FragmentOutputRoute(
                cut_id=output_cut.cut_id,
                source_activation="execute_motion",
                source_port="plan",
            ),
        ),
        exit_routes=(
            FragmentExitRoute(
                cut_id=exit_cut.cut_id,
                source_activation="execute_motion",
                outcome=ControlOutcome.SUCCESS,
            ),
        ),
    )


def _status(scheduler: ActivationScheduler, activation_id: str):
    return next(
        item.status
        for item in scheduler.snapshot().activations
        if item.activation_id == activation_id
    )


def test_fill_slot_compiles_then_commits_and_preserves_active_service_history() -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler)
    coordinator = _coordinator(scheduler, slot)
    fragment = _motion_fragment(slot)
    before_graph = scheduler.graph
    tracker_before = next(
        item
        for item in scheduler.snapshot().activations
        if item.activation_id == "tracker"
    )

    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id, base_revision=1, fragment=fragment, patch_id="patch-1"
    )

    assert receipt.accepted
    assert receipt.before_revision == 1
    assert receipt.after_revision == 2
    assert receipt.before_digest == before_graph.digest
    assert receipt.after_digest != receipt.before_digest
    assert scheduler.graph.spec.revision == 2
    assert _status(scheduler, "verify_pose") == ActivationStatus.READY
    tracker_after = next(
        item
        for item in scheduler.snapshot().activations
        if item.activation_id == "tracker"
    )
    assert tracker_after == tracker_before
    assert coordinator.frontier.slots == ()
    assert coordinator.versions[0].spec is before_graph.spec
    assert coordinator.versions[0].digest == before_graph.digest
    assert coordinator.versions[1].accepted_receipt == receipt
    assert scheduler.graph.next_activation("prepare", ControlOutcome.SUCCESS) == "verify_pose"
    assert scheduler.graph.next_activation("execute_motion", ControlOutcome.SUCCESS) == "consume"


def test_optimistic_concurrency_rejects_stale_base_without_mutation() -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler)
    coordinator = _coordinator(scheduler, slot)
    before = scheduler.snapshot()

    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=2,
        fragment=_motion_fragment(slot),
    )

    assert not receipt.accepted
    assert receipt.reason_codes == (PatchRejectCode.STALE_BASE_REVISION,)
    assert receipt.before_digest == receipt.after_digest
    assert scheduler.snapshot() == before
    assert len(coordinator.versions) == 1


@pytest.mark.parametrize(
    ("slot_kwargs", "fragment_kwargs", "code"),
    [
        (
            {"effect_ceiling": EffectScope.READ_ONLY},
            {},
            PatchRejectCode.EFFECT_CEILING_EXCEEDED,
        ),
        (
            {"capabilities": ("perception.read",)},
            {},
            PatchRejectCode.CAPABILITY_CEILING_EXCEEDED,
        ),
        (
            {"budget": ExecutionBudget(tokens=10, authoritative_actions=1)},
            {"tokens": 20},
            PatchRejectCode.BUDGET_CEILING_EXCEEDED,
        ),
        (
            {},
            {"verifier_tag": None},
            PatchRejectCode.MISSING_VERIFIER,
        ),
    ],
)
def test_slot_ceilings_and_verifier_obligations_fail_closed(
    slot_kwargs, fragment_kwargs, code
) -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler, **slot_kwargs)
    coordinator = _coordinator(scheduler, slot)
    before = scheduler.snapshot()

    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=_motion_fragment(slot, **fragment_kwargs),
    )

    assert not receipt.accepted
    assert receipt.reason_codes == (code,)
    assert scheduler.snapshot() == before
    assert coordinator.frontier.revision == 1


def test_admitted_target_is_immutable() -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler)
    coordinator = _coordinator(scheduler, slot)
    assert scheduler.next_commands()[0].activation_id == "closed"

    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=_motion_fragment(slot),
    )

    assert not receipt.accepted
    assert receipt.reason_codes == (PatchRejectCode.TARGET_IMMUTABLE,)
    assert scheduler.graph.spec.revision == 1


def test_global_quiescence_blocks_running_service_but_affected_scope_allows_it() -> None:
    global_scheduler = _scheduler(advance_to_slot=False)
    global_slot = _slot(
        global_scheduler, barrier=BarrierProfile.GLOBAL_QUIESCENCE
    )
    global_coordinator = _coordinator(global_scheduler, global_slot)

    rejected = global_coordinator.fill_slot(
        slot_id=global_slot.slot_id,
        base_revision=1,
        fragment=_motion_fragment(global_slot),
    )

    assert rejected.reason_codes == (PatchRejectCode.BARRIER_NOT_SATISFIED,)

    affected_scheduler = _scheduler()
    affected_slot = _slot(affected_scheduler)
    affected_coordinator = _coordinator(affected_scheduler, affected_slot)
    accepted = affected_coordinator.fill_slot(
        slot_id=affected_slot.slot_id,
        base_revision=1,
        fragment=_motion_fragment(affected_slot),
    )
    assert accepted.accepted
    assert _status(affected_scheduler, "tracker") == ActivationStatus.RUNNING


def test_compile_failure_rolls_back_scheduler_frontier_and_version_history() -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler)
    coordinator = _coordinator(scheduler, slot)
    before = scheduler.snapshot()

    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=_motion_fragment(slot, extra_unreachable=True),
    )

    assert not receipt.accepted
    assert receipt.reason_codes == (PatchRejectCode.COMPILE_FAILED,)
    assert receipt.before_digest == receipt.after_digest == before.graph_digest
    assert scheduler.snapshot() == before
    assert coordinator.frontier.slots == (slot,)
    assert len(coordinator.versions) == 1


def test_replace_unexecuted_fragment_rewires_multi_node_cut() -> None:
    scheduler = _scheduler()
    slot = _slot(
        scheduler,
        targets=("closed", "consume"),
        capabilities=(),
        budget=ExecutionBudget(tokens=10),
        verifier=False,
    )
    coordinator = _coordinator(scheduler, slot)
    input_cut = slot.input_cut[0]
    exit_cut = slot.control_exit_cut[0]
    replacement = ActivationSpec(
        activation_id="direct_gate",
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
        runner_ref="test.direct_gate",
        inputs=(PortSpecV2(name="pose", schema_id=POSE),),
        bindings=(
            ExternalBinding(input_port="pose", ref="slot://pose", schema_id=POSE),
        ),
        estimated_budget=ExecutionBudget(tokens=1),
    )
    fragment = GraphFragment(
        entry_activation="direct_gate",
        activations=(replacement,),
        input_routes=(
            FragmentInputRoute(
                cut_id=input_cut.cut_id,
                target_activation="direct_gate",
                target_port="pose",
            ),
        ),
        exit_routes=(
            FragmentExitRoute(
                cut_id=exit_cut.cut_id,
                source_activation="direct_gate",
                outcome=ControlOutcome.SUCCESS,
            ),
        ),
    )

    receipt = coordinator.replace_unexecuted_fragment(
        slot_id=slot.slot_id, base_revision=1, fragment=fragment
    )

    assert receipt.accepted
    assert {node.activation_id for node in scheduler.graph.spec.activations}.isdisjoint(
        {"closed", "consume"}
    )
    assert scheduler.graph.next_activation("direct_gate", ControlOutcome.SUCCESS) == "done"
    assert _status(scheduler, "direct_gate") == ActivationStatus.READY


def test_roster_update_is_audited_without_graph_revision_or_digest_change() -> None:
    scheduler = _scheduler()
    slot = _slot(scheduler)
    coordinator = _coordinator(scheduler, slot)
    before = coordinator.frontier
    update = RosterUpdate(
        episode_id="episode",
        workflow_id="workflow",
        source="arena",
        operation=RosterOperation.SPAWN,
        actor_id="candidate-2",
        slot_id=slot.slot_id,
    )

    after = coordinator.record_roster_update(update)

    assert after.revision == before.revision
    assert after.graph_digest == before.graph_digest
    assert coordinator.roster_updates == (update,)
    assert coordinator.receipts == ()
    assert len(coordinator.versions) == 1


def test_fill_slot_rejects_multi_node_closed_fragment() -> None:
    scheduler = _scheduler()
    slot = _slot(
        scheduler,
        targets=("closed", "consume"),
        verifier=False,
        capabilities=(),
    )
    coordinator = _coordinator(scheduler, slot)
    # The operation check happens before fragment shape is evaluated.
    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=GraphFragment(
            entry_activation="noop",
            activations=(
                ActivationSpec(
                    activation_id="noop",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.noop",
                ),
            ),
        ),
    )
    assert receipt.reason_codes == (PatchRejectCode.OPERATION_NOT_ALLOWED,)
    assert PatchOperation.REPLACE_UNEXECUTED_FRAGMENT in slot.allowed_operations
