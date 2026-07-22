from __future__ import annotations

import pytest

from robomex.elastic import (
    ActivationSpec,
    ComposableFrontier,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    GraphPatchProposal,
    PatchOperation,
    RunnerKind,
    VerifierObligation,
    derive_closed_slot,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.arena import (
    ArenaCandidateSpec,
    ArenaContext,
    CheckStatus,
    RiskInputs,
    RiskPolicy,
    RiskReport,
)
from robomex.orchestration.episode import EpisodeRuntime, EpisodeRuntimeError
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerDecision,
    ManagerSignal,
    ScriptedManagerInvoker,
    SwarmManagerSession,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    FeasibilityStatus,
    JointPath,
    MotionPlan,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import (
    ArenaDecision,
    FindingSeverity,
    GraphPatchOutcome,
    ManagerInvocationOutcome,
    MonitorFinding,
    MonitorFindingKind,
    RosterUpdate,
)
from robomex.test.test_graph_patch_v2 import (
    POSE,
    _base_graph,
    _motion_fragment,
)


def _actors(tmp_path):
    return ActorRegistry(
        {"in_memory": InMemoryAgentProvider(lambda *args: None)},
        namespace_root="episode",
        workspace_root=tmp_path / "actors",
    )


def _register(runtime: EpisodeRuntime, graph) -> None:
    for node in graph.spec.activations:
        if node.runner_kind in {
            RunnerKind.ACTION_SNAPSHOT,
            RunnerKind.SYSTEM_ACTION,
        }:
            continue
        runtime.register_actor_profile(
            node.runner_ref,
            ActorProfile(
                profile_id=f"profile-{node.activation_id}",
                runner_kind=node.runner_kind.value,
                lifecycle=(
                    "service" if node.lane.value == "service" else "ephemeral"
                ),
                capability_ceiling=frozenset(node.required_capabilities),
                effect_ceiling=(
                    frozenset()
                    if node.effect_scope is EffectScope.READ_ONLY
                    else frozenset({node.effect_scope.value})
                ),
            ),
        )


def _planning_fragment(slot):
    """Use the patch helper as a read-only planner, not a fake physical writer."""

    fragment = _motion_fragment(slot)
    planner = fragment.activations[1]
    planner = ActivationSpec.model_validate(
        {
            **planner.model_dump(mode="python"),
            "runner_kind": RunnerKind.CODING_WORKER,
            "runner_ref": "test.plan_motion",
            "effect_scope": EffectScope.READ_ONLY,
            "authority_world_id": None,
            "authoritative_resource": None,
            "estimated_budget": ExecutionBudget(tokens=20),
        }
    )
    return fragment.model_copy(
        update={"activations": (fragment.activations[0], planner)}
    )


def test_episode_commits_patch_and_persists_typed_receipt(tmp_path) -> None:
    graph = _base_graph()
    slot = derive_closed_slot(
        graph,
        slot_id="future-motion",
        target_activation_ids=("closed",),
        effect_ceiling=EffectScope.AUTHORITATIVE_WORLD,
        capability_ceiling=("motion.plan",),
        budget_ceiling=ExecutionBudget(tokens=100, authoritative_actions=1),
        verifier_obligations=(
            VerifierObligation(
                obligation_id="safety-gate",
                verifier_tag="safety-gate",
                required_output_schema_id=POSE,
            ),
        ),
    )
    frontier = ComposableFrontier(
        graph_id=graph.spec.graph_id,
        revision=graph.spec.revision,
        graph_digest=graph.digest,
        slots=(slot,),
    )
    runtime = EpisodeRuntime(
        episode_id="episode",
        episode_root=tmp_path / "episode",
        actors=_actors(tmp_path),
    )
    _register(runtime, graph)
    runtime.register_actor_profile(
        "test.verify_pose",
        ActorProfile(
            profile_id="profile-verify-pose",
            runner_kind="deterministic_gate",
        ),
    )
    runtime.register_actor_profile(
        "test.plan_motion",
        ActorProfile(
            profile_id="profile-plan-motion",
            runner_kind="coding_worker",
            capability_ceiling=frozenset({"motion.plan"}),
        ),
    )
    runtime.open_workflow(
        workflow_id="workflow",
        intent=SubgoalIntent(
            intent_id="place",
            instruction="place bowl",
            success_rubric="supported",
        ),
        graph=graph,
        frontier=frontier,
    )
    proposal = GraphPatchProposal(
        patch_id="patch-episode",
        operation=PatchOperation.FILL_SLOT,
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=_planning_fragment(slot),
    )

    receipt = runtime.apply_graph_patch("workflow", proposal)

    assert receipt.accepted and receipt.after_revision == 2
    event = next(
        item
        for item in runtime.event_bus.history
        if isinstance(item, GraphPatchOutcome)
    )
    assert event.accepted and event.receipt_ref is not None
    record = next(
        item
        for item in runtime.data_plane.artifacts
        if item.artifact_id == event.receipt_ref
    )
    assert runtime.data_plane.resolve(record.ref).payload["patch_id"] == "patch-episode"


def test_episode_manager_uses_fresh_snapshot_and_durable_session_record(tmp_path) -> None:
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="manager-graph",
            entry_activation="done",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
        )
    )
    runtime = EpisodeRuntime(
        episode_id="episode",
        episode_root=tmp_path / "episode",
        actors=_actors(tmp_path),
    )
    _register(runtime, graph)
    session = SwarmManagerSession(
        session_id="manager",
        episode_id="episode",
        workflow_id="workflow",
        intent_id="place",
    )
    invoker = ScriptedManagerInvoker(
        (
            ManagerDecision(
                action=ManagerAction.NOOP,
                summary="recovery requires no topology change",
                tokens_used=3,
            ),
        )
    )
    runtime.open_workflow(
        workflow_id="workflow",
        intent=SubgoalIntent(
            intent_id="place",
            instruction="place bowl",
            success_rubric="supported",
        ),
        graph=graph,
        manager_session=session,
        manager_invoker=invoker,
    )

    routine = runtime.invoke_manager(
        "workflow", signal=ManagerSignal.FRAME_TICK
    )
    recovery = runtime.invoke_manager(
        "workflow",
        signal=ManagerSignal.RECOVERY,
        triggering_event={"kind": "attachment_anomaly"},
    )

    assert not routine.step.invoked
    assert recovery.step.invoked
    assert recovery.step.session.record_revision == 2
    assert len(runtime.manager_ledger.records) == 2
    events = [
        item
        for item in runtime.event_bus.history
        if isinstance(item, ManagerInvocationOutcome)
    ]
    assert [item.invoked for item in events] == [False, True]
    assert events[-1].manager_record_ref is not None


def test_exceptional_manager_wakeup_survives_crash_without_replay(tmp_path) -> None:
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="manager-recovery-graph",
            entry_activation="done",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
        )
    )
    episode_root = tmp_path / "episode"
    first = EpisodeRuntime(
        episode_id="episode",
        episode_root=episode_root,
        actors=_actors(tmp_path / "first"),
    )
    _register(first, graph)
    first.open_workflow(
        workflow_id="workflow",
        intent=SubgoalIntent(
            intent_id="place",
            instruction="place bowl",
            success_rubric="supported",
        ),
        graph=graph,
        manager_session=SwarmManagerSession(
            session_id="manager-recovery",
            episode_id="episode",
            workflow_id="workflow",
            intent_id="place",
        ),
        manager_invoker=ScriptedManagerInvoker(()),
    )
    finding = MonitorFinding(
        event_id="finding-before-crash",
        finding_id="finding-before-crash",
        episode_id="episode",
        workflow_id="workflow",
        source="sealed_action_runner",
        monitor_id="attachment-monitor",
        finding=MonitorFindingKind.ATTACHMENT_ANOMALY,
        severity=FindingSeverity.CRITICAL,
    )
    # Fault seam: the finding is durable, but the process is lost before the
    # in-memory Manager queue is updated or drained.
    first._workflow("workflow").scheduler.on_event(finding)

    recovered_invoker = ScriptedManagerInvoker(
        (ManagerDecision(action=ManagerAction.NOOP, tokens_used=1),)
    )
    recovered = EpisodeRuntime(
        episode_id="episode",
        episode_root=episode_root,
        actors=_actors(tmp_path / "recovered"),
    )
    _register(recovered, graph)
    recovered.recover_workflow("workflow", manager_invoker=recovered_invoker)
    recovered.execute_ready("workflow")

    assert len(recovered_invoker.requests) == 1
    assert recovered_invoker.requests[0].signal is ManagerSignal.RECOVERY
    assert (
        recovered_invoker.requests[0].snapshot.triggering_event["event_id"]
        == finding.event_id
    )
    latest = recovered.manager_ledger.latest("manager-recovery")
    assert latest is not None
    assert latest.receipts[-1].triggering_event_id == finding.event_id

    final_invoker = ScriptedManagerInvoker(())
    final = EpisodeRuntime(
        episode_id="episode",
        episode_root=episode_root,
        actors=_actors(tmp_path / "final"),
    )
    _register(final, graph)
    final.recover_workflow("workflow", manager_invoker=final_invoker)
    final.execute_ready("workflow")

    assert final_invoker.requests == []


def test_episode_arena_records_roster_and_selection_without_graph_patch(tmp_path) -> None:
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="place-bowl",
            entry_activation="done",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
        )
    )
    snapshot = AdmissionSnapshot(
        world_id="robot-world",
        resource_id="arm",
        robot_revision=1,
        scene_revision=1,
        attachment_revision=1,
        config_revision=1,
        joint_names=("j1", "j2"),
        joint_positions_rad=(0.0, 0.0),
        config_digest="sha256:" + "a" * 64,
        collision_world_digest="sha256:" + "b" * 64,
    )
    robot_model_digest = "sha256:" + "c" * 64

    class Backend:
        descriptor = BackendDescriptor(backend_id="episode-test-backend")

        def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
            assert (world_id, resource_id) == ("robot-world", "arm")
            return snapshot

        def execute_joint_path(self, **_kwargs: object) -> BackendCallResult:
            return BackendCallResult(converged=True)

        def set_gripper(self, **_kwargs: object) -> BackendCallResult:
            return BackendCallResult(converged=True)

        def wait(self, **_kwargs: object) -> BackendCallResult:
            return BackendCallResult(converged=True)

    class Checker:
        def certify(self, spec, current_snapshot):
            return build_feasibility_certificate(
                spec=spec,
                snapshot=current_snapshot,
                checker_id="episode-test-checker",
                checks={
                    "kinematic_feasibility": FeasibilityStatus.PASS,
                    "collision": FeasibilityStatus.PASS,
                    "joint_limits": FeasibilityStatus.PASS,
                },
            )

    candidate_payload: dict[str, object] = {}
    provider = InMemoryAgentProvider(lambda *_args: candidate_payload)
    actors = ActorRegistry(
        {"memory": provider},
        namespace_root="episode-1",
        workspace_root=tmp_path / "actors",
    )
    runtime = EpisodeRuntime(
        episode_id="episode-1",
        episode_root=tmp_path / "episode",
        actors=actors,
        action_backends={("robot-world", "arm"): Backend()},
        feasibility_checkers={("robot-world", "arm"): Checker()},
    )
    _register(runtime, graph)
    runtime.open_workflow(
        workflow_id="workflow-1",
        intent=SubgoalIntent(
            intent_id="place",
            instruction="place bowl",
            success_rubric="supported",
        ),
        graph=graph,
    )
    snapshot_record = runtime.data_plane.publish(
        workflow_id="workflow-1",
        activation_id="arena-input",
        attempt=1,
        port="snapshot",
        schema=snapshot.schema_version,
        payload=snapshot.model_dump(mode="json"),
    )
    plan = MotionPlan(
        plan_id="episode-arena-plan",
        plan_kind="bounded-alignment",
        tcp_frame_id="tool0",
        planner_backend="test-planner",
        robot_model_digest=robot_model_digest,
        expected_snapshot=snapshot,
        max_start_deviation_rad=0.02,
        possibly_affected_revisions=("robot", "scene"),
        motion=JointPath(
            joint_names=snapshot.joint_names,
            positions_rad=((0.01, 0.02),),
        ),
    )
    plan_record = runtime.data_plane.publish(
        workflow_id="workflow-1",
        activation_id="arena-input",
        attempt=1,
        port="motion-plan",
        schema=plan.schema_version,
        payload=plan.model_dump(mode="json"),
        lineage=(snapshot_record.ref,),
    )
    candidate_payload.update(
        {
            "plan_ref": plan_record.ref,
            "snapshot_ref": snapshot_record.ref,
            "frame": "world",
            "expected_effect": "alignment",
            "utility": 0.8,
            "estimated_risk": 0.1,
            "clearance_m": 0.02,
            "path_length": 0.2,
            "terminal_position_m": (0.1, 0.2, 0.3),
        }
    )
    arena = runtime.create_swarm_arena()
    context = ArenaContext(
        arena_run_id="arena-episode",
        episode_id="episode-1",
        workflow_id="workflow-1",
        graph_id="place-bowl",
        graph_revision=1,
        slot_id="motion-proposals",
        snapshot_ref=snapshot_record.ref,
        expected_frame="world",
        world_id="robot-world",
        resource_id="arm",
        robot_model_digest=robot_model_digest,
        config_digest=snapshot.config_digest,
        candidate_budget_id="workflow-1-motion-candidates",
        candidate_budget_limit=1,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.99,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.05,
            held_pose_uncertainty_m=0.001,
        ),
        RiskPolicy(max_candidates=1),
    )
    candidate = ArenaCandidateSpec(
        candidate_id="one",
        strategy="bounded-alignment",
        profile=ActorProfile(
            profile_id="arena-candidate",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
        ),
        objective="propose one exact motion plan",
    )

    outcome = runtime.run_arena(
        "workflow-1",
        arena=arena,
        context=context,
        risk_report=risk,
        candidates=(candidate,),
        candidate_budget_remaining=1,
    )

    assert outcome.result.selected_candidate_id == "one"
    assert len(outcome.hypothesis_refs) == 1
    assert any(isinstance(item, RosterUpdate) for item in runtime.event_bus.history)
    decision = next(
        item for item in runtime.event_bus.history if isinstance(item, ArenaDecision)
    )
    assert decision.selected_candidate_id == "one"
    assert decision.graph_revision == 1
    assert runtime.arena_consumption_ledger.persistent is True
    assert [
        record.record_kind for record in runtime.arena_consumption_ledger.records
    ] == ["reservation", "completion"]

    # The Arena's guard retains the runtime-owned revision callback.  Replacing
    # only Episode's post-run lookup simulates a revision change in the narrow
    # interval after Arena selection but before artifact publication.
    second_arena = runtime.create_swarm_arena()
    second_context = context.model_copy(
        update={
            "arena_run_id": "arena-episode-post-guard",
            "candidate_budget_id": "workflow-1-post-guard-candidates",
        }
    )
    runtime._arena_graph_revision = lambda _context: ("place-bowl", 2)  # type: ignore[method-assign]
    with pytest.raises(EpisodeRuntimeError, match="refusing promotion publish"):
        runtime.run_arena(
            "workflow-1",
            arena=second_arena,
            context=second_context,
            risk_report=risk,
            candidates=(candidate,),
            candidate_budget_remaining=1,
        )
    decisions = [
        item for item in runtime.event_bus.history if isinstance(item, ArenaDecision)
    ]
    assert len(decisions) == 1
