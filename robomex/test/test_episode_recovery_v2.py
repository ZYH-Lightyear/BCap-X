from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from robomex.elastic import (
    ActivationLane,
    ActivationSpec,
    ArtifactBinding,
    ComposableFrontier,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    FragmentExitRoute,
    GraphFragment,
    GraphPatchCoordinator,
    GraphPatchProposal,
    LifecycleScope,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
    derive_closed_slot,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
    EpisodeRuntimeError,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.action_protocol import FeasibilityStatus
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ControlOutcome, NodeOutcomeEvent, ServiceOutcome
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion, _snapshot


def _actors(root: Path, handler=None) -> tuple[ActorRegistry, InMemoryAgentProvider]:
    provider = InMemoryAgentProvider(handler)
    return (
        ActorRegistry(
            {"in_memory": provider},
            namespace_root="recovery",
            workspace_root=root / "actors",
        ),
        provider,
    )


def _runtime(
    root: Path,
    *,
    episode_id: str,
    handler=None,
    action_backends=None,
    feasibility_checkers=None,
) -> tuple[EpisodeRuntime, InMemoryAgentProvider]:
    actors, provider = _actors(root, handler)
    return (
        EpisodeRuntime(
            episode_id=episode_id,
            episode_root=root / "episode",
            actors=actors,
            action_backends=action_backends,
            feasibility_checkers=feasibility_checkers,
        ),
        provider,
    )


def _register(runtime: EpisodeRuntime, graph) -> None:
    for node in graph.spec.activations:
        if node.runner_kind is RunnerKind.SYSTEM_ACTION:
            continue
        runtime.register_actor_profile(
            node.runner_ref,
            ActorProfile(
                profile_id=f"profile-{node.activation_id}",
                runner_kind=node.runner_kind.value,
                lifecycle=(
                    ActorLifecycle.EPHEMERAL
                    if node.lifecycle is LifecycleScope.INVOCATION
                    else ActorLifecycle.SERVICE
                ),
            ),
        )


def _intent(intent_id: str = "recover") -> SubgoalIntent:
    return SubgoalIntent(
        intent_id=intent_id,
        instruction="complete the durable workflow",
        success_rubric="the terminal activation succeeds",
    )


def _service_primary_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="service-primary-recovery",
            entry_activation="work",
            terminal_activations=("work",),
            activations=(
                ActivationSpec(
                    activation_id="work",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.work",
                ),
                ActivationSpec(
                    activation_id="tracker",
                    runner_kind=RunnerKind.TRACKING_SERVICE,
                    runner_ref="test.tracker",
                    lane=ActivationLane.SERVICE,
                    lifecycle=LifecycleScope.WORKFLOW,
                    subscriptions=("camera",),
                ),
            ),
        )
    )


def test_episode_restart_rehydrates_service_then_primary_without_new_attempt(
    tmp_path: Path,
) -> None:
    graph = _service_primary_graph()

    def handler(_profile, invocation, _isolation):
        if invocation.metadata["activation_id"] == "tracker":
            return None
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    first, _ = _runtime(tmp_path, episode_id="episode", handler=handler)
    _register(first, graph)
    workflow_id = first.open_workflow(
        workflow_id="workflow", intent=_intent(), graph=graph
    )
    first.execute_ready(workflow_id)  # starts and acknowledges the service
    original_service = next(
        event
        for event in first.event_bus.history
        if isinstance(event, ServiceOutcome)
    )
    original_primary = first.next_invocations(workflow_id)[0].command
    assert original_primary.activation_id == "work"

    restarted, provider = _runtime(tmp_path, episode_id="episode", handler=handler)
    _register(restarted, graph)
    restarted.recover_workflow(workflow_id)

    restarted.execute_ready(workflow_id)  # recovery service, same identity
    recovered_service_call = next(
        call for call in provider.calls if call[0] == "invoke" and "tracker" in call[1]
    )
    assert recovered_service_call
    recovered_primary = restarted.next_invocations(workflow_id)[0].command
    assert recovered_primary.command_id == original_primary.command_id
    assert recovered_primary.attempt == original_primary.attempt == 1
    recovered_service_events = [
        event
        for event in restarted.event_bus.history
        if isinstance(event, ServiceOutcome)
        and event.activation_id == "tracker"
    ]
    assert {event.command_id for event in recovered_service_events} == {
        original_service.command_id
    }
    assert {event.attempt for event in recovered_service_events} == {1}

    terminal = restarted.accept_node_outcome(
        workflow_id,
        NodeOutcomeEvent(
            episode_id="episode",
            workflow_id=workflow_id,
            source="external-recovered-runner",
            activation_id="work",
            node_id="work",
            command_id=recovered_primary.command_id,
            attempt=recovered_primary.attempt,
            outcome=ControlOutcome.SUCCESS,
            graph_revision=recovered_primary.graph_revision,
        ),
    )
    assert terminal.status.value == "succeeded"
    assert restarted.next_invocations(workflow_id) == ()
    work = next(
        item
        for item in restarted.scheduler_store.latest(
            episode_id="episode", workflow_id=workflow_id
        ).state.activations
        if item.activation_id == "work"
    )
    assert work.attempts == 1


def _patchable_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="episode-patch-recovery",
            entry_activation="prepare",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="prepare",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.prepare",
                ),
                ActivationSpec(
                    activation_id="closed",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.closed",
                ),
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
            transitions=(
                TransitionSpec(
                    source="prepare",
                    outcome=ControlOutcome.SUCCESS,
                    target="closed",
                ),
                TransitionSpec(
                    source="closed",
                    outcome=ControlOutcome.SUCCESS,
                    target="done",
                ),
            ),
        )
    )


def test_episode_restart_recovers_accepted_patch_successor_and_frontier(
    tmp_path: Path,
) -> None:
    graph = _patchable_graph()
    slot = derive_closed_slot(
        graph,
        slot_id="replace-closed",
        target_activation_ids=("closed",),
    )
    frontier = ComposableFrontier(
        graph_id=graph.spec.graph_id,
        revision=graph.spec.revision,
        graph_digest=graph.digest,
        slots=(slot,),
    )
    first, _ = _runtime(
        tmp_path,
        episode_id="patch-episode",
        handler=lambda *_: ActivationExecutionResult(outcome=ControlOutcome.SUCCESS),
    )
    _register(first, graph)
    first.register_actor_profile(
        "test.replacement",
        ActorProfile(profile_id="replacement", runner_kind="coding_worker"),
    )
    workflow_id = first.open_workflow(
        workflow_id="workflow",
        intent=_intent("patch"),
        graph=graph,
        frontier=frontier,
    )
    fragment = GraphFragment(
        entry_activation="replacement",
        activations=(
            ActivationSpec(
                activation_id="replacement",
                runner_kind=RunnerKind.CODING_WORKER,
                runner_ref="test.replacement",
            ),
        ),
        exit_routes=(
            FragmentExitRoute(
                cut_id=slot.control_exit_cut[0].cut_id,
                source_activation="replacement",
                outcome=ControlOutcome.SUCCESS,
            ),
        ),
    )
    receipt = first.apply_graph_patch(
        workflow_id,
        proposal=GraphPatchProposal(
            patch_id="patch-1",
            operation="fill_slot",
            slot_id=slot.slot_id,
            base_revision=1,
            fragment=fragment,
        ),
    )
    assert receipt.accepted

    restarted, _ = _runtime(
        tmp_path,
        episode_id="patch-episode",
        handler=lambda *_: ActivationExecutionResult(outcome=ControlOutcome.SUCCESS),
    )
    for node in first._workflow(workflow_id).scheduler.graph.spec.activations:
        if node.runner_kind is not RunnerKind.SYSTEM_ACTION:
            restarted.register_actor_profile(
                node.runner_ref,
                ActorProfile(
                    profile_id=f"recovered-{node.activation_id}",
                    runner_kind=node.runner_kind.value,
                ),
            )
    restarted.recover_workflow(workflow_id)
    recovered = restarted._workflow(workflow_id)

    assert recovered.scheduler.graph.spec.revision == 2
    assert recovered.scheduler.graph.digest == receipt.after_digest
    assert recovered.patch_coordinator is not None
    assert recovered.patch_coordinator.frontier.slots == ()
    assert recovered.patch_coordinator.receipts[-1] == receipt
    assert restarted.run_until_terminal(workflow_id).status.value == "succeeded"


class _Checker:
    def certify(self, spec, snapshot):
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="recovery-checker",
            checks={
                "exact_joint_path_interface": FeasibilityStatus.PASS,
                "controller_admissible": FeasibilityStatus.PASS,
                "resource_binding": FeasibilityStatus.PASS,
                "joint_order": FeasibilityStatus.PASS,
                "collision": FeasibilityStatus.PASS,
                "joint_limits": FeasibilityStatus.PASS,
                "robot_model": FeasibilityStatus.PASS,
            },
        )


def _physical_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="physical-recovery",
            entry_activation="author",
            terminal_activations=("execute",),
            activations=(
                ActivationSpec(
                    activation_id="author",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.author",
                    outputs=(
                        PortSpecV2(
                            name="action_spec", schema_id="robomex.motion_plan.v2"
                        ),
                    ),
                ),
                ActivationSpec(
                    activation_id="execute",
                    runner_kind=RunnerKind.SYSTEM_ACTION,
                    runner_ref="runtime.sealed_action",
                    effect_scope=EffectScope.AUTHORITATIVE_WORLD,
                    authority_world_id="live-world",
                    authoritative_resource="arm",
                    inputs=(
                        PortSpecV2(
                            name="action_spec", schema_id="robomex.motion_plan.v2"
                        ),
                    ),
                    outputs=(
                        PortSpecV2(
                            name="receipt",
                            schema_id="robomex.execution_receipt.v2",
                        ),
                    ),
                    bindings=(
                        ArtifactBinding(
                            input_port="action_spec",
                            source_activation="author",
                            source_port="action_spec",
                        ),
                    ),
                ),
            ),
            transitions=(
                TransitionSpec(
                    source="author",
                    outcome=ControlOutcome.SUCCESS,
                    target="execute",
                ),
            ),
        )
    )


class _SimulatedCrash(BaseException):
    pass


@pytest.mark.parametrize("crash_boundary", ["before_commit", "before_node_outcome"])
def test_episode_restart_uses_wal_receipt_without_replaying_physical_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_boundary: str,
) -> None:
    snapshot = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    plan = _motion(snapshot)
    graph = _physical_graph()
    backend = FakeBackend(snapshot)

    def handler(*_args):
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="action_spec",
                    schema_id=plan.schema_version,
                    payload=plan.model_dump(mode="json"),
                ),
            ),
        )

    first, _ = _runtime(
        tmp_path,
        episode_id="physical-episode",
        handler=handler,
        action_backends={(plan.world_id, plan.resource_id): backend},
        feasibility_checkers={(plan.world_id, plan.resource_id): _Checker()},
    )
    _register(first, graph)
    workflow_id = first.open_workflow(
        workflow_id="workflow", intent=_intent("physical"), graph=graph
    )
    first.execute_ready(workflow_id)  # author action spec

    if crash_boundary == "before_commit":
        def crash_before_commit(*_args, **_kwargs):
            raise _SimulatedCrash()

        monkeypatch.setattr(first, "_commit_action_receipt", crash_before_commit)
    else:
        scheduler = first._workflow(workflow_id).scheduler
        original_on_event = scheduler.on_event

        def crash_before_node_outcome(event):
            if (
                isinstance(event, NodeOutcomeEvent)
                and event.activation_id == "execute"
            ):
                raise _SimulatedCrash()
            return original_on_event(event)

        monkeypatch.setattr(scheduler, "on_event", crash_before_node_outcome)
    with pytest.raises(_SimulatedCrash):
        first.execute_ready(workflow_id)
    assert [primitive for primitive, _ in backend.calls] == ["execute_joint_path"]
    wal_before = first.action_wal.records()
    assert [record.record_kind for record in wal_before] == [
        "action_attempt",
        "primitive_receipt",
        "execution_receipt",
    ]

    restarted, _ = _runtime(
        tmp_path,
        episode_id="physical-episode",
        handler=handler,
        action_backends={(plan.world_id, plan.resource_id): backend},
        feasibility_checkers={(plan.world_id, plan.resource_id): _Checker()},
    )
    _register(restarted, graph)
    restarted.recover_workflow(workflow_id)
    terminal = restarted.execute_ready(workflow_id)

    assert terminal.status.value == "succeeded"
    assert [primitive for primitive, _ in backend.calls] == ["execute_joint_path"]
    assert restarted.action_wal.records() == wal_before
    assert restarted.authority_registry.active() == ()


def test_failed_episode_recovery_releases_restored_authority_lease(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    plan = _motion(snapshot)
    graph = _physical_graph()

    def handler(*_args):
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="action_spec",
                    schema_id=plan.schema_version,
                    payload=plan.model_dump(mode="json"),
                ),
            ),
        )

    first, _ = _runtime(tmp_path, episode_id="failed-recovery", handler=handler)
    _register(first, graph)
    workflow_id = first.open_workflow(
        workflow_id="workflow", intent=_intent("failed-recovery"), graph=graph
    )
    first.execute_ready(workflow_id)
    action = first.next_invocations(workflow_id)[0].command
    assert action.activation_id == "execute"
    assert action.lease is not None

    # Process-local actor profiles are intentionally absent.  Recovery must
    # fail before exposing work and must not retain the scheduler-restored
    # physical capability in this new process.
    restarted, _ = _runtime(tmp_path, episode_id="failed-recovery", handler=handler)
    with pytest.raises(EpisodeRuntimeError, match="unregistered actor profiles"):
        restarted.recover_workflow(workflow_id)

    assert restarted.authority_registry.active() == ()


def _single_node_graph(*, graph_id: str = "single"):
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id=graph_id,
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


def test_workflow_descriptor_rebind_and_corruption_fail_closed(tmp_path: Path) -> None:
    graph = _single_node_graph()
    first, _ = _runtime(tmp_path, episode_id="descriptor-episode")
    _register(first, graph)
    first.open_workflow(workflow_id="workflow", intent=_intent("one"), graph=graph)

    rebound, _ = _runtime(tmp_path, episode_id="descriptor-episode")
    _register(rebound, graph)
    with pytest.raises(EpisodeRuntimeError, match="rebound"):
        rebound.open_workflow(
            workflow_id="workflow", intent=_intent("different"), graph=graph
        )

    descriptor_path = rebound.workflow_descriptors.path_for("workflow")
    descriptor_path.chmod(0o600)
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["intent"]["instruction"] = "tampered instruction"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    corrupted, _ = _runtime(tmp_path, episode_id="descriptor-episode")
    _register(corrupted, graph)
    with pytest.raises(EpisodeRuntimeError, match="invalid workflow descriptor"):
        corrupted.recover_workflow("workflow")


def test_closed_terminal_workflow_recovers_read_only_outcome(tmp_path: Path) -> None:
    graph = _single_node_graph(graph_id="closed-recovery")

    def handler(*_args):
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    first, _ = _runtime(tmp_path, episode_id="closed-episode", handler=handler)
    _register(first, graph)
    workflow_id = first.open_workflow(
        workflow_id="workflow", intent=_intent("closed"), graph=graph
    )
    first.run_until_terminal(workflow_id)
    original_outcome = first.close_workflow(workflow_id)
    commit_count = len(
        first.scheduler_store.history(
            episode_id="closed-episode", workflow_id=workflow_id
        )
    )

    restarted, _ = _runtime(
        tmp_path, episode_id="closed-episode", handler=handler
    )
    _register(restarted, graph)
    restarted.recover_workflow(workflow_id)
    assert restarted.next_invocations(workflow_id) == ()
    recovered_outcome = restarted.close_workflow(workflow_id)

    assert recovered_outcome == original_outcome
    assert len(
        restarted.scheduler_store.history(
            episode_id="closed-episode", workflow_id=workflow_id
        )
    ) == commit_count
