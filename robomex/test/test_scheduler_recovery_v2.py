from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from robomex.elastic import (
    ActivationLane,
    ActivationSpec,
    ComposableFrontier,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    FragmentExitRoute,
    GraphFragment,
    GraphPatchCommit,
    GraphPatchCoordinator,
    LifecycleScope,
    RunnerKind,
    TransitionSpec,
    derive_closed_slot,
)
from robomex.runtime.activation import (
    ActivationScheduler,
    ActivationStatus,
    AuthorityRegistry,
    SchedulerError,
    SchedulerState,
)
from robomex.runtime.event_log import PersistentTypedEventBus
from robomex.runtime.events import (
    ControlOutcome,
    NodeOutcomeEvent,
    ServiceOutcome,
    ServiceStatus,
)
from robomex.runtime.scheduler_store import JsonlSchedulerStateStore


def _concurrent_store_append(
    path: str,
    state_json: str,
    reason: str,
    start,
    results,
) -> None:
    store = JsonlSchedulerStateStore(path)
    state = SchedulerState.model_validate_json(state_json)
    start.wait(timeout=10)
    try:
        results.put(("ok", store.append(state, reason=reason)))
    except Exception as exc:  # pragma: no cover - asserted in parent process
        results.put(("error", type(exc).__name__))


def _simple_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="durable",
            entry_activation="work",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="work",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.work",
                ),
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
            transitions=(
                TransitionSpec(
                    source="work",
                    outcome=ControlOutcome.SUCCESS,
                    target="done",
                ),
            ),
        )
    )


def _outcome(command, *, event_id: str = "outcome") -> NodeOutcomeEvent:
    return NodeOutcomeEvent(
        event_id=event_id,
        episode_id=command.episode_id,
        workflow_id=command.workflow_id,
        source="test.runner",
        activation_id=command.activation_id,
        node_id=command.activation_id,
        command_id=command.command_id,
        attempt=command.attempt,
        outcome=ControlOutcome.SUCCESS,
        graph_revision=command.graph_revision,
    )


def test_event_fsync_failure_does_not_advance_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus = PersistentTypedEventBus(tmp_path / "events.jsonl")
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_simple_graph(),
        event_bus=bus,
    )
    scheduler.start()
    command = scheduler.next_commands()[0]
    before = scheduler.export_state()

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr("robomex.runtime.event_log.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        scheduler.on_event(_outcome(command))

    assert scheduler.export_state() == before
    assert scheduler.snapshot().activations[0].status is ActivationStatus.RUNNING
    assert not bus.contains("outcome")
    assert (tmp_path / "events.jsonl").read_bytes() == b""


def test_scheduler_store_serializes_conflicting_process_writers(tmp_path: Path) -> None:
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_simple_graph(),
    )
    state_json = scheduler.export_state().model_dump_json()
    journal = str(tmp_path / "scheduler.jsonl")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_store_append,
            args=(journal, state_json, reason, start, results),
        )
        for reason in ("writer-a", "writer-b")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=5) for _ in processes)
    assert outcomes == [
        ("error", "SchedulerStoreIntegrityError"),
        ("ok", True),
    ]
    store = JsonlSchedulerStateStore(journal)
    assert len(store.history(episode_id="episode", workflow_id="workflow")) == 1


def _service_action_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="service-action",
            entry_activation="move",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="move",
                    runner_kind=RunnerKind.SYSTEM_ACTION,
                    runner_ref="test.move",
                    effect_scope=EffectScope.AUTHORITATIVE_WORLD,
                    authority_world_id="live",
                    authoritative_resource="robot.arm",
                ),
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
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
            transitions=(
                TransitionSpec(
                    source="move",
                    outcome=ControlOutcome.SUCCESS,
                    target="done",
                ),
            ),
        )
    )


def test_scheduler_state_roundtrip_restores_service_command_and_exact_lease(
    tmp_path: Path,
) -> None:
    bus_path = tmp_path / "events.jsonl"
    store = JsonlSchedulerStateStore(tmp_path / "scheduler.jsonl")
    authority = AuthorityRegistry()
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_service_action_graph(),
        event_bus=PersistentTypedEventBus(bus_path),
        authority_registry=authority,
        state_sink=store,
    )
    scheduler.start()
    service = scheduler.next_commands()[0]
    scheduler.on_event(
        ServiceOutcome(
            event_id="tracker-started",
            episode_id="episode",
            workflow_id="workflow",
            source="test.tracker",
            activation_id="tracker",
            command_id=service.command_id,
            attempt=service.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=service.graph_revision,
        )
    )
    action = scheduler.next_commands()[0]
    state = scheduler.export_state()

    restored_authority = AuthorityRegistry()
    restored = store.recover_scheduler(
        episode_id="episode",
        workflow_id="workflow",
        event_bus=PersistentTypedEventBus(bus_path),
        authority_registry=restored_authority,
    )

    assert restored.export_state() == state
    assert restored.next_commands() == ()
    recovered_service = restored.recovery_commands()[0]
    assert recovered_service.command_id == service.command_id
    assert recovered_service.attempt == service.attempt
    assert restored.recovery_commands() == ()
    restored.on_event(
        ServiceOutcome(
            event_id="tracker-rehydrated",
            episode_id="episode",
            workflow_id="workflow",
            source="test.tracker",
            activation_id="tracker",
            command_id=service.command_id,
            attempt=service.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=service.graph_revision,
        )
    )
    recovered_action = restored.recovery_commands()[0]
    assert recovered_action.command_id == action.command_id
    assert recovered_action.attempt == action.attempt
    assert recovered_action.lease == action.lease
    assert restored.recovery_commands() == ()
    assert action.lease is not None
    assert restored_authority.validate(
        lease_id=action.lease.lease_id,
        world_id="live",
        resource_id="robot.arm",
        action_id=action.command_id,
        holder_id="workflow:move:1",
    )
    assert not restored_authority.validate(
        lease_id=action.lease.lease_id,
        world_id="live",
        resource_id="robot.arm",
        action_id="forged-action",
    )

    restored.on_event(_outcome(recovered_action, event_id="move-complete"))
    assert restored_authority.active() == ()
    assert restored.next_commands()[0].activation_id == "done"


def test_failed_restore_does_not_leak_authority_lease(tmp_path: Path) -> None:
    store = JsonlSchedulerStateStore(tmp_path / "scheduler.jsonl")
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_service_action_graph(),
        event_bus=PersistentTypedEventBus(tmp_path / "events.jsonl"),
        authority_registry=AuthorityRegistry(),
        state_sink=store,
    )
    scheduler.start()
    service = scheduler.next_commands()[0]
    scheduler.on_event(
        ServiceOutcome(
            event_id="tracker-started",
            episode_id="episode",
            workflow_id="workflow",
            source="test.tracker",
            activation_id="tracker",
            command_id=service.command_id,
            attempt=service.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=service.graph_revision,
        )
    )
    action = scheduler.next_commands()[0]
    assert action.lease is not None

    restored_authority = AuthorityRegistry()
    with pytest.raises(SchedulerError, match="references missing event"):
        store.recover_scheduler(
            episode_id="episode",
            workflow_id="workflow",
            event_bus=PersistentTypedEventBus(tmp_path / "missing-events.jsonl"),
            authority_registry=restored_authority,
        )

    assert restored_authority.active() == ()


class _FailRuntimeEventCheckpointOnce:
    def __init__(self, delegate: JsonlSchedulerStateStore) -> None:
        self.delegate = delegate
        self.failed = False

    def append(self, state, *, reason: str, metadata=None) -> bool:
        if reason == "runtime_event" and not self.failed:
            self.failed = True
            raise OSError("injected checkpoint failure")
        return self.delegate.append(state, reason=reason, metadata=metadata)


class _FailGraphPatchCheckpointOnce:
    def __init__(self, delegate: JsonlSchedulerStateStore) -> None:
        self.delegate = delegate

    def append(self, state, *, reason: str, metadata=None) -> bool:
        if reason == "graph_patch":
            raise OSError("injected graph checkpoint failure")
        return self.delegate.append(state, reason=reason, metadata=metadata)


def test_restart_replays_event_durable_before_failed_state_checkpoint(
    tmp_path: Path,
) -> None:
    bus_path = tmp_path / "events.jsonl"
    store = JsonlSchedulerStateStore(tmp_path / "scheduler.jsonl")
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_simple_graph(),
        event_bus=PersistentTypedEventBus(bus_path),
        state_sink=_FailRuntimeEventCheckpointOnce(store),
    )
    scheduler.start()
    command = scheduler.next_commands()[0]

    with pytest.raises(OSError, match="injected checkpoint failure"):
        scheduler.on_event(_outcome(command))
    assert scheduler.snapshot().activations[0].status is ActivationStatus.RUNNING

    restored = store.recover_scheduler(
        episode_id="episode",
        workflow_id="workflow",
        event_bus=PersistentTypedEventBus(bus_path),
        replay_pending_events=True,
    )
    by_id = {item.activation_id: item for item in restored.snapshot().activations}
    assert by_id["work"].status is ActivationStatus.COMPLETED
    assert by_id["done"].status is ActivationStatus.READY
    assert restored.next_commands()[0].activation_id == "done"


def _patchable_graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="patch-restart",
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


def test_dynamic_graph_commit_recovers_successor_and_frontier(tmp_path: Path) -> None:
    bus_path = tmp_path / "events.jsonl"
    store = JsonlSchedulerStateStore(tmp_path / "scheduler.jsonl")
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_patchable_graph(),
        event_bus=PersistentTypedEventBus(bus_path),
        state_sink=store,
    )
    scheduler.start()
    prepare = scheduler.next_commands()[0]
    scheduler.on_event(_outcome(prepare, event_id="prepare-complete"))
    slot = derive_closed_slot(
        scheduler.graph,
        slot_id="replace-closed",
        target_activation_ids=("closed",),
    )
    coordinator = GraphPatchCoordinator(
        scheduler=scheduler,
        frontier=ComposableFrontier(
            graph_id=scheduler.graph.spec.graph_id,
            revision=scheduler.graph.spec.revision,
            graph_digest=scheduler.graph.digest,
            slots=(slot,),
        ),
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
    receipt = coordinator.fill_slot(
        slot_id=slot.slot_id,
        base_revision=1,
        fragment=fragment,
        patch_id="patch-1",
    )
    assert receipt.accepted

    latest = store.latest(episode_id="episode", workflow_id="workflow")
    assert latest is not None
    commit = GraphPatchCommit.model_validate(latest.metadata["graph_patch_commit"])
    assert store.graph_patch_commits(
        episode_id="episode", workflow_id="workflow"
    ) == (commit,)
    restored = store.recover_scheduler(
        episode_id="episode",
        workflow_id="workflow",
        event_bus=PersistentTypedEventBus(bus_path),
    )
    restored_coordinator = store.recover_patch_coordinator(scheduler=restored)

    assert restored.graph.spec.revision == 2
    assert restored.graph.digest == receipt.after_digest
    assert restored_coordinator is not None
    assert restored_coordinator.frontier == commit.next_frontier
    assert restored_coordinator.receipts == (receipt,)
    assert restored.next_commands()[0].activation_id == "replacement"


def test_graph_checkpoint_failure_rolls_back_scheduler_and_frontier(
    tmp_path: Path,
) -> None:
    store = JsonlSchedulerStateStore(tmp_path / "scheduler.jsonl")
    scheduler = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=_patchable_graph(),
        state_sink=_FailGraphPatchCheckpointOnce(store),
    )
    scheduler.start()
    prepare = scheduler.next_commands()[0]
    scheduler.on_event(_outcome(prepare, event_id="prepare-complete"))
    slot = derive_closed_slot(
        scheduler.graph,
        slot_id="replace-closed",
        target_activation_ids=("closed",),
    )
    coordinator = GraphPatchCoordinator(
        scheduler=scheduler,
        frontier=ComposableFrontier(
            graph_id=scheduler.graph.spec.graph_id,
            revision=scheduler.graph.spec.revision,
            graph_digest=scheduler.graph.digest,
            slots=(slot,),
        ),
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
    before = scheduler.export_state()

    with pytest.raises(OSError, match="graph checkpoint failure"):
        coordinator.fill_slot(
            slot_id=slot.slot_id,
            base_revision=1,
            fragment=fragment,
            patch_id="patch-fails-fsync",
        )

    assert scheduler.export_state() == before
    assert scheduler.graph.spec.revision == 1
    assert coordinator.frontier.slots == (slot,)
    assert coordinator.commits == ()
    assert len(coordinator.versions) == 1
