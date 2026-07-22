from __future__ import annotations

import pytest

from robomex.elastic import ElasticGraphCompiler
from robomex.runtime.activation import (
    ActivationScheduler,
    ActivationStatus,
    AuthorityConflictError,
    AuthorityRegistry,
    SchedulerError,
    TypedEventBus,
    WorkflowStatus,
)
from robomex.runtime.events import (
    ControlOutcome,
    NodeOutcomeEvent,
    ServiceOutcome,
    ServiceStatus,
)
from robomex.test.test_elastic_graph_v2 import _loop_graph


def _event(
    command,
    outcome: ControlOutcome,
    *,
    event_id: str,
    episode_id: str = "episode-1",
    workflow_id: str = "workflow-1",
) -> NodeOutcomeEvent:
    return NodeOutcomeEvent(
        event_id=event_id,
        episode_id=episode_id,
        workflow_id=workflow_id,
        source=f"runner:{command.activation_id}",
        activation_id=command.activation_id,
        node_id=command.activation_id,
        command_id=command.command_id,
        attempt=command.attempt,
        outcome=outcome,
        graph_revision=command.graph_revision,
    )


def _scheduler(
    *, authority: AuthorityRegistry | None = None, workflow_id: str = "workflow-1"
) -> ActivationScheduler:
    return ActivationScheduler(
        episode_id="episode-1",
        workflow_id=workflow_id,
        graph=ElasticGraphCompiler().compile(_loop_graph()),
        authority_registry=authority,
    )


def _by_id(snapshot, activation_id):
    return next(item for item in snapshot.activations if item.activation_id == activation_id)


def _start(scheduler: ActivationScheduler):
    scheduler.start()
    service = scheduler.next_commands()[0]
    assert service.activation_id == "tracker"
    scheduler.on_event(
        ServiceOutcome(
            episode_id=service.episode_id,
            workflow_id=service.workflow_id,
            source="runner:tracker",
            activation_id=service.activation_id,
            command_id=service.command_id,
            attempt=service.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=service.graph_revision,
        )
    )
    return scheduler.next_commands()[0]


def test_scheduler_starts_service_and_one_primary_activation() -> None:
    scheduler = _scheduler()
    scheduler.start()

    commands = scheduler.next_commands()

    assert [(command.activation_id, command.operation) for command in commands] == [
        ("tracker", "start_service"),
    ]
    service = commands[0]
    scheduler.on_event(
        ServiceOutcome(
            episode_id=service.episode_id,
            workflow_id=service.workflow_id,
            source="runner:tracker",
            activation_id=service.activation_id,
            command_id=service.command_id,
            attempt=service.attempt,
            status=ServiceStatus.STARTED,
            graph_revision=service.graph_revision,
        )
    )
    assert scheduler.next_commands()[0].activation_id == "observe"
    snapshot = scheduler.snapshot()
    assert _by_id(snapshot, "observe").status == ActivationStatus.RUNNING
    assert _by_id(snapshot, "tracker").status == ActivationStatus.RUNNING
    assert snapshot.loop_iterations == {"alignment": 1}


def test_declared_loop_stops_at_iteration_bound() -> None:
    scheduler = _scheduler()
    observe = _start(scheduler)

    scheduler.on_event(_event(observe, ControlOutcome.SUCCESS, event_id="e1"))
    gate = scheduler.next_commands()[0]
    assert gate.activation_id == "gate"
    scheduler.on_event(
        _event(gate, ControlOutcome.NEEDS_ADJUSTMENT, event_id="e2")
    )
    correct = scheduler.next_commands()[0]
    assert correct.activation_id == "correct"
    assert correct.lease is not None
    scheduler.on_event(_event(correct, ControlOutcome.SUCCESS, event_id="e3"))
    assert scheduler.snapshot().loop_iterations == {"alignment": 2}
    observe = scheduler.next_commands()[0]
    assert observe.activation_id == "observe"

    scheduler.on_event(_event(observe, ControlOutcome.SUCCESS, event_id="e4"))
    gate = scheduler.next_commands()[0]
    scheduler.on_event(
        _event(gate, ControlOutcome.NEEDS_ADJUSTMENT, event_id="e5")
    )
    correct = scheduler.next_commands()[0]
    terminal = scheduler.on_event(
        _event(correct, ControlOutcome.SUCCESS, event_id="e6")
    )

    assert terminal.status == WorkflowStatus.EXHAUSTED
    assert terminal.terminal_outcome == ControlOutcome.EXHAUSTED
    assert scheduler.next_commands() == ()


def test_success_exits_loop_and_finishes_terminal() -> None:
    scheduler = _scheduler()
    observe = _start(scheduler)
    scheduler.on_event(_event(observe, ControlOutcome.SUCCESS, event_id="e1"))
    gate = scheduler.next_commands()[0]
    scheduler.on_event(_event(gate, ControlOutcome.SUCCESS, event_id="e2"))
    done = scheduler.next_commands()[0]

    terminal = scheduler.on_event(_event(done, ControlOutcome.SUCCESS, event_id="e3"))

    assert terminal.status == WorkflowStatus.SUCCEEDED
    assert terminal.terminal_activation == "done"


def test_duplicate_event_is_idempotent_and_different_workflow_fails() -> None:
    scheduler = _scheduler()
    observe = _start(scheduler)
    event = _event(observe, ControlOutcome.SUCCESS, event_id="same")

    first = scheduler.on_event(event)
    duplicate = scheduler.on_event(event)

    assert first.event_count == duplicate.event_count == 2
    with pytest.raises(SchedulerError, match="different episode/workflow"):
        scheduler.on_event(
            _event(
                scheduler.next_commands()[0],
                ControlOutcome.SUCCESS,
                event_id="wrong",
                workflow_id="other",
            )
        )


def test_event_bus_multicasts_without_splitting_control() -> None:
    bus = TypedEventBus()
    bus.subscribe("manager", kinds={"node_outcome"})
    bus.subscribe("recorder")
    scheduler = ActivationScheduler(
        episode_id="episode-1",
        workflow_id="workflow-1",
        graph=ElasticGraphCompiler().compile(_loop_graph()),
        event_bus=bus,
    )
    observe = _start(scheduler)
    bus.drain("recorder")  # service startup is multicast but irrelevant here
    event = _event(observe, ControlOutcome.SUCCESS, event_id="multicast")

    scheduler.on_event(event)

    assert bus.drain("manager") == (event,)
    assert bus.drain("recorder") == (event,)
    assert scheduler.next_commands()[0].activation_id == "gate"


def test_authority_is_scoped_per_world_resource_and_shadow_needs_no_lease() -> None:
    authority = AuthorityRegistry()
    first = authority.acquire(
        world_id="live", resource_id="robot.arm", holder_id="one", action_id="a1"
    )
    with pytest.raises(AuthorityConflictError):
        authority.acquire(
            world_id="live", resource_id="robot.arm", holder_id="two", action_id="a2"
        )
    other = authority.acquire(
        world_id="shadow-1", resource_id="robot.arm", holder_id="two", action_id="a2"
    )

    assert {lease.world_id for lease in authority.active()} == {"live", "shadow-1"}
    authority.release(first)
    authority.release(other)
    assert authority.active() == ()


def test_service_outcome_cannot_advance_primary_control() -> None:
    scheduler = _scheduler()
    scheduler.start()
    service = scheduler.next_commands()[0]

    with pytest.raises(SchedulerError, match="service event"):
        scheduler.on_event(_event(service, ControlOutcome.SUCCESS, event_id="service"))
