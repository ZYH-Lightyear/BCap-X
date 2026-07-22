from __future__ import annotations

from pathlib import Path

import pytest

from robomex.elastic import (
    ActivationLane,
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    LifecycleScope,
    PortSpecV2,
    RunnerKind,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.episode import (
    ArtifactEmission,
    EpisodeRuntime,
    ServiceEventResult,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.events import (
    ActionOutcome,
    ActionStatus,
    FindingSeverity,
    LifecycleEvent,
    LifecycleTransition,
    MonitorFinding,
    MonitorFindingKind,
    ServiceOutcome,
    ServiceStatus,
)
from robomex.runtime.service_delivery import (
    ServiceDeliveryIntegrityError,
    ServiceDeliveryLedger,
)


def _graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="service-dataflow",
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
                    subscriptions=("action_outcome",),
                    outputs=(
                        PortSpecV2(
                            name="evidence",
                            schema_id="test.service_evidence.v1",
                        ),
                    ),
                ),
            ),
        )
    )


def _intent() -> SubgoalIntent:
    return SubgoalIntent(
        intent_id="service-dataflow",
        instruction="keep the tracker alive while work is pending",
        success_rubric="the tracker consumes matching events",
    )


def _runtime(
    root: Path, handler
) -> tuple[EpisodeRuntime, InMemoryAgentProvider]:
    provider = InMemoryAgentProvider(handler)
    runtime = EpisodeRuntime(
        episode_id="episode",
        episode_root=root / "episode",
        actors=ActorRegistry(
            {"in_memory": provider},
            namespace_root="service-dataflow",
            workspace_root=root / "actors",
        ),
    )
    runtime.register_actor_profile(
        "test.work",
        ActorProfile(profile_id="work", runner_kind="coding_worker"),
    )
    runtime.register_actor_profile(
        "test.tracker",
        ActorProfile(
            profile_id="tracker",
            runner_kind="tracking_service",
            lifecycle=ActorLifecycle.SERVICE,
        ),
    )
    return runtime, provider


def _open_and_start(runtime: EpisodeRuntime) -> str:
    workflow_id = runtime.open_workflow(
        workflow_id="workflow",
        intent=_intent(),
        graph=_graph(),
    )
    runtime.execute_ready(workflow_id)
    return workflow_id


def _publish_action(runtime: EpisodeRuntime, workflow_id: str, action_id: str) -> str:
    receipt = runtime.data_plane.publish(
        workflow_id=workflow_id,
        activation_id="fixture",
        attempt=1,
        port=f"receipt_{action_id}",
        schema="test.receipt.v1",
        payload={"action_id": action_id, "status": "succeeded"},
    )
    runtime.event_bus.publish(
        ActionOutcome(
            event_id=f"action-{action_id}",
            episode_id="episode",
            workflow_id=workflow_id,
            source="test.action_backend",
            action_id=action_id,
            status=ActionStatus.SUCCEEDED,
            receipt_ref=receipt.artifact_id,
        )
    )
    return receipt.artifact_id


def test_service_subscription_publishes_artifact_and_finding_once_across_restart(
    tmp_path: Path,
) -> None:
    evidence: dict[str, str] = {}

    def handler(_profile, invocation, _isolation):
        if "delivery_id" not in invocation.metadata:
            return None
        action = ActionOutcome.model_validate(invocation.inputs["event"])
        receipt_ref = evidence[action.action_id]
        return ServiceEventResult(
            artifacts=(
                ArtifactEmission(
                    port="evidence",
                    schema_id="test.service_evidence.v1",
                    payload={
                        "action_id": action.action_id,
                        "assessment": "attachment_anomaly",
                    },
                ),
            ),
            events=(
                MonitorFinding(
                    event_id=f"finding-{action.action_id}",
                    finding_id=f"finding-{action.action_id}",
                    episode_id="episode",
                    workflow_id="workflow",
                    timestamp=action.timestamp,
                    source="actor:tracker",
                    monitor_id="tracker",
                    finding=MonitorFindingKind.ATTACHMENT_ANOMALY,
                    severity=FindingSeverity.CRITICAL,
                    evidence_refs=(receipt_ref,),
                    action_id=action.action_id,
                ),
            ),
        )

    first, provider = _runtime(tmp_path, handler)
    workflow_id = _open_and_start(first)
    first.event_bus.publish(
        LifecycleEvent(
            event_id="unmatched-lifecycle",
            episode_id="episode",
            workflow_id=workflow_id,
            source="test",
            actor_id="unrelated",
            transition=LifecycleTransition.INVOKED,
        )
    )
    evidence["move-1"] = _publish_action(first, workflow_id, "move-1")

    reports = first.pump_service_events(workflow_id)

    assert len(reports) == 1
    report = reports[0]
    assert report.status == "succeeded"
    assert len(report.artifact_ids) == 1
    assert report.emitted_event_ids == ("finding-move-1",)
    service_invocations = [
        invocation
        for invocation in provider.runtime_for("workflow_tracker").invocations
        if "delivery_id" in invocation.metadata
    ]
    assert len(service_invocations) == 1
    assert service_invocations[0].inputs["event"]["kind"] == "action_outcome"
    artifact = first.data_plane.artifact_record(report.artifact_ids[0])
    assert [ref.artifact_id for ref in artifact.lineage] == [evidence["move-1"]]
    finding = next(
        event
        for event in first.event_bus.history
        if isinstance(event, MonitorFinding)
    )
    assert finding.evidence_refs == (evidence["move-1"],)

    restarted, restarted_provider = _runtime(tmp_path, handler)
    restarted.recover_workflow(workflow_id)
    restarted.execute_ready(workflow_id)

    assert restarted.pump_service_events(workflow_id) == ()
    restarted_invocations = restarted_provider.runtime_for(
        "workflow_tracker"
    ).invocations
    assert len(restarted_invocations) == 1  # service rehydration only
    assert len(
        [event for event in restarted.event_bus.history if isinstance(event, MonitorFinding)]
    ) == 1
    assert len(
        [
            record
            for record in restarted.data_plane.artifacts
            if record.activation_id == "tracker"
        ]
    ) == 1


def test_inflight_service_delivery_reuses_invocation_identity_after_restart(
    tmp_path: Path,
) -> None:
    first_delivery_ids: list[str] = []

    def crash_handler(_profile, invocation, _isolation):
        if "delivery_id" not in invocation.metadata:
            return None
        first_delivery_ids.append(invocation.invocation_id)
        raise KeyboardInterrupt("simulated process loss after reservation")

    first, _ = _runtime(tmp_path, crash_handler)
    workflow_id = _open_and_start(first)
    _publish_action(first, workflow_id, "move-crash")
    with pytest.raises(KeyboardInterrupt, match="simulated process loss"):
        first.pump_service_events(workflow_id)

    def recovered_handler(_profile, invocation, _isolation):
        if "delivery_id" not in invocation.metadata:
            return None
        return ServiceEventResult()

    restarted, restarted_provider = _runtime(tmp_path, recovered_handler)
    restarted.recover_workflow(workflow_id)
    restarted.execute_ready(workflow_id)
    reports = restarted.pump_service_events(workflow_id)

    assert len(reports) == 1
    assert reports[0].invocation_id == first_delivery_ids[0]
    recovered_delivery = [
        invocation
        for invocation in restarted_provider.runtime_for(
            "workflow_tracker"
        ).invocations
        if "delivery_id" in invocation.metadata
    ]
    assert [item.invocation_id for item in recovered_delivery] == first_delivery_ids

    final, final_provider = _runtime(tmp_path, recovered_handler)
    final.recover_workflow(workflow_id)
    final.execute_ready(workflow_id)
    assert final.pump_service_events(workflow_id) == ()
    assert len(final_provider.runtime_for("workflow_tracker").invocations) == 1


def test_service_delivery_failure_is_bound_to_active_service_command(
    tmp_path: Path,
) -> None:
    def handler(_profile, invocation, _isolation):
        if "delivery_id" not in invocation.metadata:
            return None
        raise RuntimeError("tracker failed")

    runtime, _ = _runtime(tmp_path, handler)
    workflow_id = _open_and_start(runtime)
    started = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, ServiceOutcome)
        and event.status is ServiceStatus.STARTED
    )
    _publish_action(runtime, workflow_id, "move-failure")

    reports = runtime.pump_service_events(workflow_id)

    assert len(reports) == 1 and reports[0].status == "failed"
    failed = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, ServiceOutcome)
        and event.status is ServiceStatus.FAILED
    )
    assert (failed.command_id, failed.attempt, failed.graph_revision) == (
        started.command_id,
        started.attempt,
        started.graph_revision,
    )
    assert "tracker failed" in (failed.reason or "")
    assert runtime.scheduler_store.latest(
        episode_id="episode", workflow_id=workflow_id
    ).state.status.value == "uncertain"


def test_delivery_ledger_rejects_event_digest_rebinding(tmp_path: Path) -> None:
    ledger = ServiceDeliveryLedger(tmp_path / "service-deliveries.jsonl")
    ledger.bind_subscription(
        subscriber_id="subscriber",
        episode_id="episode",
        workflow_id="workflow",
        activation_id="tracker",
        command_id="command",
        graph_revision=1,
        kinds=("action_outcome",),
        start_offset=0,
    )
    ledger.reserve(
        delivery_id="delivery",
        subscriber_id="subscriber",
        event_id="event",
        event_kind="action_outcome",
        event_digest="a" * 64,
        event_offset=1,
        invocation_id="invocation",
        artifact_attempt=1,
    )

    with pytest.raises(
        ServiceDeliveryIntegrityError, match="identity or digest changed"
    ):
        ledger.reserve(
            delivery_id="delivery",
            subscriber_id="subscriber",
            event_id="event",
            event_kind="action_outcome",
            event_digest="b" * 64,
            event_offset=1,
            invocation_id="invocation",
            artifact_attempt=1,
        )
