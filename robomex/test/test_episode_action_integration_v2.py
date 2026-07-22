from __future__ import annotations

from datetime import datetime, timezone

import pytest

from robomex.elastic import (
    ActivationSpec,
    ArtifactBinding,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)
from robomex.orchestration.actors import (
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
from robomex.runtime.action_protocol import ExecutionReceipt, FeasibilityStatus
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ActionOutcome, ArtifactPublished, ControlOutcome
from robomex.runtime.evidence_recorder import (
    ACTION_RUNTIME_SAMPLE_SCHEMA,
    ActionEvidenceFrame,
    ActionEvidenceVideo,
    ActionRuntimeEvidenceSample,
    decode_evidence_ref,
)
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion, _snapshot


def _fresh_motion():
    return _motion(_snapshot(captured_at=datetime.now(timezone.utc)))  # noqa: UP017


def test_episode_routes_system_action_only_through_sealed_runtime(tmp_path) -> None:
    plan = _fresh_motion()

    class Checker:
        def certify(self, spec, snapshot):
            return build_feasibility_certificate(
                spec=spec,
                snapshot=snapshot,
                checker_id="fake-ik-collision-checker",
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

    def handler(profile, invocation, isolation):
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

    provider = InMemoryAgentProvider(handler)
    actors = ActorRegistry(
        {"in_memory": provider},
        namespace_root="ep",
        workspace_root=tmp_path / "actors",
    )
    runtime = EpisodeRuntime(
        episode_id="ep",
        episode_root=tmp_path / "episode",
        actors=actors,
        action_backends={
            (plan.world_id, plan.resource_id): FakeBackend(plan.expected_snapshot)
        },
        feasibility_checkers={(plan.world_id, plan.resource_id): Checker()},
    )
    runtime.register_actor_profile(
        "test.author",
        ActorProfile(profile_id="author", runner_kind="coding_worker"),
    )
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="sealed_motion",
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
                    source="author", outcome=ControlOutcome.SUCCESS, target="execute"
                ),
            ),
        )
    )
    workflow_id = runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="motion",
            instruction="execute one sealed segment",
            success_rubric="the exact joint segment converges",
        ),
        graph=graph,
    )

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "succeeded"
    records = runtime.action_wal.records()
    assert [record.record_kind for record in records] == [
        "action_attempt",
        "primitive_receipt",
        "execution_receipt",
    ]
    assert isinstance(records[-1], ExecutionReceipt)
    assert records[-1].feasibility_certificate_digest is not None
    assert records[0].scheduler_reservation_id is not None
    assert len(records[-1].frame_refs) == 3
    assert records[-1].video_ref is not None
    evidence_frames = tuple(
        ActionEvidenceFrame.model_validate(
            runtime.data_plane.resolve(decode_evidence_ref(frame_ref)).payload
        )
        for frame_ref in records[-1].frame_refs
    )
    assert tuple(frame.phase for frame in evidence_frames) == (
        "execution/pre_primitive",
        "primitive/post",
        "execution/terminal",
    )
    assert all(
        frame.sample["schema_version"] == ACTION_RUNTIME_SAMPLE_SCHEMA
        for frame in evidence_frames
    )
    for frame in evidence_frames:
        ActionRuntimeEvidenceSample.model_validate(frame.sample)
    evidence_video_ref = decode_evidence_ref(records[-1].video_ref)
    evidence_video = ActionEvidenceVideo.model_validate(
        runtime.data_plane.resolve(evidence_video_ref).payload
    )
    assert evidence_video.representation == "deterministic_manifest"
    assert tuple(item.ref for item in evidence_video.frame_refs) == tuple(
        decode_evidence_ref(value) for value in records[-1].frame_refs
    )
    assert sum(call[0] == "invoke" for call in provider.calls) == 1
    action_event = next(
        event for event in runtime.event_bus.history if isinstance(event, ActionOutcome)
    )
    assert action_event.action_id == records[-1].action_id
    assert action_event.plan_digest == plan.content_digest
    receipt_publication = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, ArtifactPublished)
        and event.schema_id == "robomex.execution_receipt.v2"
    )
    receipt_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.artifact_id == receipt_publication.artifact_id
    )
    resolved = runtime.data_plane.resolve(receipt_record.ref)
    assert resolved.payload["action_id"] == records[-1].action_id
    action_spec_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.schema == plan.schema_version
    )
    assert tuple(item.ref for item in evidence_video.source_lineage) == (
        action_spec_record.ref,
    )
    assert receipt_record.lineage == (
        action_spec_record.ref,
        *(decode_evidence_ref(value) for value in records[-1].frame_refs),
        evidence_video_ref,
    )
    assert runtime.authority_registry.active() == ()


def test_system_action_contract_is_rejected_before_backend_call(tmp_path) -> None:
    plan = _fresh_motion()
    backend = FakeBackend(plan.expected_snapshot)
    actors = ActorRegistry(
        {"in_memory": InMemoryAgentProvider()},
        namespace_root="preflight",
        workspace_root=tmp_path / "actors",
    )
    runtime = EpisodeRuntime(
        episode_id="preflight",
        episode_root=tmp_path / "episode",
        actors=actors,
        action_backends={(plan.world_id, plan.resource_id): backend},
        feasibility_checkers={(plan.world_id, plan.resource_id): object()},
    )
    runtime.register_actor_profile(
        "test.author",
        ActorProfile(profile_id="author", runner_kind="coding_worker"),
    )
    invalid = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="invalid_physical_contract",
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
                    authority_world_id=plan.world_id,
                    authoritative_resource="arm",
                    inputs=(
                        PortSpecV2(
                            name="action_spec", schema_id="robomex.motion_plan.v2"
                        ),
                    ),
                    outputs=(
                        PortSpecV2(
                            name="receipt", schema_id="robomex.motion_plan.v2"
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

    with pytest.raises(EpisodeRuntimeError, match="execution_receipt"):
        runtime.open_workflow(
            workflow_id="wf",
            intent=SubgoalIntent(
                intent_id="invalid",
                instruction="must not execute",
                success_rubric="backend stays untouched",
            ),
            graph=invalid,
        )

    assert backend.calls == []
