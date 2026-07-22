from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.data import (
    ResolvedArtifactRef,
    StateCommitReceipt,
    StateTransitionProposal,
    StateTransitionProposalWire,
)
from robomex.elastic import (
    ActivationSpec,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExternalBinding,
    PortSpecV2,
    RunnerKind,
)
from robomex.orchestration.actors import ActorRegistry, InMemoryAgentProvider
from robomex.orchestration.episode import EpisodeRuntime, EpisodeRuntimeError
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.activation import WorkflowStatus
from robomex.runtime.events import ControlOutcome, NodeOutcomeEvent


def _runtime(
    root: Path,
    *,
    provider: InMemoryAgentProvider | None = None,
) -> tuple[EpisodeRuntime, InMemoryAgentProvider]:
    provider = provider or InMemoryAgentProvider(lambda *_args: None)
    actors = ActorRegistry(
        {"in_memory": provider},
        namespace_root="reducer",
        workspace_root=root / "actors",
    )
    return (
        EpisodeRuntime(
            episode_id="episode",
            episode_root=root / "episode",
            actors=actors,
        ),
        provider,
    )


def _reducer_graph(*, runner_ref: str = "robomex.runtime.embodied_state_reducer"):
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="state-reducer",
            entry_activation="commit_state",
            terminal_activations=("commit_state",),
            activations=(
                ActivationSpec(
                    activation_id="commit_state",
                    runner_kind=RunnerKind.REDUCER,
                    runner_ref=runner_ref,
                    effect_scope=EffectScope.READ_ONLY,
                    inputs=(
                        PortSpecV2(
                            name="proposal",
                            schema_id="robomex.state_transition_proposal.v1",
                        ),
                    ),
                    outputs=(
                        PortSpecV2(
                            name="receipt",
                            schema_id="robomex.state_commit_receipt.v1",
                        ),
                    ),
                    bindings=(
                        ExternalBinding(
                            input_port="proposal",
                            ref="proposal_ref",
                            schema_id="robomex.state_transition_proposal.v1",
                        ),
                    ),
                ),
            ),
        )
    )


def _intent() -> SubgoalIntent:
    return SubgoalIntent(
        intent_id="commit_state",
        instruction="commit the evidence-backed state transition",
        success_rubric="the sole reducer emits a durable receipt",
    )


def _evidence(runtime: EpisodeRuntime, *, workflow_id: str = "seed"):
    runtime.data_plane.open_workflow(workflow_id)
    return runtime.data_plane.publish(
        workflow_id=workflow_id,
        activation_id="camera",
        attempt=1,
        port="evidence",
        schema="test.evidence.v1",
        payload={"observation_id": "obs_1"},
    ).ref


def _publish_proposal(
    runtime: EpisodeRuntime,
    proposal: StateTransitionProposal,
    *,
    workflow_id: str = "seed",
):
    if runtime.data_plane.workflows.get(workflow_id) is None:
        runtime.data_plane.open_workflow(workflow_id)
    wire = StateTransitionProposalWire.from_domain(proposal)
    return runtime.data_plane.publish(
        workflow_id=workflow_id,
        activation_id="state_author",
        attempt=1,
        port="proposal",
        schema="robomex.state_transition_proposal.v1",
        payload=wire.model_dump(mode="json"),
        lineage=proposal.evidence_refs,
    ).ref


def _register_entity_proposal(
    evidence_ref: ResolvedArtifactRef,
    *,
    before_revision: int = 0,
    effect_id: str = "register_bowl",
    semantic_label: str = "bowl",
) -> StateTransitionProposal:
    return StateTransitionProposal.register_entity(
        episode_id="episode",
        effect_id=effect_id,
        before_revision=before_revision,
        source="geometry_worker",
        evidence_refs=(evidence_ref,),
        entity_id="bowl_1",
        semantic_label=semantic_label,
        track_id="track_bowl_1",
    )


def _open(
    runtime: EpisodeRuntime,
    proposal_ref: ResolvedArtifactRef,
    *,
    workflow_id: str = "workflow",
) -> None:
    runtime.open_workflow(
        workflow_id=workflow_id,
        intent=_intent(),
        graph=_reducer_graph(),
        external_refs={"proposal_ref": proposal_ref},
    )


def _provider_invocations(provider: InMemoryAgentProvider) -> list[tuple]:
    return [call for call in provider.calls if call[0] == "invoke"]


def test_runtime_owned_reducer_commits_and_emits_typed_receipt(tmp_path: Path) -> None:
    runtime, provider = _runtime(tmp_path)
    evidence_ref = _evidence(runtime)
    proposal_ref = _publish_proposal(
        runtime, _register_entity_proposal(evidence_ref)
    )
    _open(runtime, proposal_ref)

    terminal = runtime.run_until_terminal("workflow")

    assert terminal.status is WorkflowStatus.SUCCEEDED
    assert runtime.state_reducer.state.revision == 1
    assert runtime.state_reducer.state.entity("bowl_1") is not None
    assert runtime.state_reducer.event_count == 1
    assert _provider_invocations(provider) == []
    receipt_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.schema == "robomex.state_commit_receipt.v1"
    )
    receipt = StateCommitReceipt.model_validate(
        runtime.data_plane.resolve(receipt_record.ref).payload
    )
    assert receipt.effect_id == "register_bowl"
    assert receipt.before_revision == 0
    assert receipt.after_revision == 1
    assert receipt.proposal_ref.to_domain() == proposal_ref
    node_outcome = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent)
    )
    assert node_outcome.outcome is ControlOutcome.SUCCESS
    assert node_outcome.artifact_ids == (receipt_record.artifact_id,)


@pytest.mark.parametrize("failure", ["stale", "evidence"])
def test_reducer_rejects_stale_or_unresolvable_evidence_without_state_change(
    tmp_path: Path,
    failure: str,
) -> None:
    runtime, provider = _runtime(tmp_path)
    evidence_ref = _evidence(runtime)
    if failure == "evidence":
        evidence_ref = ResolvedArtifactRef(
            artifact_id="art:episode:missing:evidence",
            content_digest="sha256:" + "0" * 64,
        )
    proposal = _register_entity_proposal(
        evidence_ref,
        before_revision=1 if failure == "stale" else 0,
    )
    # An unresolvable evidence claim is structurally valid.  Do not put the
    # forged ref in artifact lineage: publication provenance is itself strict.
    runtime.data_plane.open_workflow("proposal_seed")
    proposal_ref = runtime.data_plane.publish(
        workflow_id="proposal_seed",
        activation_id="state_author",
        attempt=1,
        port="proposal",
        schema="robomex.state_transition_proposal.v1",
        payload=StateTransitionProposalWire.from_domain(proposal).model_dump(
            mode="json"
        ),
    ).ref
    _open(runtime, proposal_ref)

    terminal = runtime.run_until_terminal("workflow")

    assert terminal.status is not WorkflowStatus.SUCCEEDED
    assert runtime.state_reducer.state.revision == 0
    assert runtime.state_reducer.event_count == 0
    assert not any(
        record.schema == "robomex.state_commit_receipt.v1"
        for record in runtime.data_plane.artifacts
    )
    assert _provider_invocations(provider) == []
    outcome = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent)
    )
    assert outcome.outcome is ControlOutcome.STALE_INPUT


def test_commit_then_crash_recovers_same_command_without_second_state_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, provider = _runtime(tmp_path)
    evidence_ref = _evidence(first)
    proposal_ref = _publish_proposal(
        first, _register_entity_proposal(evidence_ref)
    )
    _open(first, proposal_ref)

    def crash_after_commit(*_args, **_kwargs):
        raise RuntimeError("simulated process crash after reducer fsync")

    monkeypatch.setattr(first, "_publish_outputs", crash_after_commit)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        first.execute_ready("workflow")
    scheduler_commit = first.scheduler_store.latest(
        episode_id="episode", workflow_id="workflow"
    )
    assert scheduler_commit is not None
    active = scheduler_commit.state.activations[0]
    assert active.active_command_id is not None
    assert first.state_reducer.state.revision == 1
    assert first.state_reducer.event_count == 1

    restarted, restarted_provider = _runtime(tmp_path)
    restarted.recover_workflow("workflow")
    terminal = restarted.run_until_terminal("workflow")

    assert terminal.status is WorkflowStatus.SUCCEEDED
    assert restarted.state_reducer.state.revision == 1
    assert restarted.state_reducer.event_count == 1
    assert _provider_invocations(provider) == []
    assert _provider_invocations(restarted_provider) == []
    receipts = [
        record
        for record in restarted.data_plane.artifacts
        if record.schema == "robomex.state_commit_receipt.v1"
    ]
    assert len(receipts) == 1
    outcome = next(
        event
        for event in restarted.event_bus.history
        if isinstance(event, NodeOutcomeEvent)
    )
    assert outcome.command_id == active.active_command_id
    assert outcome.attempt == active.attempts == 1


def test_same_effect_cannot_be_rebound_to_different_proposal(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    evidence_ref = _evidence(runtime)
    first_ref = _publish_proposal(
        runtime, _register_entity_proposal(evidence_ref), workflow_id="seed"
    )
    _open(runtime, first_ref, workflow_id="first")
    assert runtime.run_until_terminal("first").status is WorkflowStatus.SUCCEEDED

    second_ref = _publish_proposal(
        runtime,
        _register_entity_proposal(evidence_ref, semantic_label="different"),
        workflow_id="second_seed",
    )
    _open(runtime, second_ref, workflow_id="second")
    terminal = runtime.run_until_terminal("second")

    assert terminal.status is not WorkflowStatus.SUCCEEDED
    assert runtime.state_reducer.state.revision == 1
    assert runtime.state_reducer.event_count == 1


def test_reducer_contract_is_reserved_and_wire_refs_are_strict(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    with pytest.raises(EpisodeRuntimeError, match="runtime-owned"):
        runtime.open_workflow(
            workflow_id="bad",
            intent=_intent(),
            graph=_reducer_graph(runner_ref="agent.fake_reducer"),
        )

    evidence = {
        "artifact_id": "art:episode:seed:evidence",
        "content_digest": "sha256:" + "a" * 64,
    }
    payload = _register_entity_proposal(
        ResolvedArtifactRef.from_any(evidence)
    ).to_mapping()
    payload["extra"] = "forbidden"
    with pytest.raises(ValidationError):
        StateTransitionProposalWire.model_validate(payload)
    payload.pop("extra")
    payload["evidence_refs"] = ["art:episode:seed:evidence"]
    with pytest.raises(ValidationError):
        StateTransitionProposalWire.model_validate(payload)
    payload["evidence_refs"] = [evidence]
    payload.update(
        {
            "kind": "set_localization",
            "localization_status": "localized",
            "world_pose": {
                "position_xyz_m": [float("nan"), 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "frame_id": "world",
            },
        }
    )
    with pytest.raises(ValidationError):
        StateTransitionProposalWire.model_validate(payload)
