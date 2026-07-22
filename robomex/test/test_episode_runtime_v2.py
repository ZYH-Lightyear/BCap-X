from __future__ import annotations

from pathlib import Path

import pytest

from robomex.data import AdmissionError, EpisodeDataPlane
from robomex.elastic import ElasticGraphCompiler, LifecycleScope
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    ActorState,
    InMemoryAgentProvider,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
    EpisodeRuntimeError,
)
from robomex.orchestration.intent import IntentStatus, SubgoalIntent
from robomex.runtime.activation import WorkflowStatus
from robomex.runtime.events import ControlOutcome, NodeOutcomeEvent
from robomex.test.test_elastic_graph_v2 import VALUE, _loop_graph


def _episode_graph():
    """Data/lifecycle fixture; physical system actions have a separate test."""

    raw = _loop_graph().model_dump(mode="json")
    correct = next(node for node in raw["activations"] if node["activation_id"] == "correct")
    correct.update(
        runner_kind="deterministic_gate",
        effect_scope="read_only",
        authority_world_id=None,
        authoritative_resource=None,
    )
    observe = next(node for node in raw["activations"] if node["activation_id"] == "observe")
    observe["estimated_budget"] = {
        "tokens": 37,
        "wall_time_ms": 250,
        "shadow_rollouts": 0,
        "authoritative_actions": 0,
        "actor_spawns": 1,
    }
    return ElasticGraphCompiler().compile(raw)


def _runtime(tmp_path: Path) -> tuple[EpisodeRuntime, InMemoryAgentProvider]:
    gate_calls = 0

    def handler(profile, invocation, isolation):
        nonlocal gate_calls
        activation_id = invocation.metadata["activation_id"]
        if activation_id == "tracker":
            return None
        if activation_id == "observe":
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    ArtifactEmission(
                        port="value",
                        schema_id=VALUE,
                        payload={"iteration": invocation.metadata["attempt"]},
                    ),
                ),
            )
        if activation_id == "gate":
            gate_calls += 1
            return ActivationExecutionResult(
                outcome=(
                    ControlOutcome.NEEDS_ADJUSTMENT
                    if gate_calls == 1
                    else ControlOutcome.SUCCESS
                ),
                artifacts=(
                    ArtifactEmission(
                        port="error", schema_id=VALUE, payload={"error": 0.01}
                    ),
                ),
            )
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    provider = InMemoryAgentProvider(handler)
    actors = ActorRegistry(
        {"in_memory": provider},
        namespace_root="ep_01",
        workspace_root=tmp_path / "actors",
    )
    runtime = EpisodeRuntime(
        episode_id="ep_01",
        episode_root=tmp_path / "episode",
        actors=actors,
    )
    for node in _episode_graph().spec.activations:
        runtime.register_actor_profile(
            node.runner_ref,
            ActorProfile(
                profile_id=f"profile_{node.activation_id}",
                runner_kind=node.runner_kind.value,
                lifecycle=(
                    ActorLifecycle.EPHEMERAL
                    if node.lifecycle == LifecycleScope.INVOCATION
                    else ActorLifecycle.SERVICE
                ),
                effect_ceiling=(
                    frozenset()
                    if node.effect_scope.value == "read_only"
                    else frozenset({node.effect_scope.value})
                ),
            ),
        )
    return runtime, provider


def test_episode_runtime_executes_loop_services_and_append_only_data(tmp_path: Path) -> None:
    runtime, provider = _runtime(tmp_path)
    workflow_id = runtime.open_workflow(
        workflow_id="place",
        intent=SubgoalIntent(
            intent_id="place_bowl",
            instruction="put the bowl on the plate",
            success_rubric="the bowl is supported by the plate",
        ),
        graph=_episode_graph(),
    )

    terminal = runtime.run_until_terminal(workflow_id)
    outcome = runtime.close_workflow(workflow_id)

    assert terminal.status == WorkflowStatus.SUCCEEDED
    assert terminal.loop_iterations == {"alignment": 2}
    assert outcome.status == IntentStatus.SUCCEEDED
    records = runtime.data_plane.artifacts
    assert [record.generation for record in records if record.port == "value"] == [1, 2]
    assert any(call[0] == "spawn" and "tracker" in call[1] for call in provider.calls)
    observe_handle = next(
        handle for handle in runtime.actors.handles() if "observe" in handle.actor_id
    )
    assert observe_handle.runtime.invocations[0].budget == {
        "model_calls": 0.0,
        "tokens": 37.0,
        "wall_time_ms": 250.0,
        "shadow_rollouts": 0.0,
        "authoritative_actions": 0.0,
        "actor_spawns": 1.0,
    }
    tracker = next(
        handle for handle in runtime.actors.handles() if "tracker" in handle.actor_id
    )
    assert tracker.state == ActorState.RETIRED
    assert runtime.authority_registry.active() == ()
    replay = EpisodeDataPlane(runtime.episode_root, episode_id="ep_01")
    assert replay.events() == runtime.data_plane.events()


def test_episode_service_survives_workflow_close_until_episode_close(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    raw = _episode_graph().spec.model_dump(mode="json")
    tracker = next(node for node in raw["activations"] if node["activation_id"] == "tracker")
    tracker["lifecycle"] = "episode"
    graph = ElasticGraphCompiler().compile(raw)
    workflow_id = runtime.open_workflow(
        workflow_id="place",
        intent=SubgoalIntent(
            intent_id="place_bowl",
            instruction="put the bowl on the plate",
            success_rubric="the bowl is supported by the plate",
        ),
        graph=graph,
    )
    runtime.run_until_terminal(workflow_id)
    runtime.close_workflow(workflow_id)
    tracker_handle = next(
        handle for handle in runtime.actors.handles() if "tracker" in handle.actor_id
    )

    assert tracker_handle.state == ActorState.ACTIVE
    runtime.close_episode()
    assert tracker_handle.state == ActorState.RETIRED


def test_graph_capabilities_must_fit_profile_before_actor_spawn(tmp_path: Path) -> None:
    runtime, provider = _runtime(tmp_path)
    raw = _episode_graph().spec.model_dump(mode="json")
    observe = next(
        node for node in raw["activations"] if node["activation_id"] == "observe"
    )
    observe["required_capabilities"] = ["perception.read"]
    graph = ElasticGraphCompiler().compile(raw)

    with pytest.raises(EpisodeRuntimeError, match="perception.read"):
        runtime.open_workflow(
            workflow_id="unauthorized",
            intent=SubgoalIntent(
                intent_id="inspect",
                instruction="inspect the scene",
                success_rubric="a typed observation exists",
            ),
            graph=graph,
        )

    assert provider.calls == ()


def test_async_success_without_required_artifact_fails_next_admission(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime(tmp_path)
    workflow_id = runtime.open_workflow(
        workflow_id="place",
        intent=SubgoalIntent(
            intent_id="place_bowl",
            instruction="put the bowl on the plate",
            success_rubric="the bowl is supported by the plate",
        ),
        graph=_episode_graph(),
    )
    runtime.execute_ready(workflow_id)  # service-first admission barrier
    invocations = runtime.next_invocations(workflow_id)
    observe = next(item for item in invocations if item.command.activation_id == "observe")

    snapshot = runtime.accept_node_outcome(
        workflow_id,
        NodeOutcomeEvent(
            episode_id="ep_01",
            workflow_id=workflow_id,
            source="external_runner",
            activation_id="observe",
            node_id="observe",
            command_id=observe.command.command_id,
            attempt=observe.command.attempt,
            outcome=ControlOutcome.SUCCESS,
            graph_revision=1,
        ),
    )

    assert observe.admission.bindings == ()
    assert snapshot.status == WorkflowStatus.RUNNING
    with pytest.raises(AdmissionError, match="observe.value"):
        runtime.next_invocations(workflow_id)
