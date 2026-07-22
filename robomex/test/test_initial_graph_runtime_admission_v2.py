from __future__ import annotations

from pathlib import Path

import pytest

from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    RunnerKind,
)
from robomex.orchestration.actors import (
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.bootstrap import (
    IdentityConflictError,
    V2RuntimeFactory,
)
from robomex.orchestration.episode import EpisodeRuntime, EpisodeRuntimeError
from robomex.orchestration.intent import SubgoalIntent
from robomex.test.test_v2_bootstrap_manifest import _fixtures, _rebuild_manifest


def _graph(graph_id: str):
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id=graph_id,
            entry_activation="worker",
            terminal_activations=("worker",),
            activations=(
                ActivationSpec(
                    activation_id="worker",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="worker",
                ),
            ),
        )
    )


def _intent() -> SubgoalIntent:
    return SubgoalIntent(
        intent_id="graph-admission",
        instruction="run only a manifest-pinned workflow",
        success_rubric="the initial graph identity remains sealed",
    )


def test_factory_validates_graph_metadata_before_binding_identity(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    invalid = _rebuild_manifest(
        manifest,
        metadata={"allowed_initial_graph_digests": [manifest.graph_digest, 7]},
    )

    with pytest.raises(IdentityConflictError, match="strict JSON string list"):
        V2RuntimeFactory().build(
            config=config,
            manifest=invalid,
            dependencies=dependencies,
        )

    assert not config.identity_root.exists()


def test_factory_episode_rejects_direct_unpinned_open_workflow(tmp_path: Path) -> None:
    allowed_graph = _graph("allowed-graph")
    rejected_graph = _graph("rejected-graph")
    config, manifest, dependencies = _fixtures(tmp_path)
    pinned_digest = f"sha256:{allowed_graph.digest}"
    manifest = _rebuild_manifest(manifest, graph_digest=pinned_digest)
    config = config.model_copy(update={"graph_digest": pinned_digest})
    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )

    with pytest.raises(EpisodeRuntimeError, match="not admitted by the run manifest"):
        application.episode.open_workflow(
            workflow_id="bypass",
            intent=_intent(),
            graph=rejected_graph,
        )

    assert application.episode.workflow_descriptors.load("bypass") is None


def test_recovery_rechecks_durable_descriptor_against_current_allowset(
    tmp_path: Path,
) -> None:
    graph = _graph("recover-allowed")
    root = tmp_path / "episode"
    actors = ActorRegistry(
        {"in_memory": InMemoryAgentProvider()}, workspace_root=tmp_path / "actors-a"
    )
    first = EpisodeRuntime(
        episode_id="graph-recovery",
        episode_root=root,
        actors=actors,
        allowed_initial_graph_digests=frozenset({f"sha256:{graph.digest}"}),
    )
    first.register_actor_profile(
        "worker", ActorProfile(profile_id="worker", runner_kind="coding_worker")
    )
    first.open_workflow(workflow_id="wf", intent=_intent(), graph=graph)

    recovered = EpisodeRuntime(
        episode_id="graph-recovery",
        episode_root=root,
        actors=ActorRegistry(
            {"in_memory": InMemoryAgentProvider()},
            workspace_root=tmp_path / "actors-b",
        ),
        allowed_initial_graph_digests=frozenset({"sha256:" + "0" * 64}),
    )
    recovered.register_actor_profile(
        "worker", ActorProfile(profile_id="worker", runner_kind="coding_worker")
    )

    with pytest.raises(EpisodeRuntimeError, match="not admitted by the run manifest"):
        recovered.recover_workflow("wf")

    assert "wf" not in recovered._workflows
