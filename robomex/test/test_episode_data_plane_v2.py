from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from robomex.data import (
    AdmissionError,
    ArtifactIntegrityError,
    ArtifactResolutionError,
    AttachmentStatus,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    ResolvedArtifactRef,
    StateTransitionKind,
    StateTransitionProposal,
    StateTransitionRejected,
    SymbolicArtifactRef,
    WorkflowScopeError,
)


def _plane(tmp_path: Path) -> EpisodeDataPlane:
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="ep_01")
    plane.open_workflow("pick")
    return plane


def _publish(
    plane: EpisodeDataPlane,
    *,
    activation: str = "ground",
    port: str = "mask",
    value: int = 1,
):
    return plane.publish(
        workflow_id="pick",
        activation_id=activation,
        attempt=1,
        port=port,
        schema="robomex.test_evidence.v1",
        payload={"value": value},
    )


def test_append_only_generations_and_admission_freeze(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    first = _publish(plane, value=1)
    admission = plane.admit_inputs(
        admission_id="admit_geom",
        workflow_id="pick",
        activation_id="geometry",
        bindings={"mask": SymbolicArtifactRef("ground.mask")},
    )
    second = _publish(plane, value=2)

    assert first.generation == 1
    assert second.generation == 2
    assert first.artifact_id != second.artifact_id
    assert first.content_digest != second.content_digest
    assert plane.resolve(first.ref).payload == {"value": 1}
    assert plane.resolve(second.ref).payload == {"value": 2}

    # The same admission remains bound to generation 1 after alias advancement.
    repeated = plane.admit_inputs(
        admission_id="admit_geom",
        workflow_id="pick",
        activation_id="geometry",
        bindings={"mask": {"$ref": "ground.mask"}},
    )
    assert repeated == admission
    assert repeated.refs()["mask"] == first.ref
    with pytest.raises(AdmissionError, match="immutable"):
        plane.admit_inputs(
            admission_id="admit_geom",
            workflow_id="pick",
            activation_id="other_consumer",
            bindings={"mask": {"$ref": "ground.mask"}},
        )


def test_alias_is_workflow_local_but_explicit_ref_is_episode_scoped(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path)
    artifact = _publish(plane)
    plane.open_workflow("place")

    with pytest.raises(AdmissionError, match="Workflow-local"):
        plane.admit_inputs(
            admission_id="place_symbolic",
            workflow_id="place",
            activation_id="place_plan",
            bindings={"mask": {"$ref": "ground.mask"}},
        )

    explicit = plane.admit_inputs(
        admission_id="place_explicit",
        workflow_id="place",
        activation_id="place_plan",
        bindings={"mask": artifact.ref},
    )
    assert explicit.refs()["mask"] == artifact.ref

    plane.close_workflow("pick")
    with pytest.raises(WorkflowScopeError, match="not open"):
        _publish(plane, value=3)


def test_resolver_rejects_paths_cross_episode_digest_and_escape(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path)
    first = _publish(plane, value=1)
    second = _publish(plane, value=2)

    with pytest.raises(ArtifactResolutionError, match="both artifact_id"):
        plane.resolver.resolve(first.artifact_id)
    with pytest.raises(ArtifactResolutionError, match="paths"):
        plane.resolver.resolve(Path(first.content_path))
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        plane.resolver.resolve(ResolvedArtifactRef(first.artifact_id, second.content_digest))
    with pytest.raises(ArtifactResolutionError):
        plane.resolver.resolve(
            ResolvedArtifactRef(
                first.artifact_id.replace("art:ep_01:", "art:other:"),
                first.content_digest,
            )
        )

    # Even a path injected into the derived index cannot redirect resolution.
    index = json.loads(plane.index_path.read_text(encoding="utf-8"))
    index["artifacts"][first.artifact_id]["content_path"] = "../outside.json"
    plane.index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        plane.resolver.resolve(first.ref)


def test_artifact_events_rebuild_index_and_replay_exactly(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    first = _publish(plane, value=7)
    plane.admit_inputs(
        admission_id="consume",
        workflow_id="pick",
        activation_id="consumer",
        bindings={"evidence": {"$ref": "ground.mask"}},
    )
    expected_events = plane.events()
    plane.index_path.unlink()

    replayed = EpisodeDataPlane(plane.episode_root, episode_id="ep_01")

    assert replayed.events() == expected_events
    assert replayed.resolve(first.ref).payload == {"value": 7}
    assert replayed.admission("consume").refs()["evidence"] == first.ref
    assert json.loads(replayed.index_path.read_text(encoding="utf-8"))["event_count"] == len(
        expected_events
    )
    with pytest.raises(ArtifactIntegrityError, match="different identity"):
        EpisodeDataPlane(plane.episode_root, episode_id="ep_02")


def test_payload_digest_is_canonical_and_non_finite_values_fail(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    first = plane.publish(
        workflow_id="pick",
        activation_id="a",
        attempt=1,
        port="out",
        schema="test.mapping.v1",
        payload={"b": 2, "a": 1},
    )
    second = plane.publish(
        workflow_id="pick",
        activation_id="b",
        attempt=1,
        port="out",
        schema="test.mapping.v1",
        payload={"a": 1, "b": 2},
    )
    assert first.content_digest == second.content_digest
    assert first.content_path == second.content_path

    with pytest.raises(ValueError, match="finite"):
        plane.publish(
            workflow_id="pick",
            activation_id="bad",
            attempt=1,
            port="out",
            schema="test.mapping.v1",
            payload={"value": float("nan")},
        )


def test_reducer_is_only_commit_path_and_rejects_stale_duplicate_illegal(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path)
    entity_evidence = _publish(plane, activation="ground", port="entity", value=1)
    attempt_evidence = _publish(plane, activation="runtime", port="attempt", value=2)
    verify_evidence = _publish(plane, activation="verify", port="held", value=3)
    reducer = EmbodiedStateReducer(
        plane.episode_root,
        episode_id="ep_01",
        resolver=plane.resolver,
    )

    state = reducer.commit(
        StateTransitionProposal.register_entity(
            episode_id="ep_01",
            effect_id="register_bowl",
            before_revision=0,
            source="grounding_agent",
            evidence_refs=(entity_evidence.ref,),
            entity_id="bowl_01",
            semantic_label="bowl",
            track_id="track_7",
        )
    )
    assert state.revision == 1
    assert state.entity("bowl_01") is not None
    with pytest.raises(FrozenInstanceError):
        state.revision = 99  # type: ignore[misc]

    before_events = reducer.event_count
    with pytest.raises(StateTransitionRejected, match="reachable only|Illegal"):
        reducer.commit(
            StateTransitionProposal.set_attachment(
                episode_id="ep_01",
                effect_id="skip_attempt",
                before_revision=1,
                source="learned_verifier",
                evidence_refs=(verify_evidence.ref,),
                entity_id="bowl_01",
                status=AttachmentStatus.VERIFIED_HELD,
                action_id="grasp_1",
            )
        )
    assert reducer.state.revision == 1
    assert reducer.event_count == before_events

    attempted = reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_01",
            effect_id="attempt_grasp",
            before_revision=1,
            source="action_runtime",
            evidence_refs=(attempt_evidence.ref,),
            entity_id="bowl_01",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp_1",
        )
    )
    assert attempted.attachment.status is AttachmentStatus.ATTEMPTED
    assert attempted.held_entity_id is None

    held = reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_01",
            effect_id="verify_grasp",
            before_revision=2,
            source="learned_verifier",
            evidence_refs=(verify_evidence.ref,),
            entity_id="bowl_01",
            status=AttachmentStatus.VERIFIED_HELD,
            action_id="grasp_1",
        )
    )
    assert held.held_entity_id == "bowl_01"

    with pytest.raises(StateTransitionRejected, match="Stale"):
        reducer.commit(
            StateTransitionProposal.set_attachment(
                episode_id="ep_01",
                effect_id="stale_open",
                before_revision=2,
                source="monitor",
                evidence_refs=(verify_evidence.ref,),
                entity_id="bowl_01",
                status=AttachmentStatus.UNKNOWN,
            )
        )
    with pytest.raises(StateTransitionRejected, match="Duplicate"):
        reducer.commit(
            StateTransitionProposal.set_attachment(
                episode_id="ep_01",
                effect_id="verify_grasp",
                before_revision=3,
                source="monitor",
                evidence_refs=(verify_evidence.ref,),
                entity_id="bowl_01",
                status=AttachmentStatus.UNKNOWN,
            )
        )


def test_reducer_requires_resolved_evidence_and_replays_state(tmp_path: Path) -> None:
    plane = _plane(tmp_path)
    evidence = _publish(plane, activation="ground", port="entity")
    reducer = EmbodiedStateReducer(plane.episode_root, episode_id="ep_01")

    with pytest.raises(StateTransitionRejected, match="admission-resolved"):
        reducer.commit(
            StateTransitionProposal(
                episode_id="ep_01",
                effect_id="bad_evidence",
                before_revision=0,
                kind=StateTransitionKind.REGISTER_ENTITY,
                source="agent",
                evidence_refs=({"$ref": "ground.entity"},),  # type: ignore[arg-type]
                entity_id="obj_1",
                semantic_label="object",
            )
        )

    reducer.commit(
        StateTransitionProposal.register_entity(
            episode_id="ep_01",
            effect_id="register_obj",
            before_revision=0,
            source="grounding_agent",
            evidence_refs=(evidence.ref,),
            entity_id="obj_1",
            semantic_label="object",
        )
    )
    expected_state = reducer.state
    expected_events = reducer.events()
    reducer.index_path.unlink()

    replayed = EmbodiedStateReducer(plane.episode_root, episode_id="ep_01")

    assert replayed.state == expected_state
    assert replayed.events() == expected_events
    assert (
        json.loads(replayed.index_path.read_text(encoding="utf-8"))["state"]
        == expected_state.to_mapping()
    )
