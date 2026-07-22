from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.data import (
    AttachmentEvidence,
    AttachmentStatus,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    LocalizationStatus,
    PhysicalStateTrigger,
    Pose,
    RelationPredicate,
    RelationValue,
    StateTransitionProposal,
    StateTransitionRejected,
)


def _validity(*, observation_id: str, observation_revision: int, state_revision: int):
    return {
        "lifecycle": "derived",
        "depends_on_revisions": {"camera.front": observation_revision},
        "observation_id": observation_id,
        "state_revision": state_revision,
        "method": "causal_revision_match",
        "reason": "the verifier result is scoped to one frame and one state",
        "confidence": 0.99,
    }


def _fixture(tmp_path: Path, *, strict: bool):
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="ep_evidence")
    plane.open_workflow("place")
    sensor = plane.publish(
        workflow_id="place",
        activation_id="camera",
        attempt=1,
        port="frame",
        schema="test.camera_frame.v1",
        payload={"observation_id": "obs_7", "revision": 7},
    )
    reducer = EmbodiedStateReducer(
        plane.episode_root,
        episode_id="ep_evidence",
        resolver=plane.resolver,
        strict_evidence=strict,
    )
    for entity_id, label, track_id in (
        ("bowl_1", "bowl", "track_bowl"),
        ("plate_1", "plate", "track_plate"),
    ):
        reducer.commit(
            StateTransitionProposal.register_entity(
                episode_id="ep_evidence",
                effect_id=f"register_{entity_id}",
                before_revision=reducer.state.revision,
                source="grounding_agent",
                evidence_refs=(sensor.ref,),
                entity_id=entity_id,
                semantic_label=label,
                track_id=track_id,
            )
        )
    return plane, sensor, reducer


def _relation_payload(
    sensor_id: str,
    *,
    action_id: str = "place_7",
    target_entity_id: str = "plate_1",
    target_track_id: str = "track_plate",
    base_state_revision: int = 2,
    observation_revision: int = 7,
):
    return {
        "evidence_id": "relation_7",
        "subject_entity_id": "bowl_1",
        "subject_track_id": "track_bowl",
        "predicate": "on_top_of",
        "target_entity_id": target_entity_id,
        "target_track_id": target_track_id,
        "value": "asserted",
        "source_observation_id": "obs_7",
        "source_observation_revision": observation_revision,
        "observation_domain": "camera.front",
        "base_state_revision": base_state_revision,
        "action_id": action_id,
        "evidence_refs": [sensor_id],
        "method": "support_overlap_and_height",
        "reason": "the observed bowl footprint is supported by the plate",
        "confidence": 0.96,
        "validity": _validity(
            observation_id="obs_7",
            observation_revision=observation_revision,
            state_revision=base_state_revision,
        ),
    }


def test_state_evidence_schema_binds_observation_and_state_revision() -> None:
    payload = {
        "evidence_id": "attachment_1",
        "entity_id": "bowl_1",
        "entity_track_id": "track_bowl",
        "status": "verified_held",
        "source_observation_id": "obs_7",
        "source_observation_revision": 7,
        "observation_domain": "camera.front",
        "base_state_revision": 3,
        "action_id": "grasp_1",
        "evidence_refs": ["frame_artifact"],
        "method": "finger_and_visual_guard",
        "reason": "finger closure and tracked-object motion agree",
        "confidence": 0.98,
        "validity": _validity(
            observation_id="obs_7",
            observation_revision=8,
            state_revision=3,
        ),
    }
    with pytest.raises(ValidationError, match="source observation revision"):
        AttachmentEvidence.model_validate(payload)


def test_strict_relation_rejects_untyped_and_semantically_rebound_evidence(
    tmp_path: Path,
) -> None:
    plane, sensor, reducer = _fixture(tmp_path, strict=True)

    def proposal(effect_id: str, *evidence_refs):
        return StateTransitionProposal.set_relation(
            episode_id="ep_evidence",
            effect_id=effect_id,
            before_revision=2,
            source="relation_verifier",
            evidence_refs=evidence_refs,
            subject_entity_id="bowl_1",
            predicate=RelationPredicate.ON_TOP_OF,
            target_entity_id="plate_1",
            value=RelationValue.ASSERTED,
            source_observation_id="obs_7",
            source_observation_revision=7,
            source_observation_domain="camera.front",
            action_id="place_7",
            subject_track_id="track_bowl",
            target_track_id="track_plate",
        )

    with pytest.raises(StateTransitionRejected, match="typed evidence schema"):
        reducer.commit(proposal("untyped_relation", sensor.ref))

    unlined = plane.publish(
        workflow_id="place",
        activation_id="relation_verifier",
        attempt=1,
        port="unlined",
        schema="robomex.relation_evidence.v1",
        payload=_relation_payload(sensor.artifact_id),
    )
    with pytest.raises(StateTransitionRejected, match="evidence_lineage"):
        reducer.commit(proposal("unlined_relation", sensor.ref, unlined.ref))

    rebound = plane.publish(
        workflow_id="place",
        activation_id="relation_verifier",
        attempt=1,
        port="rebound",
        schema="robomex.relation_evidence.v1",
        payload=_relation_payload(sensor.artifact_id, action_id="some_other_action"),
        lineage=(sensor.ref,),
    )
    with pytest.raises(StateTransitionRejected, match="action_id"):
        reducer.commit(proposal("rebound_relation", sensor.ref, rebound.ref))

    rebound_cases = (
        ("wrong_target", {"target_entity_id": "other_plate"}, "target_entity_id"),
        ("wrong_track", {"target_track_id": "track_other_plate"}, "target_track_id"),
        ("wrong_state", {"base_state_revision": 99}, "base_state_revision"),
        ("wrong_frame", {"observation_revision": 8}, "source_observation_revision"),
    )
    for port, changes, expected_field in rebound_cases:
        rebound = plane.publish(
            workflow_id="place",
            activation_id="relation_verifier",
            attempt=1,
            port=port,
            schema="robomex.relation_evidence.v1",
            payload=_relation_payload(sensor.artifact_id, **changes),
            lineage=(sensor.ref,),
        )
        with pytest.raises(StateTransitionRejected, match=expected_field):
            reducer.commit(
                proposal(f"rebound_{port}", sensor.ref, rebound.ref)
            )

    evidence = plane.publish(
        workflow_id="place",
        activation_id="relation_verifier",
        attempt=1,
        port="verified",
        schema="robomex.relation_evidence.v1",
        payload=_relation_payload(sensor.artifact_id),
        lineage=(sensor.ref,),
    )
    with pytest.raises(StateTransitionRejected, match="evidence_refs"):
        reducer.commit(proposal("unbound_lineage", evidence.ref))
    state = reducer.commit(proposal("verified_relation", sensor.ref, evidence.ref))
    relation = state.relation("bowl_1", "on_top_of", "plate_1")
    assert relation is not None
    assert relation.value is RelationValue.ASSERTED
    assert relation.source_observation_revision == 7

    replayed = EmbodiedStateReducer(
        plane.episode_root,
        episode_id="ep_evidence",
        strict_evidence=True,
    )
    assert replayed.state == state
    assert replayed.events() == reducer.events()


def test_strict_verified_held_requires_matching_typed_attachment_evidence(
    tmp_path: Path,
) -> None:
    plane, sensor, reducer = _fixture(tmp_path, strict=True)
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_evidence",
            effect_id="attempt_grasp",
            before_revision=2,
            source="action_runtime",
            evidence_refs=(sensor.ref,),
            entity_id="bowl_1",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp_1",
            geometry_ref=sensor.ref,
        )
    )

    def proposal(effect_id: str, *evidence_refs):
        return StateTransitionProposal.set_attachment(
            episode_id="ep_evidence",
            effect_id=effect_id,
            before_revision=3,
            source="attachment_verifier",
            evidence_refs=evidence_refs,
            entity_id="bowl_1",
            status=AttachmentStatus.VERIFIED_HELD,
            observation_ref=sensor.ref,
            action_id="grasp_1",
            source_observation_id="obs_7",
            source_observation_revision=7,
            source_observation_domain="camera.front",
            track_id="track_bowl",
        )

    with pytest.raises(StateTransitionRejected, match="typed evidence schema"):
        reducer.commit(proposal("untyped_held", sensor.ref))

    payload = {
        "evidence_id": "attachment_7",
        "entity_id": "bowl_1",
        "entity_track_id": "track_bowl",
        "status": "verified_held",
        "source_observation_id": "obs_7",
        "source_observation_revision": 7,
        "observation_domain": "camera.front",
        "base_state_revision": 3,
        "action_id": "grasp_1",
        "evidence_refs": [sensor.artifact_id],
        "method": "finger_and_visual_guard",
        "reason": "finger closure and tracked-object motion agree",
        "confidence": 0.98,
        "validity": _validity(
            observation_id="obs_7",
            observation_revision=7,
            state_revision=3,
        ),
    }
    evidence = plane.publish(
        workflow_id="place",
        activation_id="attachment_verifier",
        attempt=1,
        port="verified",
        schema="robomex.attachment_evidence.v1",
        payload=payload,
        lineage=(sensor.ref,),
    )
    state = reducer.commit(proposal("verified_held", sensor.ref, evidence.ref))
    assert state.attachment.status is AttachmentStatus.VERIFIED_HELD
    assert state.attachment.source_observation_revision == 7

    stale_payload = {
        **payload,
        "evidence_id": "attachment_7_refresh",
        "base_state_revision": state.revision,
        "validity": _validity(
            observation_id="obs_7",
            observation_revision=7,
            state_revision=state.revision,
        ),
    }
    stale_evidence = plane.publish(
        workflow_id="place",
        activation_id="attachment_verifier",
        attempt=2,
        port="stale_refresh",
        schema="robomex.attachment_evidence.v1",
        payload=stale_payload,
        lineage=(sensor.ref,),
    )
    stale_refresh = StateTransitionProposal.set_attachment(
        episode_id="ep_evidence",
        effect_id="stale_verified_held_refresh",
        before_revision=state.revision,
        source="attachment_verifier",
        evidence_refs=(sensor.ref, stale_evidence.ref),
        entity_id="bowl_1",
        status=AttachmentStatus.VERIFIED_HELD,
        observation_ref=sensor.ref,
        action_id="grasp_1",
        source_observation_id="obs_7",
        source_observation_revision=7,
        source_observation_domain="camera.front",
        track_id="track_bowl",
    )
    with pytest.raises(StateTransitionRejected, match="new explicit observation_ref"):
        reducer.commit(stale_refresh)

    sensor_8 = plane.publish(
        workflow_id="place",
        activation_id="camera",
        attempt=2,
        port="frame",
        schema="test.camera_frame.v1",
        payload={"observation_id": "obs_8", "revision": 8},
    )
    refreshed_payload = {
        **payload,
        "evidence_id": "attachment_8",
        "source_observation_id": "obs_8",
        "source_observation_revision": 8,
        "base_state_revision": state.revision,
        "evidence_refs": [sensor_8.artifact_id],
        "validity": _validity(
            observation_id="obs_8",
            observation_revision=8,
            state_revision=state.revision,
        ),
    }
    refreshed_evidence = plane.publish(
        workflow_id="place",
        activation_id="attachment_verifier",
        attempt=2,
        port="refreshed",
        schema="robomex.attachment_evidence.v1",
        payload=refreshed_payload,
        lineage=(sensor_8.ref,),
    )
    refreshed = reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_evidence",
            effect_id="fresh_verified_held_refresh",
            before_revision=state.revision,
            source="attachment_verifier",
            evidence_refs=(sensor_8.ref, refreshed_evidence.ref),
            entity_id="bowl_1",
            status=AttachmentStatus.VERIFIED_HELD,
            observation_ref=sensor_8.ref,
            action_id="grasp_1",
            source_observation_id="obs_8",
            source_observation_revision=8,
            source_observation_domain="camera.front",
            track_id="track_bowl",
        )
    )
    assert refreshed.attachment.status is AttachmentStatus.VERIFIED_HELD
    assert refreshed.attachment.action_id == "grasp_1"
    assert refreshed.attachment.observation_ref == sensor_8.ref
    assert refreshed.attachment.source_observation_revision == 8


def test_strict_localization_requires_pose_bound_typed_evidence(tmp_path: Path) -> None:
    plane, sensor, reducer = _fixture(tmp_path, strict=True)
    pose = Pose(
        position_xyz_m=(0.5, 0.0, 0.1),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame_id="world",
        observation_id="obs_7",
    )

    def proposal(effect_id: str, *evidence_refs):
        return StateTransitionProposal.set_localization(
            episode_id="ep_evidence",
            effect_id=effect_id,
            before_revision=2,
            source="object_tracker",
            evidence_refs=evidence_refs,
            entity_id="bowl_1",
            status=LocalizationStatus.LOCALIZED,
            source_observation_id="obs_7",
            source_observation_revision=7,
            source_observation_domain="camera.front",
            world_pose=pose,
            track_id="track_bowl",
        )

    with pytest.raises(StateTransitionRejected, match="localization_evidence"):
        reducer.commit(proposal("untyped_localization", sensor.ref))

    evidence = plane.publish(
        workflow_id="place",
        activation_id="object_tracker",
        attempt=1,
        port="localization",
        schema="robomex.localization_evidence.v1",
        payload={
            "evidence_id": "localization_7",
            "entity_id": "bowl_1",
            "entity_track_id": "track_bowl",
            "status": "localized",
            "world_pose": pose.model_dump(mode="json"),
            "source_observation_id": "obs_7",
            "source_observation_revision": 7,
            "observation_domain": "camera.front",
            "base_state_revision": 2,
            "action_id": None,
            "evidence_refs": [sensor.artifact_id],
            "method": "multi_view_tracking",
            "reason": "the same registered track is visible in the current frame",
            "confidence": 0.97,
            "validity": _validity(
                observation_id="obs_7",
                observation_revision=7,
                state_revision=2,
            ),
        },
        lineage=(sensor.ref,),
    )
    state = reducer.commit(proposal("localized", sensor.ref, evidence.ref))
    localization = state.localization("bowl_1")
    assert localization is not None
    assert localization.status is LocalizationStatus.LOCALIZED
    assert localization.source_observation_revision == 7


def test_interruption_atomically_invalidates_attachment_pose_and_relations(
    tmp_path: Path,
) -> None:
    _, sensor, reducer = _fixture(tmp_path, strict=False)
    pose = Pose(
        position_xyz_m=(0.5, 0.0, 0.1),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame_id="world",
        observation_id="obs_7",
    )
    reducer.commit(
        StateTransitionProposal.set_localization(
            episode_id="ep_evidence",
            effect_id="localize_bowl",
            before_revision=2,
            source="tracker",
            evidence_refs=(sensor.ref,),
            entity_id="bowl_1",
            status=LocalizationStatus.LOCALIZED,
            source_observation_id="obs_7",
            world_pose=pose,
            track_id="track_bowl",
        )
    )
    for effect_id, subject, target, subject_track, target_track in (
        ("bowl_on_plate", "bowl_1", "plate_1", "track_bowl", "track_plate"),
        ("plate_under_bowl", "plate_1", "bowl_1", "track_plate", "track_bowl"),
    ):
        reducer.commit(
            StateTransitionProposal.set_relation(
                episode_id="ep_evidence",
                effect_id=effect_id,
                before_revision=reducer.state.revision,
                source="relation_verifier",
                evidence_refs=(sensor.ref,),
                subject_entity_id=subject,
                predicate=RelationPredicate.ON_TOP_OF,
                target_entity_id=target,
                value=RelationValue.ASSERTED,
                source_observation_id="obs_7",
                subject_track_id=subject_track,
                target_track_id=target_track,
            )
        )
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_evidence",
            effect_id="attempt_grasp",
            before_revision=reducer.state.revision,
            source="action_runtime",
            evidence_refs=(sensor.ref,),
            entity_id="bowl_1",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp_1",
            geometry_ref=sensor.ref,
        )
    )
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_evidence",
            effect_id="verify_grasp",
            before_revision=reducer.state.revision,
            source="attachment_verifier",
            evidence_refs=(sensor.ref,),
            entity_id="bowl_1",
            status=AttachmentStatus.VERIFIED_HELD,
            action_id="grasp_1",
        )
    )
    assert reducer.state.held_geometry_ref == sensor.ref
    before_revision = reducer.state.revision
    before_events = reducer.event_count
    state = reducer.commit(
        StateTransitionProposal.mark_interrupted(
            episode_id="ep_evidence",
            effect_id="transport_indeterminate",
            before_revision=before_revision,
            source="sealed_action_runner",
            evidence_refs=(sensor.ref,),
            entity_id="bowl_1",
            action_id="transport_1",
        )
    )

    assert state.revision == before_revision + 1
    assert reducer.event_count == before_events + 1
    assert state.attachment.status is AttachmentStatus.UNKNOWN
    assert state.held_geometry_ref is None
    assert state.attachment.geometry_ref is None
    localization = state.localization("bowl_1")
    assert localization is not None
    assert localization.status is LocalizationStatus.UNKNOWN
    assert localization.world_pose is None
    assert all(
        relation.value is RelationValue.UNKNOWN
        for relation in state.relations
        if "bowl_1" in {relation.subject_entity_id, relation.target_entity_id}
    )
    assert reducer.events()[-1]["proposal"]["trigger"] == (
        PhysicalStateTrigger.INTERRUPTION.value
    )
