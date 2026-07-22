from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.data import (
    Affordance,
    AttachmentStatus,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    FreshnessRejected,
    LocalizationStatus,
    ObjectGeometry,
    PhysicalStateTrigger,
    Point3,
    Pose,
    RelationPredicate,
    RelationValue,
    RevisionVector,
    SchemaPayloadError,
    SchemaRegistry,
    StateTransitionProposal,
    StateTransitionRejected,
    UnregisteredSchemaError,
    ValidityLifecycle,
    ValidityVector,
)


def _validity(
    *,
    observation_id: str = "obs_1",
    revisions: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "lifecycle": "derived",
        "depends_on_revisions": revisions or {"scene": 4, "camera.front": 2},
        "observation_id": observation_id,
        "method": "runtime_revision_clock",
        "reason": "all action-facing dependencies are explicit",
        "confidence": 1.0,
    }


def _geometry_payload() -> dict[str, object]:
    return {
        "entity_id": "bowl_1",
        "semantic_label": "bowl",
        "track_id": "track_bowl",
        "source_observation_id": "obs_1",
        "obb": {
            "center": {
                "xyz_m": [0.42, -0.1, 0.08],
                "frame_id": "world",
                "observation_id": "obs_1",
            },
            "full_extents_m": [0.14, 0.14, 0.07],
            "rotation_world_from_obb": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "fit_residual_m": 0.003,
        },
        "method": "robust_pca_obb",
        "reason": "cylinder fit was not selected because the rim is partially occluded",
        "confidence": 0.88,
        "validity": _validity(),
    }


def test_registry_has_real_core_validators_and_rejects_unknown_schema() -> None:
    registry = SchemaRegistry()
    parsed = registry.validate("robomex.object_geometry.v2", _geometry_payload())
    assert isinstance(parsed, ObjectGeometry)

    with pytest.raises(UnregisteredSchemaError, match="untyped fallback"):
        registry.validate("robomex.unregistered_core.v1", {})
    with pytest.raises(UnregisteredSchemaError, match="not registered"):
        SchemaRegistry(install_core=False).validate(
            "robomex.object_geometry.v2",
            _geometry_payload(),
        )
    with pytest.raises(SchemaPayloadError, match="failed schema"):
        registry.validate("robomex.object_geometry.v2", {"obb": {}})


def test_object_geometry_rejects_flat_nested_obb_and_lineage_mismatch() -> None:
    payload = _geometry_payload()
    payload["center_xyz_m"] = [9.0, 9.0, 9.0]
    with pytest.raises(ValidationError, match="Flat OBB fields"):
        ObjectGeometry.model_validate(payload)

    payload = _geometry_payload()
    obb = dict(payload["obb"])  # type: ignore[arg-type]
    center = dict(obb["center"])  # type: ignore[arg-type]
    center["observation_id"] = "obs_old"
    obb["center"] = center
    payload["obb"] = obb
    with pytest.raises(ValidationError, match="source observation"):
        ObjectGeometry.model_validate(payload)

    payload = _geometry_payload()
    obb = dict(payload["obb"])  # type: ignore[arg-type]
    obb["rotation_world_from_obb"] = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    payload["obb"] = obb
    with pytest.raises(ValidationError, match="orthogonal"):
        ObjectGeometry.model_validate(payload)


def test_position_cannot_enter_direction_port_and_quaternion_is_explicit() -> None:
    point = Point3(xyz_m=(0.0, 0.0, 1.0), frame_id="world", observation_id="obs_1")
    payload = {
        "affordance_id": "grasp_1",
        "action_type": "grasp",
        "object_entity_id": "bowl_1",
        "target_pose": {
            "position_xyz_m": [0.4, 0.0, 0.1],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "frame_id": "world",
            "observation_id": "obs_1",
        },
        "approach": {"direction": point.model_dump(), "distance_m": 0.08},
        "strategy": "rim_grasp",
        "source_geometry_id": "geometry_1",
        "selection": {
            "status": "degraded",
            "method": "rim_clearance_search",
            "reason": "top surface is occluded",
            "confidence": 0.72,
        },
        "validity": _validity(),
    }
    with pytest.raises(ValidationError, match="xyz_unit"):
        Affordance.model_validate(payload)

    with pytest.raises(ValidationError, match="unit length"):
        Pose(
            position_xyz_m=(0.0, 0.0, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 1.0),
            frame_id="world",
        )
    with pytest.raises(ValidationError, match="wxyz"):
        Pose.model_validate(
            {
                "position_xyz_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                "quaternion_convention": "xyzw",
                "frame_id": "world",
            }
        )


def test_validity_vector_checks_each_domain_and_never_falls_back() -> None:
    validity = ValidityVector(
        lifecycle=ValidityLifecycle.DERIVED,
        depends_on_revisions={"scene": 4, "attachment": 2, "camera.front": 7},
        observation_id="obs_8",
        method="causal_revision_match",
        reason="geometry is valid only in its admitted scene and camera state",
        confidence=0.93,
    )
    current = RevisionVector(
        scene=4,
        arm=10,
        gripper=3,
        attachment=2,
        camera={"front": 7},
    )
    decision = validity.assert_admissible(current, current_observation_id="obs_8")
    assert decision.checked_domains == ("attachment", "camera.front", "scene")

    stale_scene = current.model_copy(update={"scene": 5})
    with pytest.raises(FreshnessRejected, match="Stale 'scene' revision"):
        validity.assert_admissible(stale_scene, current_observation_id="obs_8")
    with pytest.raises(FreshnessRejected, match="Stale observation"):
        validity.assert_admissible(current, current_observation_id="obs_9")
    missing_camera = current.model_copy(update={"camera": {}})
    with pytest.raises(FreshnessRejected, match="no camera domain"):
        validity.assert_admissible(missing_camera, current_observation_id="obs_8")


def _state_fixture(tmp_path: Path):
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="ep_relation")
    plane.open_workflow("place")
    evidence = plane.publish(
        workflow_id="place",
        activation_id="verify",
        attempt=1,
        port="evidence",
        schema="robomex.test_evidence.v1",
        payload={"observation_id": "obs_place"},
    )
    reducer = EmbodiedStateReducer(
        plane.episode_root,
        episode_id="ep_relation",
        resolver=plane.resolver,
    )
    for entity_id, label, track_id in (
        ("bowl_1", "bowl", "track_bowl"),
        ("plate_1", "plate", "track_plate"),
    ):
        reducer.commit(
            StateTransitionProposal.register_entity(
                episode_id="ep_relation",
                effect_id=f"register_{entity_id}",
                before_revision=reducer.state.revision,
                source="grounding_agent",
                evidence_refs=(evidence.ref,),
                entity_id=entity_id,
                semantic_label=label,
                track_id=track_id,
            )
        )
    return plane, evidence, reducer


def test_localization_relation_commit_identity_guards_and_replay(tmp_path: Path) -> None:
    plane, evidence, reducer = _state_fixture(tmp_path)
    localized = reducer.commit(
        StateTransitionProposal.set_localization(
            episode_id="ep_relation",
            effect_id="localize_bowl",
            before_revision=2,
            source="learned_verifier",
            evidence_refs=(evidence.ref,),
            entity_id="bowl_1",
            status=LocalizationStatus.LOCALIZED,
            source_observation_id="obs_place",
            world_pose=Pose(
                position_xyz_m=(0.5, 0.0, 0.04),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                frame_id="world",
                observation_id="obs_place",
            ),
            observation_ref=evidence.ref,
            track_id="track_bowl",
        )
    )
    assert localized.localization("bowl_1").status is LocalizationStatus.LOCALIZED  # type: ignore[union-attr]

    related = reducer.commit(
        StateTransitionProposal.set_relation(
            episode_id="ep_relation",
            effect_id="verify_on_plate",
            before_revision=3,
            source="learned_verifier",
            evidence_refs=(evidence.ref,),
            subject_entity_id="bowl_1",
            predicate=RelationPredicate.ON_TOP_OF,
            target_entity_id="plate_1",
            value=RelationValue.ASSERTED,
            source_observation_id="obs_place",
            observation_ref=evidence.ref,
            action_id="place_7",
            subject_track_id="track_bowl",
            target_track_id="track_plate",
        )
    )
    relation = related.relation("bowl_1", RelationPredicate.ON_TOP_OF, "plate_1")
    assert relation is not None and relation.value is RelationValue.ASSERTED

    event_count = reducer.event_count
    with pytest.raises(StateTransitionRejected, match="Wrong relation target identity"):
        reducer.commit(
            StateTransitionProposal.set_relation(
                episode_id="ep_relation",
                effect_id="wrong_target_track",
                before_revision=4,
                source="learned_verifier",
                evidence_refs=(evidence.ref,),
                subject_entity_id="bowl_1",
                predicate=RelationPredicate.INSIDE,
                target_entity_id="plate_1",
                value=RelationValue.NEGATED,
                source_observation_id="obs_place",
                target_track_id="some_other_plate",
            )
        )
    assert reducer.event_count == event_count

    with pytest.raises(StateTransitionRejected, match="Stale"):
        reducer.commit(
            StateTransitionProposal.set_relation(
                episode_id="ep_relation",
                effect_id="stale_relation",
                before_revision=3,
                source="learned_verifier",
                evidence_refs=(evidence.ref,),
                subject_entity_id="bowl_1",
                predicate=RelationPredicate.SUPPORTED_BY,
                target_entity_id="plate_1",
                value=RelationValue.ASSERTED,
                source_observation_id="obs_place",
            )
        )

    expected = reducer.state
    reducer.index_path.unlink()
    replayed = EmbodiedStateReducer(plane.episode_root, episode_id="ep_relation")
    assert replayed.state == expected
    replayed_relation = replayed.state.relation("bowl_1", "on_top_of", "plate_1")
    assert replayed_relation is not None and replayed_relation.value is RelationValue.ASSERTED


def test_open_and_interruption_can_only_commit_conservative_unknown(tmp_path: Path) -> None:
    _, evidence, reducer = _state_fixture(tmp_path)
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_relation",
            effect_id="attempt",
            before_revision=2,
            source="action_runtime",
            evidence_refs=(evidence.ref,),
            entity_id="bowl_1",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp_1",
        )
    )
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="ep_relation",
            effect_id="held",
            before_revision=3,
            source="learned_verifier",
            evidence_refs=(evidence.ref,),
            entity_id="bowl_1",
            status=AttachmentStatus.VERIFIED_HELD,
            action_id="grasp_1",
        )
    )
    opened = reducer.commit(
        StateTransitionProposal.mark_open_admitted(
            episode_id="ep_relation",
            effect_id="open_admitted",
            before_revision=4,
            source="action_runtime",
            evidence_refs=(evidence.ref,),
            entity_id="bowl_1",
            action_id="open_1",
        )
    )
    assert opened.attachment.status is AttachmentStatus.UNKNOWN

    with pytest.raises(StateTransitionRejected, match="conservatively"):
        reducer.commit(
            StateTransitionProposal.set_attachment(
                episode_id="ep_relation",
                effect_id="interruption_claims_detached",
                before_revision=5,
                source="action_runtime",
                evidence_refs=(evidence.ref,),
                entity_id="bowl_1",
                status=AttachmentStatus.NOT_HELD,
                trigger=PhysicalStateTrigger.INTERRUPTION,
            )
        )
