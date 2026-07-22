"""Audited evidence-to-proposal adapter for the final support relation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from robomex.data import (
    RelationEvidence,
    RelationPredicate,
    RelationValue,
    ResolvedArtifactRef,
    StateTransitionProposal,
    StateTransitionProposalWire,
)
from robomex.manipulation import (
    BowlPlaceObservation,
    PlacementVerdict,
    PlacementVerdictStatus,
    RelationAssessment,
)


def _ref(value: Mapping[str, object]) -> ResolvedArtifactRef:
    return ResolvedArtifactRef.from_any(value)


def _proposal_refs(
    observation_input: Mapping[str, object],
    evidence_input: Mapping[str, object],
    verdict_input: Mapping[str, object],
) -> tuple[ResolvedArtifactRef, ...]:
    values = [
        _ref(observation_input["ref"]),
        _ref(evidence_input["ref"]),
        _ref(verdict_input["ref"]),
        *(_ref(item) for item in evidence_input.get("lineage", ())),
        *(_ref(item) for item in verdict_input.get("lineage", ())),
    ]
    unique: dict[tuple[str, str], ResolvedArtifactRef] = {}
    for value in values:
        unique[(value.artifact_id, value.content_digest)] = value
    return tuple(unique.values())


def build_relation_transition_proposal(
    observation_input: Mapping[str, object],
    evidence_input: Mapping[str, object],
    verdict_input: Mapping[str, object],
    *,
    episode_id: str,
) -> dict[str, object]:
    """Return a strict ``supported_by`` proposal after all final proofs agree."""

    observation = BowlPlaceObservation.model_validate(observation_input["payload"])
    evidence = RelationEvidence.model_validate(evidence_input["payload"])
    verdict = PlacementVerdict.model_validate(verdict_input["payload"])
    camera_revision = observation.revisions.camera.get(
        evidence.observation_domain.removeprefix("camera.")
    )
    if (
        evidence.predicate is not RelationPredicate.SUPPORTED_BY
        or evidence.value is not RelationValue.ASSERTED
        or evidence.subject_entity_id != observation.expected_bowl_entity_id
        or evidence.target_entity_id != observation.expected_target_entity_id
        or evidence.source_observation_id != observation.snapshot_id
        or evidence.source_observation_revision != camera_revision
    ):
        raise ValueError("relation evidence is not the admitted bowl-supported-by-plate proof")
    if (
        verdict.status is not PlacementVerdictStatus.SUCCEEDED
        or verdict.relation is not RelationAssessment.ASSERTED
        or verdict.bowl_entity_id != evidence.subject_entity_id
        or verdict.target_entity_id != evidence.target_entity_id
        or verdict.source_observation_id != evidence.source_observation_id
        or verdict.state_revision != evidence.base_state_revision
    ):
        raise ValueError("placement verdict and relation evidence do not agree")
    identity = f"{episode_id}|{evidence.evidence_id}|supported_by".encode()
    proposal = StateTransitionProposal.set_relation(
        episode_id=episode_id,
        effect_id="relation-" + hashlib.sha256(identity).hexdigest()[:24],
        before_revision=evidence.base_state_revision,
        source="bowl_relation_evidence_worker",
        evidence_refs=_proposal_refs(observation_input, evidence_input, verdict_input),
        subject_entity_id=evidence.subject_entity_id,
        predicate=evidence.predicate,
        target_entity_id=evidence.target_entity_id,
        value=evidence.value,
        source_observation_id=evidence.source_observation_id,
        source_observation_revision=evidence.source_observation_revision,
        source_observation_domain=evidence.observation_domain,
        observation_ref=_ref(observation_input["ref"]),
        action_id=evidence.action_id or "",
        subject_track_id=evidence.subject_track_id,
        target_track_id=evidence.target_track_id,
    )
    return StateTransitionProposalWire.from_domain(proposal).model_dump(mode="json")


__all__ = ["build_relation_transition_proposal"]
