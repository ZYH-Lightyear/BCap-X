"""Audited evidence-to-proposal adapter for attachment state."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from robomex.data import (
    AttachmentEvidence,
    AttachmentStatus,
    PhysicalStateTrigger,
    ResolvedArtifactRef,
    StateTransitionProposal,
    StateTransitionProposalWire,
)
from robomex.manipulation import BowlPlaceObservation


def _ref(value: Mapping[str, object]) -> ResolvedArtifactRef:
    return ResolvedArtifactRef.from_any(value)


def _proposal_refs(
    observation_input: Mapping[str, object],
    evidence_input: Mapping[str, object],
) -> tuple[ResolvedArtifactRef, ...]:
    values = [
        _ref(observation_input["ref"]),
        _ref(evidence_input["ref"]),
        *(_ref(item) for item in evidence_input.get("lineage", ())),
    ]
    unique: dict[tuple[str, str], ResolvedArtifactRef] = {}
    for value in values:
        unique[(value.artifact_id, value.content_digest)] = value
    return tuple(unique.values())


def build_attachment_transition_proposal(
    observation_input: Mapping[str, object],
    evidence_input: Mapping[str, object],
    *,
    episode_id: str,
) -> dict[str, object]:
    """Return a strict reducer proposal from one admitted evidence bundle."""

    observation = BowlPlaceObservation.model_validate(observation_input["payload"])
    evidence = AttachmentEvidence.model_validate(evidence_input["payload"])
    observation_ref = _ref(observation_input["ref"])
    camera_revision = observation.revisions.camera.get(
        evidence.observation_domain.removeprefix("camera.")
    )
    if (
        evidence.entity_id != observation.expected_bowl_entity_id
        or evidence.source_observation_id != observation.snapshot_id
        or evidence.source_observation_revision != camera_revision
        or evidence.status != observation.attachment_status.value
    ):
        raise ValueError("attachment evidence and observation identities do not agree")
    target = AttachmentStatus(evidence.status)
    if target not in {AttachmentStatus.VERIFIED_HELD, AttachmentStatus.NOT_HELD}:
        raise ValueError("only determinate evidence-backed attachment states may be proposed")
    identity = f"{episode_id}|{evidence.evidence_id}|{evidence.status}".encode()
    proposal = StateTransitionProposal.set_attachment(
        episode_id=episode_id,
        effect_id="attachment-" + hashlib.sha256(identity).hexdigest()[:24],
        before_revision=evidence.base_state_revision,
        source="bowl_attachment_evidence_worker",
        evidence_refs=_proposal_refs(observation_input, evidence_input),
        entity_id=evidence.entity_id,
        status=target,
        observation_ref=observation_ref,
        action_id=evidence.action_id or "",
        trigger=PhysicalStateTrigger.EVIDENCE,
        source_observation_id=evidence.source_observation_id,
        source_observation_revision=evidence.source_observation_revision,
        source_observation_domain=evidence.observation_domain,
        track_id=evidence.entity_track_id,
    )
    return StateTransitionProposalWire.from_domain(proposal).model_dump(mode="json")


__all__ = ["build_attachment_transition_proposal"]
