"""Audited synchronized bowl-to-plate alignment computation."""

from __future__ import annotations

from collections.abc import Mapping

from robomex.data import AttachmentEvidence, AttachmentStatus
from robomex.manipulation import (
    AlignmentTolerance,
    BowlPlaceObservation,
    VisibilityStatus,
    compute_alignment_error,
)


def estimate_support_alignment(
    observation: Mapping[str, object],
    attachment_evidence: Mapping[str, object],
    *,
    tolerance_xy_m: float = 0.006,
    tolerance_z_m: float = 0.006,
    tolerance_yaw_rad: float = 0.08,
) -> dict[str, dict[str, object]]:
    """Validate one checkpoint and return its geometry plus measured error."""

    observed = BowlPlaceObservation.model_validate(observation)
    evidence = AttachmentEvidence.model_validate(attachment_evidence)
    if (
        observed.held_visibility is not VisibilityStatus.VISIBLE
        or observed.target_visibility is not VisibilityStatus.VISIBLE
        or observed.held is None
        or observed.target is None
    ):
        raise ValueError("support alignment requires visible held-bowl and plate geometry")
    camera_revision = observed.revisions.camera.get(
        evidence.observation_domain.removeprefix("camera.")
    )
    if (
        evidence.status != AttachmentStatus.VERIFIED_HELD.value
        or evidence.entity_id != observed.expected_bowl_entity_id
        or evidence.source_observation_id != observed.snapshot_id
        or evidence.source_observation_revision != camera_revision
        or observed.attachment_status is not AttachmentStatus.VERIFIED_HELD
    ):
        raise ValueError("attachment evidence is not bound to the synchronized checkpoint")
    error = compute_alignment_error(
        observed.held,
        observed.target,
        tolerance=AlignmentTolerance(
            translation_xy_m=tolerance_xy_m,
            translation_z_m=tolerance_z_m,
            yaw_rad=tolerance_yaw_rad,
        ),
        expected_bowl_entity_id=observed.expected_bowl_entity_id,
        expected_target_entity_id=observed.expected_target_entity_id,
    )
    return {
        "held_bowl": observed.held.model_dump(mode="json"),
        "plate_target": observed.target.model_dump(mode="json"),
        "alignment_error": error.model_dump(mode="json"),
    }


__all__ = ["estimate_support_alignment"]
