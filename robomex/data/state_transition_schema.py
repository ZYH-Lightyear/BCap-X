"""Strict wire contracts for runtime-owned embodied-state commits.

The domain reducer deliberately consumes a frozen :class:`StateTransitionProposal`
rather than an open dictionary.  These Pydantic models are the artifact boundary
that turns an untrusted agent publication into that closed domain object.  Artifact
references are always the content-addressed pair; symbolic aliases, paths, and bare
artifact IDs have no representation in this protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.data.artifact_resolver import ResolvedArtifactRef
from robomex.data.embodied_state import (
    AttachmentStatus,
    LocalizationStatus,
    PhysicalStateTrigger,
    StateTransitionKind,
    StateTransitionProposal,
)
from robomex.data.physical_schema import Pose, RelationPredicate, RelationValue

_DOMAIN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_NON_EMPTY = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_DOMAIN_ID = Annotated[str, Field(pattern=_DOMAIN_ID_PATTERN)]


class _StrictWireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class StateArtifactRef(_StrictWireModel):
    """The only artifact-reference spelling admitted by state protocols."""

    artifact_id: _NON_EMPTY
    content_digest: str = Field(pattern=_DIGEST_PATTERN)

    def to_domain(self) -> ResolvedArtifactRef:
        return ResolvedArtifactRef(
            artifact_id=self.artifact_id,
            content_digest=self.content_digest,
        )

    @classmethod
    def from_domain(cls, ref: ResolvedArtifactRef) -> StateArtifactRef:
        if not isinstance(ref, ResolvedArtifactRef):
            raise TypeError("state artifact references must already be admission-resolved")
        return cls.model_validate(ref.to_mapping())


class StateTransitionProposalWire(_StrictWireModel):
    """Exact JSON representation of :class:`StateTransitionProposal`."""

    episode_id: _DOMAIN_ID
    effect_id: _DOMAIN_ID
    before_revision: int = Field(ge=0)
    kind: StateTransitionKind
    source: _NON_EMPTY
    evidence_refs: tuple[StateArtifactRef, ...] = Field(min_length=1)
    entity_id: _DOMAIN_ID
    semantic_label: str = ""
    track_id: str = ""
    attachment_status: AttachmentStatus | None = None
    observation_ref: StateArtifactRef | None = None
    geometry_ref: StateArtifactRef | None = None
    action_id: str = ""
    trigger: PhysicalStateTrigger = PhysicalStateTrigger.EVIDENCE
    localization_status: LocalizationStatus | None = None
    world_pose: Pose | None = None
    source_observation_id: str = ""
    source_observation_revision: int | None = Field(default=None, ge=0)
    source_observation_domain: str = ""
    target_entity_id: str = ""
    relation_predicate: RelationPredicate | None = None
    relation_value: RelationValue | None = None
    subject_track_id: str = ""
    target_track_id: str = ""

    @field_validator("before_revision", "source_observation_revision", mode="before")
    @classmethod
    def _reject_boolean_revisions(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("state revisions must be integers, not booleans")
        return value

    @field_validator(
        "track_id",
        "action_id",
        "source_observation_id",
        "target_entity_id",
        "subject_track_id",
        "target_track_id",
    )
    @classmethod
    def _optional_domain_ids(cls, value: str) -> str:
        if value and re.fullmatch(_DOMAIN_ID_PATTERN, value) is None:
            raise ValueError("non-empty state identity fields must use the closed ID grammar")
        return value

    @model_validator(mode="after")
    def _closed_reference_set(self) -> StateTransitionProposalWire:
        identities = [
            (ref.artifact_id, ref.content_digest) for ref in self.evidence_refs
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("evidence_refs cannot contain duplicates")
        evidence = set(identities)
        for name, ref in (
            ("observation_ref", self.observation_ref),
            ("geometry_ref", self.geometry_ref),
        ):
            if ref is not None and (ref.artifact_id, ref.content_digest) not in evidence:
                raise ValueError(f"{name} must also appear in evidence_refs")
        return self

    def to_domain(self) -> StateTransitionProposal:
        """Create the frozen reducer input without accepting any loose fields."""

        return StateTransitionProposal(
            episode_id=self.episode_id,
            effect_id=self.effect_id,
            before_revision=self.before_revision,
            kind=self.kind,
            source=self.source,
            evidence_refs=tuple(ref.to_domain() for ref in self.evidence_refs),
            entity_id=self.entity_id,
            semantic_label=self.semantic_label,
            track_id=self.track_id,
            attachment_status=self.attachment_status,
            observation_ref=(
                self.observation_ref.to_domain()
                if self.observation_ref is not None
                else None
            ),
            geometry_ref=(
                self.geometry_ref.to_domain() if self.geometry_ref is not None else None
            ),
            action_id=self.action_id,
            trigger=self.trigger,
            localization_status=self.localization_status,
            world_pose=self.world_pose,
            source_observation_id=self.source_observation_id,
            source_observation_revision=self.source_observation_revision,
            source_observation_domain=self.source_observation_domain,
            target_entity_id=self.target_entity_id,
            relation_predicate=self.relation_predicate,
            relation_value=self.relation_value,
            subject_track_id=self.subject_track_id,
            target_track_id=self.target_track_id,
        )

    @classmethod
    def from_domain(
        cls, proposal: StateTransitionProposal
    ) -> StateTransitionProposalWire:
        if not isinstance(proposal, StateTransitionProposal):
            raise TypeError("proposal must be a StateTransitionProposal")
        return cls.model_validate(proposal.to_mapping())

    @property
    def proposal_digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class StateCommitReceipt(_StrictWireModel):
    """Stable proof that the sole reducer durably committed one proposal."""

    schema_version: Literal["robomex.state_commit_receipt.v1"] = (
        "robomex.state_commit_receipt.v1"
    )
    episode_id: _DOMAIN_ID
    effect_id: _DOMAIN_ID
    proposal_ref: StateArtifactRef
    proposal_digest: str = Field(pattern=_DIGEST_PATTERN)
    before_revision: int = Field(ge=0)
    after_revision: int = Field(ge=1)
    state_event_id: _NON_EMPTY
    state_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("before_revision", "after_revision", mode="before")
    @classmethod
    def _reject_boolean_revisions(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("state revisions must be integers, not booleans")
        return value

    @model_validator(mode="after")
    def _one_revision_commit(self) -> StateCommitReceipt:
        if self.after_revision != self.before_revision + 1:
            raise ValueError("a state commit receipt must advance exactly one revision")
        return self


STATE_TRANSITION_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "robomex.state_transition_proposal.v1": StateTransitionProposalWire,
    "robomex.state_commit_receipt.v1": StateCommitReceipt,
}


__all__ = [
    "STATE_TRANSITION_SCHEMA_MODELS",
    "StateArtifactRef",
    "StateCommitReceipt",
    "StateTransitionProposalWire",
]
