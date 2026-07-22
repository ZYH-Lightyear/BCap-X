"""Immutable component, safety-boundary, and candidate snapshots.

These schemas make a future evolution loop auditable; they intentionally do
not contain any mutation, search, or automatic-promotion algorithm.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from robomex.contracts.common import (
    ContentPin,
    ContractId,
    DigestStr,
    JsonObject,
    NonEmptyStr,
    SealedContract,
    require_unique,
    require_unique_strings,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


def _validate_pointer(path: str) -> str:
    if not path.startswith("/") or path == "/":
        raise ValueError("Mutable/protected paths must be non-root JSON pointers.")
    if "//" in path:
        raise ValueError("JSON pointer paths must not contain empty segments.")
    return path


class EvolvableComponentKind(str, Enum):  # noqa: UP042 - Python 3.10 support
    PROMPT = "prompt"
    SKILL_MANIFEST = "skill_manifest"
    SKILL_CODE = "skill_code"
    ACTOR_PROFILE = "actor_profile"
    MANAGER_POLICY = "manager_policy"
    GRAPH_TEMPLATE = "graph_template"
    VERIFIER = "verifier"
    EVALUATOR = "evaluator"


class PromotionStage(str, Enum):  # noqa: UP042 - Python 3.10 support
    BASELINE = "baseline"
    DRAFT = "draft"
    OFFLINE_EVALUATED = "offline_evaluated"
    SHADOW_EVALUATED = "shadow_evaluated"
    QUALIFIED = "qualified"
    APPROVED = "approved"
    REJECTED = "rejected"


class SafetyBoundary(SealedContract):
    """Immutable policy surface that candidates may reference but never edit."""

    schema_version: Literal["robomex.safety_boundary.v1"] = (
        "robomex.safety_boundary.v1"
    )
    boundary_id: ContractId
    revision: int = Field(default=1, ge=1)
    description: NonEmptyStr
    protected_paths: tuple[NonEmptyStr, ...] = Field(min_length=1)
    required_verifier_ids: tuple[ContractId, ...] = Field(min_length=1)
    authoritative_effects_require_approval: Literal[True] = True
    immutable: Literal[True] = True

    @field_validator("protected_paths")
    @classmethod
    def _valid_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_pointer(path)
        require_unique_strings(value, label="protected safety paths")
        return value

    @field_validator("required_verifier_ids")
    @classmethod
    def _unique_verifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique_strings(value, label="required safety verifier IDs")
        return value


class EvolvableComponentSpec(SealedContract):
    """Registry declaration of exactly which fields may vary in candidates."""

    schema_version: Literal["robomex.evolvable_component.v1"] = (
        "robomex.evolvable_component.v1"
    )
    component_id: ContractId
    kind: EvolvableComponentKind
    base_content_digest: DigestStr
    mutable_paths: tuple[NonEmptyStr, ...] = ()
    safety_boundary_ids: tuple[ContractId, ...] = Field(min_length=1)
    description: str = ""

    @field_validator("mutable_paths")
    @classmethod
    def _valid_mutable_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_pointer(path)
        require_unique_strings(value, label="mutable component paths")
        return value

    @field_validator("safety_boundary_ids")
    @classmethod
    def _unique_boundaries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique_strings(value, label="component safety boundaries")
        return value


class CandidateComponentSnapshot(SealedContract):
    """Frozen candidate content and its declared diff from one registry base."""

    schema_version: Literal["robomex.candidate_component.v1"] = (
        "robomex.candidate_component.v1"
    )
    component_id: ContractId
    base_content_digest: DigestStr
    candidate_content_digest: DigestStr
    changed_paths: tuple[NonEmptyStr, ...] = ()
    configuration: JsonObject = Field(default_factory=dict)

    @field_validator("changed_paths")
    @classmethod
    def _valid_changed_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            _validate_pointer(path)
        require_unique_strings(value, label="candidate changed paths")
        return value

    @model_validator(mode="after")
    def _match_change_marker(self) -> CandidateComponentSnapshot:
        changed_digest = self.candidate_content_digest != self.base_content_digest
        if changed_digest != bool(self.changed_paths):
            raise ValueError(
                "candidate/base digest difference and changed_paths must agree."
            )
        return self


class CandidateConfigSnapshot(SealedContract):
    """Immutable complete candidate; promotion creates a new revision/snapshot."""

    schema_version: Literal["robomex.candidate_config.v1"] = (
        "robomex.candidate_config.v1"
    )
    candidate_id: ContractId
    revision: int = Field(default=1, ge=1)
    parent_config_digest: DigestStr | None = None
    stage: PromotionStage = PromotionStage.BASELINE
    components: tuple[CandidateComponentSnapshot, ...] = Field(min_length=1)
    safety_boundary_pins: tuple[ContentPin, ...] = Field(min_length=1)
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    created_by: NonEmptyStr
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @model_validator(mode="after")
    def _check_candidate(self) -> CandidateConfigSnapshot:
        require_unique(self.components, key="component_id", label="candidate components")
        require_unique(
            self.safety_boundary_pins,
            key="component_id",
            label="candidate safety-boundary pins",
        )
        require_unique_strings(self.evidence_refs, label="candidate evidence references")
        changed = [
            component
            for component in self.components
            if component.candidate_content_digest != component.base_content_digest
        ]
        if self.stage is PromotionStage.BASELINE:
            if changed:
                raise ValueError("A baseline candidate cannot contain mutated components.")
            if self.parent_config_digest is not None:
                raise ValueError("A baseline candidate cannot have a parent candidate.")
        elif self.parent_config_digest is None:
            raise ValueError("A non-baseline candidate must pin its parent configuration.")
        if (
            self.stage not in {PromotionStage.BASELINE, PromotionStage.DRAFT}
            and not self.evidence_refs
        ):
            raise ValueError("An evaluated/promoted candidate requires evidence_refs.")
        return self


_ALLOWED_PROMOTIONS: dict[PromotionStage, frozenset[PromotionStage]] = {
    PromotionStage.BASELINE: frozenset({PromotionStage.DRAFT}),
    PromotionStage.DRAFT: frozenset(
        {PromotionStage.OFFLINE_EVALUATED, PromotionStage.REJECTED}
    ),
    PromotionStage.OFFLINE_EVALUATED: frozenset(
        {PromotionStage.SHADOW_EVALUATED, PromotionStage.REJECTED}
    ),
    PromotionStage.SHADOW_EVALUATED: frozenset(
        {PromotionStage.QUALIFIED, PromotionStage.REJECTED}
    ),
    PromotionStage.QUALIFIED: frozenset(
        {PromotionStage.APPROVED, PromotionStage.REJECTED}
    ),
    PromotionStage.APPROVED: frozenset(),
    PromotionStage.REJECTED: frozenset(),
}


def is_promotion_transition_allowed(
    from_stage: PromotionStage, to_stage: PromotionStage
) -> bool:
    """Return whether an auditable stage transition is forward and legal."""

    return to_stage in _ALLOWED_PROMOTIONS[from_stage]


class PromotionRecord(SealedContract):
    """Auditable stage decision; it never deploys a candidate."""

    schema_version: Literal["robomex.promotion_record.v1"] = (
        "robomex.promotion_record.v1"
    )
    candidate_id: ContractId
    candidate_config_digest: DigestStr
    from_stage: PromotionStage
    to_stage: PromotionStage
    decision_by: NonEmptyStr
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reason: NonEmptyStr
    decided_at: datetime = Field(default_factory=_utc_now)

    @field_validator("decided_at")
    @classmethod
    def _decision_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @model_validator(mode="after")
    def _legal_transition(self) -> PromotionRecord:
        if not is_promotion_transition_allowed(self.from_stage, self.to_stage):
            raise ValueError(
                f"Illegal promotion transition {self.from_stage.value!r} -> "
                f"{self.to_stage.value!r}."
            )
        require_unique_strings(self.evidence_refs, label="promotion evidence references")
        return self


__all__ = [
    "CandidateComponentSnapshot",
    "CandidateConfigSnapshot",
    "EvolvableComponentKind",
    "EvolvableComponentSpec",
    "PromotionRecord",
    "PromotionStage",
    "SafetyBoundary",
    "is_promotion_transition_allowed",
]
