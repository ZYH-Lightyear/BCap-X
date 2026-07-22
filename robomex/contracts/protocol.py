"""Typed skill invocation protocols and verifier obligations."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from robomex.contracts.common import (
    ContractId,
    NonEmptyStr,
    SealedContract,
    StrictContract,
    require_unique,
    require_unique_strings,
)
from robomex.contracts.effects import EffectContract
from robomex.runtime.events import ControlOutcome


class SlotCardinality(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    ONE = "one"
    ZERO_OR_ONE = "zero_or_one"
    MANY = "many"


class SlotSource(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    EPISODE_ARTIFACT = "episode_artifact"
    WORKFLOW_ARTIFACT = "workflow_artifact"
    SERVICE_SNAPSHOT = "service_snapshot"
    INVOCATION_LITERAL = "invocation_literal"


class SlotPolicy(StrictContract):
    """Admission policy for one typed input or output slot."""

    slot_id: ContractId
    schema_id: ContractId
    required: bool = True
    cardinality: SlotCardinality = SlotCardinality.ONE
    accepted_sources: tuple[SlotSource, ...] = (
        SlotSource.EPISODE_ARTIFACT,
        SlotSource.WORKFLOW_ARTIFACT,
    )
    pin_digest_at_admission: bool = True
    max_age_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    description: str = ""

    @model_validator(mode="after")
    def _check_cardinality(self) -> SlotPolicy:
        if self.required and self.cardinality is SlotCardinality.ZERO_OR_ONE:
            raise ValueError("A required slot cannot use zero_or_one cardinality.")
        if not self.accepted_sources:
            raise ValueError("accepted_sources must not be empty.")
        if len(self.accepted_sources) != len(set(self.accepted_sources)):
            raise ValueError("accepted_sources must not contain duplicates.")
        if (
            SlotSource.SERVICE_SNAPSHOT in self.accepted_sources
            and not self.pin_digest_at_admission
        ):
            raise ValueError(
                "A service snapshot must be digest-pinned at activation admission."
            )
        return self


class VerificationPhase(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    PRE_ADMISSION = "pre_admission"
    PRE_COMMIT = "pre_commit"
    CONTINUOUS = "continuous"
    POSTCONDITION = "postcondition"


class ObligationSeverity(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    ADVISORY = "advisory"
    HARD_GATE = "hard_gate"


class VerifierObligation(StrictContract):
    """Evidence-backed gate that a protocol runner must discharge."""

    obligation_id: ContractId
    phase: VerificationPhase
    verifier_ref: ContractId
    predicate: NonEmptyStr
    evidence_slots: tuple[ContractId, ...]
    severity: ObligationSeverity = ObligationSeverity.HARD_GATE
    applies_to_outcomes: tuple[ControlOutcome, ...] = (ControlOutcome.SUCCESS,)
    max_evidence_age_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    description: str = ""

    @model_validator(mode="after")
    def _check_obligation(self) -> VerifierObligation:
        if not self.evidence_slots:
            raise ValueError("A verifier obligation must name at least one evidence slot.")
        require_unique_strings(self.evidence_slots, label="verifier evidence slots")
        if not self.applies_to_outcomes:
            raise ValueError("applies_to_outcomes must not be empty.")
        if len(self.applies_to_outcomes) != len(set(self.applies_to_outcomes)):
            raise ValueError("applies_to_outcomes must not contain duplicates.")
        return self


class ProtocolSpec(SealedContract):
    """Versioned callable surface of a skill or deterministic service."""

    schema_version: Literal["robomex.protocol_spec.v1"] = "robomex.protocol_spec.v1"
    protocol_id: ContractId
    revision: int = Field(default=1, ge=1)
    summary: NonEmptyStr
    inputs: tuple[SlotPolicy, ...] = ()
    outputs: tuple[SlotPolicy, ...] = ()
    outcomes: tuple[ControlOutcome, ...] = (ControlOutcome.SUCCESS,)
    effect_contract: EffectContract
    verifier_obligations: tuple[VerifierObligation, ...] = ()
    required_capabilities: tuple[ContractId, ...] = ()
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_protocol(self) -> ProtocolSpec:
        require_unique(self.inputs, key="slot_id", label="input slot IDs")
        require_unique(self.outputs, key="slot_id", label="output slot IDs")
        input_ids = {slot.slot_id for slot in self.inputs}
        output_ids = {slot.slot_id for slot in self.outputs}
        overlap = input_ids & output_ids
        if overlap:
            raise ValueError(
                "Input and output slot IDs must be disjoint: "
                + ", ".join(sorted(overlap))
                + "."
            )
        if not self.outcomes:
            raise ValueError("A protocol must declare at least one outcome.")
        if len(self.outcomes) != len(set(self.outcomes)):
            raise ValueError("Protocol outcomes must not contain duplicates.")
        if ControlOutcome.SUCCESS not in self.outcomes:
            raise ValueError("A protocol must declare the canonical success outcome.")
        require_unique(
            self.verifier_obligations,
            key="obligation_id",
            label="verifier obligation IDs",
        )
        known_slots = input_ids | output_ids
        for obligation in self.verifier_obligations:
            unknown = set(obligation.evidence_slots) - known_slots
            if unknown:
                raise ValueError(
                    f"Verifier obligation {obligation.obligation_id!r} references "
                    f"unknown evidence slots: {', '.join(sorted(unknown))}."
                )
            undeclared = set(obligation.applies_to_outcomes) - set(self.outcomes)
            if undeclared:
                raise ValueError(
                    f"Verifier obligation {obligation.obligation_id!r} applies to "
                    "outcomes not declared by the protocol: "
                    + ", ".join(sorted(item.value for item in undeclared))
                    + "."
                )
        require_unique_strings(
            self.required_capabilities, label="required capability IDs"
        )
        return self


__all__ = [
    "ObligationSeverity",
    "ProtocolSpec",
    "SlotCardinality",
    "SlotPolicy",
    "SlotSource",
    "VerificationPhase",
    "VerifierObligation",
]
