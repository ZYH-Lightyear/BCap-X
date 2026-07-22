"""Declarative preconditions and effects shared by skills and protocols."""

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
)


class EffectScope(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    """Authority domain in which an effect may occur."""

    READ_ONLY = "read_only"
    SHADOW_WORLD = "shadow_world"
    AUTHORITATIVE_WORLD = "authoritative_world"


class Reversibility(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    REVERSIBLE = "reversible"
    COMPENSATABLE = "compensatable"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class StatePredicate(StrictContract):
    """Named state fact; ``expression`` is evaluated only by its declared verifier."""

    predicate_id: ContractId
    expression: NonEmptyStr
    schema_id: ContractId | None = None
    description: str = ""


class DeclaredEffect(StrictContract):
    """One bounded effect capability, not an instruction to execute it."""

    effect_id: ContractId
    scope: EffectScope
    operation: NonEmptyStr
    resource_selector: NonEmptyStr | None = None
    reversibility: Reversibility = Reversibility.UNKNOWN
    compensation_protocol_id: ContractId | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check_authority_shape(self) -> DeclaredEffect:
        if self.scope is EffectScope.READ_ONLY and self.resource_selector is not None:
            raise ValueError("A read_only effect cannot declare a writable resource.")
        if self.scope is not EffectScope.READ_ONLY and self.resource_selector is None:
            raise ValueError("A world effect must declare resource_selector.")
        if self.reversibility is Reversibility.COMPENSATABLE:
            if self.compensation_protocol_id is None:
                raise ValueError(
                    "A compensatable effect must name compensation_protocol_id."
                )
        elif self.compensation_protocol_id is not None:
            raise ValueError(
                "compensation_protocol_id is valid only for compensatable effects."
            )
        return self


class EffectContract(SealedContract):
    """Machine-authoritative state transition contract.

    ``requires`` are admission conditions.  ``invalidates`` and ``establishes``
    describe facts whose truth may change after successful completion.  The
    separate ``effects`` list bounds actual read/shadow/authoritative effects.
    """

    schema_version: Literal["robomex.effect_contract.v1"] = (
        "robomex.effect_contract.v1"
    )
    contract_id: ContractId
    revision: int = Field(default=1, ge=1)
    requires: tuple[StatePredicate, ...] = ()
    effects: tuple[DeclaredEffect, ...] = ()
    invalidates: tuple[StatePredicate, ...] = ()
    establishes: tuple[StatePredicate, ...] = ()

    @model_validator(mode="after")
    def _check_facts_and_effects(self) -> EffectContract:
        require_unique(self.requires, key="predicate_id", label="required predicate IDs")
        require_unique(
            self.invalidates, key="predicate_id", label="invalidated predicate IDs"
        )
        require_unique(
            self.establishes, key="predicate_id", label="established predicate IDs"
        )
        require_unique(self.effects, key="effect_id", label="effect IDs")
        invalidated = {item.predicate_id for item in self.invalidates}
        established = {item.predicate_id for item in self.establishes}
        overlap = invalidated & established
        if overlap:
            raise ValueError(
                "A predicate cannot be both invalidated and established in one contract: "
                + ", ".join(sorted(overlap))
                + "."
            )
        return self

    @property
    def permits_authoritative_effects(self) -> bool:
        return any(
            effect.scope is EffectScope.AUTHORITATIVE_WORLD for effect in self.effects
        )


__all__ = [
    "DeclaredEffect",
    "EffectContract",
    "EffectScope",
    "Reversibility",
    "StatePredicate",
]
