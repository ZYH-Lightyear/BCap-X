"""Versioned task-intent contracts for the RoboMEx v2 control plane.

An intent deliberately keeps the task instruction and success rubric as open
language.  The runtime should not turn every manipulation goal into a fixed
predicate template.  The surrounding references and budgets are nevertheless
typed so a Manager, evaluator, or future evolution layer can consume them
without scraping prose.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictContract(BaseModel):
    """Immutable, closed-world base for persisted orchestration contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class EntityRef(_StrictContract):
    """Reference to an episode entity, not a duplicate physical-state record."""

    entity_id: NonEmptyStr
    role: NonEmptyStr | None = None
    entity_type: NonEmptyStr | None = None
    description: str = ""
    state_ref: NonEmptyStr | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class BudgetHint(_StrictContract):
    """Optional ceilings used when opening one intent.

    These are hints to the Orchestrator rather than permission to exceed an
    episode-wide hard budget.  A missing field means "inherit policy", not
    "unbounded".
    """

    max_model_calls: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_wall_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_physical_actions: int | None = Field(default=None, ge=0)
    max_candidates: int | None = Field(default=None, ge=0)
    max_recoveries: int | None = Field(default=None, ge=0)


class SubgoalIntent(_StrictContract):
    """Open semantic objective presented to a v2 Swarm workflow."""

    schema_version: Literal["robomex.subgoal_intent.v1"] = "robomex.subgoal_intent.v1"
    intent_id: NonEmptyStr
    revision: int = Field(default=1, ge=1)
    instruction: NonEmptyStr
    success_rubric: NonEmptyStr
    entity_refs: tuple[EntityRef, ...] = ()
    protected_invariants: tuple[NonEmptyStr, ...] = ()
    evidence_types: tuple[NonEmptyStr, ...] = ()
    budget_hint: BudgetHint = Field(default_factory=BudgetHint)
    context: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("entity_refs", mode="before")
    @classmethod
    def _expand_entity_ids(cls, value: Any) -> Any:
        """Permit concise opaque IDs while retaining one canonical model."""

        if isinstance(value, (list, tuple)):
            return tuple({"entity_id": item} if isinstance(item, str) else item for item in value)
        return value


class IntentStatus(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """Closed terminal vocabulary for a v1 intent contract."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"
    NEEDS_REFINEMENT = "needs_refinement"


class IntentOutcome(_StrictContract):
    """Workflow-level result returned to the Episode Orchestrator."""

    schema_version: Literal["robomex.intent_outcome.v1"] = "robomex.intent_outcome.v1"
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    intent_id: NonEmptyStr
    intent_revision: int = Field(ge=1)
    status: IntentStatus
    summary: str = ""
    reason: str | None = None
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    final_state_revision: int | None = Field(default=None, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


__all__ = [
    "BudgetHint",
    "EntityRef",
    "IntentOutcome",
    "IntentStatus",
    "SubgoalIntent",
]
