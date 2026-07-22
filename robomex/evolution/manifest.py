"""Reproducible, frozen run manifests for baseline and future candidates."""

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
    StrictContract,
    require_unique,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class TaskSnapshot(SealedContract):
    schema_version: Literal["robomex.task_snapshot.v1"] = "robomex.task_snapshot.v1"
    task_id: ContractId
    instruction: NonEmptyStr
    success_rubric: NonEmptyStr
    dataset_id: ContractId | None = None
    episode_spec_digest: DigestStr
    metadata: JsonObject = Field(default_factory=dict)


class ModelPin(StrictContract):
    model_id: ContractId
    provider_id: ContractId
    weights_digest: DigestStr
    generation_config_digest: DigestStr
    tokenizer_digest: DigestStr | None = None


class PromptPin(StrictContract):
    prompt_id: ContractId
    content_digest: DigestStr


class SkillPin(StrictContract):
    skill_id: ContractId
    revision: int = Field(ge=1)
    manifest_digest: DigestStr


class FunctionPin(StrictContract):
    function_id: ContractId
    skill_id: ContractId
    implementation_digest: DigestStr
    interface_digest: DigestStr


class BackendRole(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    AUTHORITATIVE = "authoritative"
    SHADOW = "shadow"
    PERCEPTION = "perception"


class BackendPin(StrictContract):
    backend_id: ContractId
    role: BackendRole
    implementation_digest: DigestStr
    configuration_digest: DigestStr
    version: NonEmptyStr


class RunBudgets(StrictContract):
    max_model_calls: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    max_wall_time_s: float = Field(ge=0, allow_inf_nan=False)
    max_physical_actions: int = Field(ge=0)
    max_shadow_rollouts: int = Field(default=0, ge=0)
    max_candidates: int = Field(default=1, ge=1)
    max_recoveries: int = Field(default=0, ge=0)


class RunManifest(SealedContract):
    """Everything needed to attribute and reproduce one baseline/candidate run.

    The v2 baseline fixes ``mutation_policy`` to ``disabled``.  A future
    evolution controller may construct a new schema version, but this runtime
    manifest can never authorize or perform mutation by itself.
    """

    schema_version: Literal["robomex.run_manifest.v2"] = "robomex.run_manifest.v2"
    run_id: ContractId
    task: TaskSnapshot
    seed: int = Field(ge=0, le=9_223_372_036_854_775_807)
    model: ModelPin
    prompts: tuple[PromptPin, ...] = Field(min_length=1)
    skills: tuple[SkillPin, ...] = Field(min_length=1)
    functions: tuple[FunctionPin, ...] = Field(min_length=1)
    budgets: RunBudgets
    backends: tuple[BackendPin, ...] = Field(min_length=1)
    graph_digest: DigestStr
    contract_catalog_digest: DigestStr
    schema_registry_digest: DigestStr
    runtime_code_digest: DigestStr
    actor_profile_pins: tuple[ContentPin, ...] = ()
    candidate_config_digest: DigestStr | None = None
    mutation_policy: Literal["disabled"] = "disabled"
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @model_validator(mode="after")
    def _check_pins(self) -> RunManifest:
        require_unique(self.prompts, key="prompt_id", label="prompt pins")
        require_unique(self.skills, key="skill_id", label="skill pins")
        require_unique(self.functions, key="function_id", label="function pins")
        require_unique(self.backends, key="backend_id", label="backend pins")
        require_unique(
            self.actor_profile_pins, key="component_id", label="actor profile pins"
        )
        authoritative = [
            backend for backend in self.backends if backend.role is BackendRole.AUTHORITATIVE
        ]
        if len(authoritative) != 1:
            raise ValueError("A run manifest requires exactly one authoritative backend.")
        skill_ids = {skill.skill_id for skill in self.skills}
        unknown_functions = sorted(
            function.function_id
            for function in self.functions
            if function.skill_id not in skill_ids
        )
        if unknown_functions:
            raise ValueError(
                "Function pins reference skills absent from the run manifest: "
                + ", ".join(unknown_functions)
                + "."
            )
        return self


__all__ = [
    "BackendPin",
    "BackendRole",
    "FunctionPin",
    "ModelPin",
    "PromptPin",
    "RunBudgets",
    "RunManifest",
    "SkillPin",
    "TaskSnapshot",
]
