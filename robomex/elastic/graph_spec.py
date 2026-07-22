"""Strict schema for a fixed or elastic RoboMEx v2 workflow.

This module deliberately models activations rather than a single executor
cursor.  A workflow has one primary control token, while read-only services
may remain active beside it.  Structural patching is added later; the same
schema already carries graph identity and revision.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.runtime.events import ControlOutcome

_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_SCHEMA_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,191}$"
ACTION_SNAPSHOT_RUNNER_REF = "robomex.runtime.capture_action_snapshot"
ACTION_SNAPSHOT_SCHEMA_ID = "robomex.admission_snapshot.v1"


class RunnerKind(str, Enum):  # noqa: UP042 - package supports Python 3.10
    CODING_WORKER = "coding_worker"
    DETERMINISTIC_GATE = "deterministic_gate"
    ARENA = "arena"
    MONITOR_PROGRAM = "monitor_program"
    TRACKING_SERVICE = "tracking_service"
    SYSTEM_ACTION = "system_action"
    ACTION_SNAPSHOT = "action_snapshot"
    REDUCER = "reducer"


class ActivationLane(str, Enum):  # noqa: UP042 - package supports Python 3.10
    PRIMARY = "primary"
    SERVICE = "service"


class LifecycleScope(str, Enum):  # noqa: UP042 - package supports Python 3.10
    INVOCATION = "invocation"
    WORKFLOW = "workflow"
    EPISODE = "episode"


class EffectScope(str, Enum):  # noqa: UP042 - package supports Python 3.10
    READ_ONLY = "read_only"
    SHADOW_WORLD = "shadow_world"
    AUTHORITATIVE_WORLD = "authoritative_world"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionBudget(_StrictModel):
    """Auditable upper-bound estimate attached to one activation.

    These units are deliberately provider-neutral.  A graph patch aggregates
    them before it is compiled and may therefore reject an otherwise valid
    fragment without invoking an actor or spending physical actions.
    """

    model_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    wall_time_ms: int = Field(default=0, ge=0)
    shadow_rollouts: int = Field(default=0, ge=0)
    authoritative_actions: int = Field(default=0, ge=0)
    actor_spawns: int = Field(default=0, ge=0)


class PortSpecV2(_StrictModel):
    name: str = Field(pattern=_ID_PATTERN)
    schema_id: str = Field(pattern=_SCHEMA_PATTERN)
    required: bool = True


class ArtifactBinding(_StrictModel):
    kind: Literal["artifact"] = "artifact"
    input_port: str = Field(pattern=_ID_PATTERN)
    source_activation: str = Field(pattern=_ID_PATTERN)
    source_port: str = Field(pattern=_ID_PATTERN)


class ExternalBinding(_StrictModel):
    kind: Literal["external"] = "external"
    input_port: str = Field(pattern=_ID_PATTERN)
    ref: str = Field(min_length=1, max_length=512)
    schema_id: str = Field(pattern=_SCHEMA_PATTERN)


InputBinding = Annotated[ArtifactBinding | ExternalBinding, Field(discriminator="kind")]


class ActivationSpec(_StrictModel):
    activation_id: str = Field(pattern=_ID_PATTERN)
    runner_kind: RunnerKind
    runner_ref: str = Field(min_length=1, max_length=256)
    lane: ActivationLane = ActivationLane.PRIMARY
    lifecycle: LifecycleScope = LifecycleScope.INVOCATION
    effect_scope: EffectScope = EffectScope.READ_ONLY
    authority_world_id: str | None = Field(default=None, min_length=1, max_length=256)
    authoritative_resource: str | None = Field(default=None, min_length=1, max_length=256)
    inputs: tuple[PortSpecV2, ...] = ()
    outputs: tuple[PortSpecV2, ...] = ()
    bindings: tuple[InputBinding, ...] = ()
    subscriptions: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    estimated_budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    verifier_tags: tuple[str, ...] = ()
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_local_contract(self) -> ActivationSpec:
        input_names = [port.name for port in self.inputs]
        output_names = [port.name for port in self.outputs]
        binding_names = [binding.input_port for binding in self.bindings]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Activation {self.activation_id!r} has duplicate input ports.")
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Activation {self.activation_id!r} has duplicate output ports.")
        if len(binding_names) != len(set(binding_names)):
            raise ValueError(f"Activation {self.activation_id!r} binds an input more than once.")
        if len(self.required_capabilities) != len(set(self.required_capabilities)):
            raise ValueError(
                f"Activation {self.activation_id!r} repeats a required capability."
            )
        if any(not value.strip() for value in self.required_capabilities):
            raise ValueError("Required capabilities must be non-empty strings.")
        if "requested_capabilities" in self.params:
            raise ValueError(
                "params.requested_capabilities is not an authority channel; "
                "declare required_capabilities instead."
            )
        if "budget" in self.params:
            raise ValueError(
                "params.budget is not a budget authority channel; "
                "declare estimated_budget instead."
            )
        if "world_id" in self.params:
            raise ValueError(
                "params.world_id is not an authority channel; declare "
                "authority_world_id instead."
            )
        if len(self.verifier_tags) != len(set(self.verifier_tags)):
            raise ValueError(f"Activation {self.activation_id!r} repeats a verifier tag.")
        if any(not value.strip() for value in self.verifier_tags):
            raise ValueError("Verifier tags must be non-empty strings.")
        unknown = set(binding_names) - set(input_names)
        if unknown:
            raise ValueError(
                f"Activation {self.activation_id!r} binds undeclared inputs: "
                f"{', '.join(sorted(unknown))}."
            )
        missing = [port.name for port in self.inputs if port.required and port.name not in binding_names]
        if missing:
            raise ValueError(
                f"Activation {self.activation_id!r} lacks required bindings: "
                f"{', '.join(missing)}."
            )
        if self.effect_scope == EffectScope.AUTHORITATIVE_WORLD:
            if self.runner_kind != RunnerKind.SYSTEM_ACTION:
                raise ValueError(
                    "Only a system_action runner may request authoritative-world effects."
                )
            if not self.authority_world_id or not self.authoritative_resource:
                raise ValueError(
                    "An authoritative-world activation must name its world and leased resource."
                )
        elif self.runner_kind != RunnerKind.ACTION_SNAPSHOT and (
            self.authority_world_id is not None
            or self.authoritative_resource is not None
        ):
            raise ValueError(
                "authority_world_id/authoritative_resource are valid only for "
                "system_action or action_snapshot runners."
            )
        if self.runner_kind == RunnerKind.SYSTEM_ACTION and (
            self.effect_scope != EffectScope.AUTHORITATIVE_WORLD
        ):
            raise ValueError("A system_action runner must use authoritative_world effects.")
        if self.runner_kind == RunnerKind.ACTION_SNAPSHOT:
            if self.runner_ref != ACTION_SNAPSHOT_RUNNER_REF:
                raise ValueError(
                    "An action_snapshot must use the fixed runtime-owned runner_ref."
                )
            if self.lane != ActivationLane.PRIMARY:
                raise ValueError("An action_snapshot must run on the primary lane.")
            if self.lifecycle != LifecycleScope.INVOCATION:
                raise ValueError("An action_snapshot must use invocation lifecycle.")
            if self.effect_scope != EffectScope.READ_ONLY:
                raise ValueError("An action_snapshot must be read_only.")
            if not self.authority_world_id or not self.authoritative_resource:
                raise ValueError(
                    "An action_snapshot must name its authoritative world and resource."
                )
            if self.inputs or self.bindings:
                raise ValueError("An action_snapshot cannot declare inputs or bindings.")
            if tuple(
                (port.name, port.schema_id, port.required) for port in self.outputs
            ) != (("snapshot", ACTION_SNAPSHOT_SCHEMA_ID, True),):
                raise ValueError(
                    "An action_snapshot requires exactly one required snapshot output."
                )
            if (
                self.subscriptions
                or self.required_capabilities
                or self.verifier_tags
                or self.params
            ):
                raise ValueError(
                    "An action_snapshot accepts no subscriptions, capabilities, verifier "
                    "tags, or params."
                )
            if any(self.estimated_budget.model_dump(mode="python").values()):
                raise ValueError("An action_snapshot must declare a zero execution budget.")
        if self.lane == ActivationLane.SERVICE:
            if self.lifecycle == LifecycleScope.INVOCATION:
                raise ValueError("A service activation must have workflow or episode lifecycle.")
            if self.effect_scope == EffectScope.AUTHORITATIVE_WORLD:
                raise ValueError("A service activation cannot own authoritative-world effects.")
        return self


class TransitionSpec(_StrictModel):
    source: str = Field(pattern=_ID_PATTERN)
    outcome: ControlOutcome
    target: str = Field(pattern=_ID_PATTERN)


class BoundedLoopSpec(_StrictModel):
    loop_id: str = Field(pattern=_ID_PATTERN)
    activation_ids: tuple[str, ...] = Field(min_length=1)
    entry_activation: str = Field(pattern=_ID_PATTERN)
    max_iterations: int = Field(ge=1, le=1000)
    progress_schema_id: str | None = Field(default=None, pattern=_SCHEMA_PATTERN)

    @model_validator(mode="after")
    def _unique_members(self) -> BoundedLoopSpec:
        if len(self.activation_ids) != len(set(self.activation_ids)):
            raise ValueError(f"Loop {self.loop_id!r} contains duplicate activations.")
        if self.entry_activation not in self.activation_ids:
            raise ValueError(
                f"Loop {self.loop_id!r} entry must be one of its declared activations."
            )
        return self


class ElasticGraphSpec(_StrictModel):
    schema_id: Literal["robomex.elastic_graph.v2"] = "robomex.elastic_graph.v2"
    graph_id: str = Field(pattern=_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    entry_activation: str = Field(pattern=_ID_PATTERN)
    terminal_activations: tuple[str, ...] = Field(min_length=1)
    activations: tuple[ActivationSpec, ...] = Field(min_length=1)
    transitions: tuple[TransitionSpec, ...] = ()
    bounded_loops: tuple[BoundedLoopSpec, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
