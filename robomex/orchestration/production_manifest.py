"""Derive and validate the immutable identity of a production v2 run.

This module contains no convenient fake hashes.  Every skill, exported
function, actor profile, backend and graph budget is derived from the exact
runtime object that will be admitted by :class:`V2RuntimeFactory`.  Operator
owned identities (model weights, prompts and runtime source build) remain
explicit inputs because Python introspection is not a reproducible provenance
mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from robomex.contracts import ContentPin, ContractCatalogSnapshot
from robomex.elastic import CompiledElasticGraph, ElasticGraphSpec, RunnerKind
from robomex.evolution import (
    BackendPin,
    BackendRole,
    FunctionPin,
    RunBudgets,
    SkillPin,
)
from robomex.orchestration.actors import ActorProfile
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    ObservationBackendBinding,
    ShadowBackendBinding,
    actor_profile_digest,
)


class ProductionManifestError(ValueError):
    """Runtime dependencies cannot be represented by one closed manifest."""


@dataclass(frozen=True)
class GraphBudgetEnvelope:
    """Conservative worst-case graph consumption including bounded loops."""

    model_calls: int
    tokens: int
    wall_time_s: float
    physical_actions: int
    shadow_rollouts: int
    candidates: int

    def __post_init__(self) -> None:
        values = (
            self.model_calls,
            self.tokens,
            self.wall_time_s,
            self.physical_actions,
            self.shadow_rollouts,
            self.candidates,
        )
        if any(value < 0 for value in values):
            raise ValueError("graph budget envelope values must be non-negative")


def catalog_skill_pins(catalog: ContractCatalogSnapshot) -> tuple[SkillPin, ...]:
    """Pin every and only skill version present in ``catalog``."""

    if not isinstance(catalog, ContractCatalogSnapshot):
        raise TypeError("catalog must be a ContractCatalogSnapshot")
    return tuple(
        SkillPin(
            skill_id=skill.skill_id,
            revision=skill.revision,
            manifest_digest=skill.content_digest,
        )
        for skill in sorted(catalog.skills, key=lambda item: (item.skill_id, item.revision))
    )


def catalog_function_pins(
    catalog: ContractCatalogSnapshot,
) -> tuple[FunctionPin, ...]:
    """Pin the exact implementation and interface of every catalog export."""

    if not isinstance(catalog, ContractCatalogSnapshot):
        raise TypeError("catalog must be a ContractCatalogSnapshot")
    values = [
        FunctionPin(
            function_id=function.function_id,
            skill_id=skill.skill_id,
            implementation_digest=function.function_digest,
            interface_digest=function.interface_digest,
        )
        for skill in catalog.skills
        for function in skill.functions
    ]
    values.sort(key=lambda item: item.function_id)
    if len({item.function_id for item in values}) != len(values):
        raise ProductionManifestError("catalog contains duplicate function IDs")
    return tuple(values)


def runtime_actor_profile_pins(
    profiles: Mapping[str, ActorProfile] | Iterable[ActorProfile],
) -> tuple[ContentPin, ...]:
    """Pin unique profile definitions, independent of runner aliases."""

    values = profiles.values() if isinstance(profiles, Mapping) else profiles
    unique: dict[str, ActorProfile] = {}
    for profile in values:
        if not isinstance(profile, ActorProfile):
            raise TypeError("profiles must contain ActorProfile values")
        previous = unique.get(profile.profile_id)
        if previous is not None and previous != profile:
            raise ProductionManifestError(
                f"actor profile ID {profile.profile_id!r} has conflicting definitions"
            )
        unique[profile.profile_id] = profile
    return tuple(
        ContentPin(
            component_id=profile_id,
            revision=1,
            content_digest=actor_profile_digest(profile),
        )
        for profile_id, profile in sorted(unique.items())
    )


def runtime_backend_pins(
    *,
    action_bindings: Sequence[ActionBackendBinding],
    observation_bindings: Sequence[ObservationBackendBinding] = (),
    shadow_bindings: Sequence[ShadowBackendBinding] = (),
) -> tuple[BackendPin, ...]:
    """Derive exact backend pins from admitted provenance and roles."""

    admitted: dict[str, BackendPin] = {}

    def add(binding: object, role: BackendRole) -> None:
        provenance = binding.provenance
        pin = BackendPin(
            backend_id=provenance.backend_id,
            role=role,
            implementation_digest=provenance.implementation_digest,
            configuration_digest=provenance.configuration_digest,
            version=provenance.version,
        )
        previous = admitted.get(pin.backend_id)
        if previous is not None and previous != pin:
            raise ProductionManifestError(
                f"backend {pin.backend_id!r} is bound with conflicting provenance or roles"
            )
        admitted[pin.backend_id] = pin

    for binding in action_bindings:
        if not isinstance(binding, ActionBackendBinding):
            raise TypeError("action_bindings must contain ActionBackendBinding values")
        add(binding, BackendRole.AUTHORITATIVE)
    for binding in observation_bindings:
        if not isinstance(binding, ObservationBackendBinding):
            raise TypeError(
                "observation_bindings must contain ObservationBackendBinding values"
            )
        add(binding, BackendRole.PERCEPTION)
    for binding in shadow_bindings:
        if not isinstance(binding, ShadowBackendBinding):
            raise TypeError("shadow_bindings must contain ShadowBackendBinding values")
        add(binding, BackendRole.SHADOW)
    authoritative = [pin for pin in admitted.values() if pin.role is BackendRole.AUTHORITATIVE]
    if len(authoritative) != 1:
        raise ProductionManifestError(
            "production dependencies require exactly one authoritative backend identity"
        )
    return tuple(admitted[key] for key in sorted(admitted))


def graph_budget_envelope(
    graph: CompiledElasticGraph | ElasticGraphSpec,
) -> GraphBudgetEnvelope:
    """Calculate a conservative bound for all declared graph activations.

    Each activation inside a bounded loop is charged once per allowed loop
    iteration.  If future compilers permit nested/overlapping bounded loops,
    their iteration limits multiply, which is conservative and fail closed.
    Arena actor-spawn declarations map to the run-wide candidate dimension;
    ordinary actor lifecycles do not consume that dimension.
    """

    spec = graph.spec if isinstance(graph, CompiledElasticGraph) else graph
    if not isinstance(spec, ElasticGraphSpec):
        raise TypeError("graph must be an ElasticGraphSpec or CompiledElasticGraph")
    multiplier = {node.activation_id: 1 for node in spec.activations}
    for loop in spec.bounded_loops:
        for activation_id in loop.activation_ids:
            multiplier[activation_id] *= loop.max_iterations
    totals = {
        "model_calls": 0,
        "tokens": 0,
        "wall_time_ms": 0,
        "physical_actions": 0,
        "shadow_rollouts": 0,
        "candidates": 0,
    }
    for node in spec.activations:
        factor = multiplier[node.activation_id]
        budget = node.estimated_budget
        totals["model_calls"] += factor * budget.model_calls
        totals["tokens"] += factor * budget.tokens
        totals["wall_time_ms"] += factor * budget.wall_time_ms
        totals["physical_actions"] += factor * budget.authoritative_actions
        totals["shadow_rollouts"] += factor * budget.shadow_rollouts
        if node.runner_kind is RunnerKind.ARENA:
            totals["candidates"] += factor * budget.actor_spawns
    return GraphBudgetEnvelope(
        model_calls=totals["model_calls"],
        tokens=totals["tokens"],
        wall_time_s=totals["wall_time_ms"] / 1000.0,
        physical_actions=totals["physical_actions"],
        shadow_rollouts=totals["shadow_rollouts"],
        candidates=totals["candidates"],
    )


def assert_run_budgets_cover(
    budgets: RunBudgets,
    envelope: GraphBudgetEnvelope,
    *,
    supplemental_model_calls: int = 0,
    supplemental_tokens: int = 0,
    supplemental_wall_time_s: float = 0.0,
    required_recoveries: int = 0,
) -> None:
    """Reject a manifest that cannot complete one worst-case admitted graph."""

    if not isinstance(budgets, RunBudgets):
        raise TypeError("budgets must be RunBudgets")
    for name, value in (
        ("supplemental_model_calls", supplemental_model_calls),
        ("supplemental_tokens", supplemental_tokens),
        ("supplemental_wall_time_s", supplemental_wall_time_s),
        ("required_recoveries", required_recoveries),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    required = {
        "max_model_calls": envelope.model_calls + supplemental_model_calls,
        "max_tokens": envelope.tokens + supplemental_tokens,
        "max_wall_time_s": envelope.wall_time_s + supplemental_wall_time_s,
        "max_physical_actions": envelope.physical_actions,
        "max_shadow_rollouts": envelope.shadow_rollouts,
        # A graph without Arena still needs RunBudgets' schema minimum of one.
        "max_candidates": max(envelope.candidates, 1),
        "max_recoveries": required_recoveries,
    }
    short = {
        name: (getattr(budgets, name), minimum)
        for name, minimum in required.items()
        if getattr(budgets, name) < minimum
    }
    if short:
        detail = ", ".join(
            f"{name}={actual} < required {minimum}"
            for name, (actual, minimum) in sorted(short.items())
        )
        raise ProductionManifestError("run budgets do not cover the admitted graph: " + detail)


__all__ = [
    "GraphBudgetEnvelope",
    "ProductionManifestError",
    "assert_run_budgets_cover",
    "catalog_function_pins",
    "catalog_skill_pins",
    "graph_budget_envelope",
    "runtime_actor_profile_pins",
    "runtime_backend_pins",
]
