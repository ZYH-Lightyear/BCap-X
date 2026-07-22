from __future__ import annotations

import pytest

from robomex.contracts import ContractCatalogSnapshot
from robomex.elastic import (
    ActivationSpec,
    BoundedLoopSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    RunnerKind,
    TransitionSpec,
)
from robomex.evolution import RunBudgets
from robomex.orchestration.actors import ActorProfile
from robomex.orchestration.production_manifest import (
    ProductionManifestError,
    assert_run_budgets_cover,
    catalog_function_pins,
    catalog_skill_pins,
    graph_budget_envelope,
    runtime_actor_profile_pins,
)
from robomex.runtime.events import ControlOutcome


def _graph():
    worker = ActivationSpec(
        activation_id="worker",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="worker",
        estimated_budget=ExecutionBudget(
            model_calls=2,
            tokens=100,
            wall_time_ms=1_000,
        ),
    )
    arena = ActivationSpec(
        activation_id="arena",
        runner_kind=RunnerKind.ARENA,
        runner_ref="arena",
        estimated_budget=ExecutionBudget(
            model_calls=3,
            tokens=200,
            wall_time_ms=2_000,
            actor_spawns=3,
            shadow_rollouts=2,
        ),
    )
    action = ActivationSpec(
        activation_id="action",
        runner_kind=RunnerKind.SYSTEM_ACTION,
        runner_ref="runtime.action",
        effect_scope="authoritative_world",
        authority_world_id="live",
        authoritative_resource="arm",
        estimated_budget=ExecutionBudget(authoritative_actions=1),
    )
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="budget",
            entry_activation="worker",
            terminal_activations=("action",),
            activations=(worker, arena, action),
            transitions=(
                TransitionSpec(
                    source="worker",
                    outcome=ControlOutcome.SUCCESS,
                    target="arena",
                ),
                TransitionSpec(
                    source="arena",
                    outcome=ControlOutcome.NEEDS_ADJUSTMENT,
                    target="worker",
                ),
                TransitionSpec(
                    source="arena",
                    outcome=ControlOutcome.SUCCESS,
                    target="action",
                ),
            ),
            bounded_loops=(
                BoundedLoopSpec(
                    loop_id="bounded",
                    activation_ids=("worker", "arena"),
                    entry_activation="worker",
                    max_iterations=2,
                    progress_schema_id="test.progress.v1",
                ),
            ),
        )
    )


def test_graph_budget_multiplies_bounded_loop_and_arena_candidates() -> None:
    envelope = graph_budget_envelope(_graph())
    assert envelope.model_calls == 10
    assert envelope.tokens == 600
    assert envelope.wall_time_s == 6.0
    assert envelope.physical_actions == 1
    assert envelope.shadow_rollouts == 4
    assert envelope.candidates == 6


def test_run_budget_coverage_fails_with_exact_shortfall() -> None:
    envelope = graph_budget_envelope(_graph())
    with pytest.raises(ProductionManifestError, match="max_candidates=5"):
        assert_run_budgets_cover(
            RunBudgets(
                max_model_calls=10,
                max_tokens=600,
                max_wall_time_s=6,
                max_physical_actions=1,
                max_shadow_rollouts=4,
                max_candidates=5,
            ),
            envelope,
        )


def test_catalog_and_profile_pin_derivation_is_exact() -> None:
    empty = ContractCatalogSnapshot()
    assert catalog_skill_pins(empty) == ()
    assert catalog_function_pins(empty) == ()
    profile = ActorProfile(profile_id="profile", provider_id="provider")
    pins = runtime_actor_profile_pins({"runner-a": profile, "runner-b": profile})
    assert len(pins) == 1
    assert pins[0].component_id == "profile"
