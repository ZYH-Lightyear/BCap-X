from __future__ import annotations

import pytest

from robomex.elastic import EffectScope, RunnerKind
from robomex.orchestration.risk_provider import (
    DETERMINISTIC_MOTION_RISK_RUNNER_REF,
    MOTION_RISK_CAPABILITIES,
)
from robomex.protocols.risk_adaptive_bowl_place import (
    ARENA_HYPOTHESES,
    ARENA_PROMOTION_RECEIPT,
    ARENA_RESULT,
    CORRECTION_ARENA_ACTIVATION_ID,
    CORRECTION_CONTEXT_SCHEMAS,
    CORRECTION_EXECUTE_ACTIVATION_ID,
    CORRECTION_RISK_ACTIVATION_ID,
    CORRECTION_SNAPSHOT_ACTIVATION_ID,
    DEFAULT_CORRECTION_STRATEGIES,
    LEGACY_CORRECTION_PLANNER_ACTIVATION_ID,
    RISK_REPORT,
    BowlCorrectionCandidateStrategy,
    RiskAdaptiveBowlPlaceProtocolConfig,
    build_risk_adaptive_bowl_place_graph,
    build_risk_adaptive_bowl_place_protocol,
)
from robomex.runtime.events import ControlOutcome


def _nodes(spec):
    return {node.activation_id: node for node in spec.activations}


def _bindings(node):
    return {binding.input_port: binding for binding in node.bindings}


def test_correction_path_is_fresh_snapshot_then_risk_arena_and_sealed_execute() -> None:
    protocol = build_risk_adaptive_bowl_place_protocol()
    nodes = _nodes(protocol.spec)

    assert LEGACY_CORRECTION_PLANNER_ACTIVATION_ID not in nodes
    risk = nodes[CORRECTION_RISK_ACTIVATION_ID]
    assert risk.runner_kind is RunnerKind.DETERMINISTIC_GATE
    assert risk.runner_ref == DETERMINISTIC_MOTION_RISK_RUNNER_REF
    assert risk.required_capabilities == tuple(sorted(MOTION_RISK_CAPABILITIES))
    assert {port.name: port.schema_id for port in risk.inputs} == dict(CORRECTION_CONTEXT_SCHEMAS)
    assert {port.name: port.schema_id for port in risk.outputs} == {"risk_report": RISK_REPORT}
    assert {
        name: (binding.source_activation, binding.source_port)
        for name, binding in _bindings(risk).items()
    } == {
        "observation": ("capture_alignment", "observation"),
        "attachment_evidence": (
            "verify_alignment_attachment",
            "attachment_evidence",
        ),
        "alignment_error": ("estimate_alignment", "alignment_error"),
        "servo_decision": ("alignment_gate", "servo_decision"),
    }

    arena = nodes[CORRECTION_ARENA_ACTIVATION_ID]
    assert arena.runner_kind is RunnerKind.ARENA
    assert arena.effect_scope is EffectScope.READ_ONLY
    assert arena.params == {}
    assert arena.required_capabilities == ()
    assert {port.name: port.schema_id for port in arena.inputs} == {
        "snapshot": "robomex.admission_snapshot.v1",
        "risk": RISK_REPORT,
        **dict(CORRECTION_CONTEXT_SCHEMAS),
    }
    assert {port.name: (port.schema_id, port.required) for port in arena.outputs} == {
        "result": (ARENA_RESULT, True),
        "hypotheses": (ARENA_HYPOTHESES, True),
        "promotion_receipt": (ARENA_PROMOTION_RECEIPT, False),
        "selected_action_spec": ("robomex.motion_plan.v2", False),
    }
    arena_bindings = _bindings(arena)
    assert (
        arena_bindings["snapshot"].source_activation,
        arena_bindings["snapshot"].source_port,
    ) == (CORRECTION_SNAPSHOT_ACTIVATION_ID, "snapshot")
    assert (
        arena_bindings["risk"].source_activation,
        arena_bindings["risk"].source_port,
    ) == (CORRECTION_RISK_ACTIVATION_ID, "risk_report")
    for name in CORRECTION_CONTEXT_SCHEMAS:
        assert (
            arena_bindings[name].source_activation,
            arena_bindings[name].source_port,
        ) == (
            _bindings(risk)[name].source_activation,
            _bindings(risk)[name].source_port,
        )

    execute = nodes[CORRECTION_EXECUTE_ACTIVATION_ID]
    selected = _bindings(execute)["action_spec"]
    assert execute.runner_kind is RunnerKind.SYSTEM_ACTION
    assert execute.effect_scope is EffectScope.AUTHORITATIVE_WORLD
    assert (selected.source_activation, selected.source_port) == (
        CORRECTION_ARENA_ACTIVATION_ID,
        "selected_action_spec",
    )
    assert execute.params["selected_by"] == CORRECTION_ARENA_ACTIVATION_ID
    assert "requires_promotion_receipt" not in execute.params
    assert protocol.compiled.spec == protocol.spec


def test_risk_and_arena_live_inside_loop_and_fail_only_to_recovery() -> None:
    config = RiskAdaptiveBowlPlaceProtocolConfig(max_alignment_iterations=5)
    spec = build_risk_adaptive_bowl_place_graph(config)
    edges = {(edge.source, edge.outcome, edge.target) for edge in spec.transitions}

    success_path = (
        (
            CORRECTION_SNAPSHOT_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_RISK_ACTIVATION_ID,
        ),
        (
            CORRECTION_RISK_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_ARENA_ACTIVATION_ID,
        ),
        (
            CORRECTION_ARENA_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            CORRECTION_EXECUTE_ACTIVATION_ID,
        ),
        (
            CORRECTION_EXECUTE_ACTIVATION_ID,
            ControlOutcome.SUCCESS,
            "capture_alignment",
        ),
    )
    assert set(success_path).issubset(edges)
    assert {
        target
        for source, outcome, target in edges
        if source == CORRECTION_SNAPSHOT_ACTIVATION_ID and outcome is ControlOutcome.SUCCESS
    } == {CORRECTION_RISK_ACTIVATION_ID}
    assert {
        target
        for source, outcome, target in edges
        if source == CORRECTION_RISK_ACTIVATION_ID and outcome is ControlOutcome.SUCCESS
    } == {CORRECTION_ARENA_ACTIVATION_ID}
    assert {
        target
        for source, outcome, target in edges
        if source == CORRECTION_ARENA_ACTIVATION_ID and outcome is ControlOutcome.SUCCESS
    } == {CORRECTION_EXECUTE_ACTIVATION_ID}

    expected_risk_failures = {
        ControlOutcome.FAILED,
        ControlOutcome.STALE_INPUT,
        ControlOutcome.STALE_OBSERVATION,
        ControlOutcome.UNCERTAIN,
        ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
        ControlOutcome.WRONG_GROUNDING,
        ControlOutcome.INFEASIBLE,
        ControlOutcome.EXHAUSTED,
    }
    expected_arena_failures = {
        ControlOutcome.INFEASIBLE,
        ControlOutcome.EXHAUSTED,
        ControlOutcome.STALE_INPUT,
        ControlOutcome.FAILED,
    }
    assert all(
        (CORRECTION_RISK_ACTIVATION_ID, outcome, "recovery_frontier") in edges
        for outcome in expected_risk_failures
    )
    assert all(
        (CORRECTION_ARENA_ACTIVATION_ID, outcome, "recovery_frontier") in edges
        for outcome in expected_arena_failures
    )
    loop = next(loop for loop in spec.bounded_loops if loop.loop_id == "alignment_visual_servo")
    assert loop.max_iterations == 5
    assert {
        CORRECTION_SNAPSHOT_ACTIVATION_ID,
        CORRECTION_RISK_ACTIVATION_ID,
        CORRECTION_ARENA_ACTIVATION_ID,
        CORRECTION_EXECUTE_ACTIVATION_ID,
    }.issubset(loop.activation_ids)
    assert LEGACY_CORRECTION_PLANNER_ACTIVATION_ID not in loop.activation_ids


def test_arena_activation_budget_covers_one_full_k_candidate_round() -> None:
    config = RiskAdaptiveBowlPlaceProtocolConfig(
        max_alignment_iterations=5,
        max_arena_candidates=3,
    )
    arena = _nodes(build_risk_adaptive_bowl_place_graph(config))[CORRECTION_ARENA_ACTIVATION_ID]
    per_candidate = config.arena_candidate_budget

    assert arena.estimated_budget.actor_spawns == 3
    assert arena.estimated_budget.model_calls == 3 * per_candidate.model_calls
    assert arena.estimated_budget.tokens == 3 * per_candidate.tokens
    assert arena.estimated_budget.wall_time_ms == 3 * per_candidate.wall_time_ms
    assert arena.runner_ref == config.correction_arena_binding_id
    assert config.correction_candidate_budget_limit == 15


def test_default_strategies_are_neutral_and_cannot_preselect_a_winner() -> None:
    assert len(DEFAULT_CORRECTION_STRATEGIES) == 3
    assert len({item.candidate_id for item in DEFAULT_CORRECTION_STRATEGIES}) == 3
    assert len({item.strategy for item in DEFAULT_CORRECTION_STRATEGIES}) == 3
    assert len({item.objective for item in DEFAULT_CORRECTION_STRATEGIES}) == 3
    assert {
        (item.utility, item.estimated_risk, item.clearance_m)
        for item in DEFAULT_CORRECTION_STRATEGIES
    } == {(0.0, 0.5, None)}


def test_strategy_contract_rejects_ambiguous_or_unbounded_priors() -> None:
    with pytest.raises(ValueError, match="unique"):
        BowlCorrectionCandidateStrategy(
            candidate_id="duplicate-precondition",
            strategy="test",
            objective="test objective",
            preconditions=("same", "same"),
        )
    with pytest.raises(ValueError):
        BowlCorrectionCandidateStrategy(
            candidate_id="invalid-risk",
            strategy="test",
            objective="test objective",
            estimated_risk=1.1,
        )
