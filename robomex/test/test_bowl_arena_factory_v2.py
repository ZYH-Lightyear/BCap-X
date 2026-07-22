from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from robomex.contracts import canonical_payload_digest
from robomex.core.coder import ScriptedCodePolicy
from robomex.data import ResolvedArtifactRef
from robomex.elastic import EffectScope
from robomex.orchestration.actors import ActorLifecycle
from robomex.orchestration.arena import (
    ActionHypothesis,
    ArenaConsumptionLedger,
    ArenaContext,
    CheckStatus,
    RiskInputs,
    RiskLevel,
    RiskPolicy,
    RiskReport,
)
from robomex.orchestration.arena_coding_provider import ArenaCodingAgentProvider
from robomex.orchestration.bowl_arena import (
    BowlArenaComponentProvenance,
    BowlArenaFactoryError,
    BowlArenaPointCloudPreviewPin,
    BowlArenaPreviewRuntimeProvenance,
    BowlCorrectionArenaFactoryConfig,
    BowlMotionConfigurationGate,
    BowlMotionPlanningPin,
    build_bowl_correction_arena_factory,
    pointcloud_preview_configuration_digest,
)
from robomex.orchestration.coding_provider import SkillCodingAgentProvider
from robomex.orchestration.motion_preview import (
    PointCloudMotionPreviewRenderer,
    PointCloudPreviewConfig,
)
from robomex.orchestration.risk_provider import MotionRiskProviderConfig
from robomex.protocols.bowl_place import CodingPhaseBudgetConfig
from robomex.protocols.risk_adaptive_bowl_place import (
    CORRECTION_CONTEXT_SCHEMAS,
    RiskAdaptiveBowlPlaceProtocolConfig,
)
from robomex.skills import Skill, SkillLibrary


def _digest(label: str) -> str:
    return canonical_payload_digest("robomex.test_pin.v1", {"label": label})


def _protocol(*, iterations: int = 2, count: int = 3):
    return RiskAdaptiveBowlPlaceProtocolConfig(
        max_alignment_iterations=iterations,
        max_arena_candidates=count,
    )


def _risk(*, count: int = 3) -> MotionRiskProviderConfig:
    return MotionRiskProviderConfig(
        expected_bowl_entity_id="bowl-1",
        expected_target_entity_id="plate-1",
        risk_policy=RiskPolicy(max_candidates=count),
        fixed_ik_status=CheckStatus.PASS,
        fixed_collision_status=CheckStatus.PASS,
        fixed_clearance_m=0.03,
    )


def _planning() -> BowlMotionPlanningPin:
    return BowlMotionPlanningPin(
        robot_model_digest=_digest("panda-model"),
        robot_config_digest=_digest("panda-runtime-config"),
        expected_frame="world",
        tcp_frame_id="panda_hand",
        planner_backend="curobo",
        planner_configuration_digest=_digest("curobo-production-config"),
    )


def _coding_provider(tmp_path: Path) -> SkillCodingAgentProvider:
    library = SkillLibrary(tmp_path / "skill-library")
    builtin = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "builtin"
        / "motion"
        / "author_sealed_phase_motion"
    )
    library.admit(Skill.from_dir(builtin))
    return SkillCodingAgentProvider(
        executor_factory=lambda *_: object(),
        library=library,
        policy=ScriptedCodePolicy([]),
        artifacts_root=tmp_path / "coding-provider",
        trusted_skill_sidecars=frozenset({"author_sealed_phase_motion"}),
    )


def _config(
    *,
    protocol: RiskAdaptiveBowlPlaceProtocolConfig,
    risk: MotionRiskProviderConfig,
    preview: BowlArenaPointCloudPreviewPin | None = None,
    budgets: tuple[CodingPhaseBudgetConfig, ...] | None = None,
) -> BowlCorrectionArenaFactoryConfig:
    return BowlCorrectionArenaFactoryConfig.from_protocol(
        protocol_config=protocol,
        risk_config=risk,
        planning=_planning(),
        preview=preview,
        candidate_budgets=budgets,
        candidate_models=("small-direct", "small-clearance", "small-conservative"),
    )


def test_factory_builds_strict_heterogeneous_neutral_read_only_candidates(
    tmp_path: Path,
) -> None:
    protocol = _protocol(iterations=2, count=3)
    risk = _risk(count=3)
    budgets = (
        CodingPhaseBudgetConfig(model_calls=1, tokens=16_000, wall_time_ms=10_000),
        CodingPhaseBudgetConfig(model_calls=2, tokens=32_000, wall_time_ms=20_000),
        CodingPhaseBudgetConfig(model_calls=3, tokens=64_000, wall_time_ms=60_000),
    )
    config = _config(protocol=protocol, risk=risk, budgets=budgets)

    assembly = build_bowl_correction_arena_factory(
        config,
        protocol_config=protocol,
        risk_config=risk,
        coding_provider=_coding_provider(tmp_path),
    )

    assert set(assembly.providers) == {config.provider_id}
    assert isinstance(assembly.provider, ArenaCodingAgentProvider)
    assert assembly.provider.renderer is None
    assert assembly.renderer is None
    assert assembly.preview_runtime_provenance is None
    assert assembly.binding.context_input_schemas == CORRECTION_CONTEXT_SCHEMAS
    assert assembly.binding.policy.max_candidates == 3
    assert assembly.binding.risk_policy.max_candidates == 3
    # Quota is cumulative across the two correction-loop rounds, not one K.
    assert assembly.binding.candidate_budget_limit == 2 * 3
    assert assembly.binding.candidate_budget_id == config.candidate_budget_id
    assert assembly.required_skill_ids == ("author_sealed_phase_motion",)
    assert assembly.candidate_config_digest.startswith("sha256:")
    assert set(assembly.profile_digests) == set(assembly.profiles)

    candidates = assembly.binding.candidates
    assert len(candidates) == 3
    assert len({item.strategy for item in candidates}) == 3
    assert len({item.objective for item in candidates}) == 3
    assert tuple(item.estimated_budget for item in candidates) == tuple(
        budget.execution_budget() for budget in budgets
    )
    for candidate in candidates:
        profile = candidate.profile
        assert profile.lifecycle is ActorLifecycle.EPHEMERAL
        assert profile.provider_id == config.provider_id
        assert profile.capability_ceiling == frozenset({"motion.plan"})
        assert profile.effect_ceiling == frozenset()
        assert profile.metadata["strict_runtime_context"] is True
        assert profile.metadata["execution_mode"] == "read_only"
        assert candidate.effect_scope is EffectScope.READ_ONLY
        assert candidate.requested_capabilities == frozenset({"motion.plan"})
        assert candidate.requested_effects == frozenset()
        assert candidate.shadow_backend_id is None
        assert candidate.shadow_resource_id is None
        assert candidate.preview_config.enabled is False
        assert candidate.hypothesis_config.utility == 0.0
        assert candidate.hypothesis_config.estimated_risk == 0.5
        assert candidate.hypothesis_config.clearance_m is None
        rows = profile.metadata["node_config_v1"]
        assert len(rows) == 1
        node = json.loads(rows[0][1])
        assert node == {
            "plan_kind": "bounded_correction",
            "planner_backend": "curobo",
            "planner_configuration_digest": _planning().planner_configuration_digest,
            "robot_model_digest": _planning().robot_model_digest,
            "schema_version": "robomex.coding_node_config.v1",
            "tcp_frame_id": "panda_hand",
        }


def test_factory_rejects_sealed_config_drift_and_dependency_spoof(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    risk = _risk()
    config = _config(protocol=protocol, risk=risk)
    provider = _coding_provider(tmp_path)
    drifted = config.model_copy(update={"provider_id": "spoofed-provider"})

    with pytest.raises(ValueError, match="content_digest"):
        build_bowl_correction_arena_factory(
            drifted,
            protocol_config=protocol,
            risk_config=risk,
            coding_provider=provider,
        )

    changed_protocol = _protocol(iterations=3)
    with pytest.raises(BowlArenaFactoryError, match="protocol_config_digest"):
        build_bowl_correction_arena_factory(
            config,
            protocol_config=changed_protocol,
            risk_config=risk,
            coding_provider=provider,
        )

    changed_risk = _risk(count=2)
    with pytest.raises(BowlArenaFactoryError, match="risk_config_digest"):
        build_bowl_correction_arena_factory(
            config,
            protocol_config=protocol,
            risk_config=changed_risk,
            coding_provider=provider,
        )


def test_configuration_gate_rejects_bridge_metadata_spoof() -> None:
    planning = _planning()
    gate = BowlMotionConfigurationGate(
        robot_model_digest=planning.robot_model_digest,
        robot_config_digest=planning.robot_config_digest,
        expected_frame=planning.expected_frame,
        tcp_frame_id=planning.tcp_frame_id,
        planner_backend=planning.planner_backend,
        planner_configuration_digest=planning.planner_configuration_digest,
    )
    ref = ResolvedArtifactRef("artifact-1", _digest("artifact-1"))
    metadata = {
        "robot_model_digest": planning.robot_model_digest,
        "config_digest": planning.robot_config_digest,
        "expected_frame": planning.expected_frame,
        "required_tcp_frame_id": planning.tcp_frame_id,
        "planner_backend": planning.planner_backend,
        "planner_configuration_digest": planning.planner_configuration_digest,
        "plan_kind": planning.plan_kind,
    }
    hypothesis = ActionHypothesis(
        candidate_id="direct",
        strategy="direct_bounded_servo",
        plan_ref=ref,
        snapshot_ref=ref,
        frame="world",
        expected_effect="bounded_alignment_correction",
        metadata=metadata,
    )

    assert gate.evaluate(hypothesis).passed
    spoofed = hypothesis.model_copy(
        update={"metadata": {**metadata, "config_digest": _digest("attacker")}}
    )
    result = gate.evaluate(spoofed)
    assert not result.passed
    assert "config_digest" in result.reason


class _Geometry:
    provider_id = "curobo-fk-scene-v1"

    def scene_points(self, *, context):
        del context
        return np.zeros((32, 3), dtype=float)

    def tcp_positions(self, *, plan, context):
        del context
        return np.zeros((len(plan.motion.positions_rad), 3), dtype=float)


def _preview() -> tuple[
    BowlArenaPointCloudPreviewPin,
    BowlArenaPreviewRuntimeProvenance,
]:
    renderer_config = PointCloudPreviewConfig(
        renderer_id="robomex.pointcloud_motion_preview.v1",
        max_scene_points=128,
        views=("perspective", "top_down"),
    )
    provenance = BowlArenaPreviewRuntimeProvenance(
        renderer=BowlArenaComponentProvenance(
            component_id=renderer_config.renderer_id,
            implementation_digest=_digest("pointcloud-renderer-wheel"),
            configuration_digest=pointcloud_preview_configuration_digest(
                renderer_config
            ),
            version="2.0.0",
        ),
        geometry=BowlArenaComponentProvenance(
            component_id="curobo-fk-scene-v1",
            implementation_digest=_digest("curobo-geometry-adapter-wheel"),
            configuration_digest=_digest("curobo-geometry-adapter-config"),
            version="1.0.0",
        ),
    )
    return (
        BowlArenaPointCloudPreviewPin(
            enabled=True,
            renderer_config=renderer_config,
            provenance=provenance,
        ),
        provenance,
    )


def test_enabled_preview_requires_exact_renderer_and_geometry_provenance(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    risk = _risk()
    preview, provenance = _preview()
    config = _config(protocol=protocol, risk=risk, preview=preview)
    provider = _coding_provider(tmp_path)

    assembly = build_bowl_correction_arena_factory(
        config,
        protocol_config=protocol,
        risk_config=risk,
        coding_provider=provider,
        preview_geometry=_Geometry(),
        preview_output_root=tmp_path / "previews",
        preview_runtime_provenance=provenance,
    )

    assert isinstance(assembly.renderer, PointCloudMotionPreviewRenderer)
    assert assembly.provider.renderer is assembly.renderer
    assert assembly.renderer.renderer_id == preview.renderer_config.renderer_id
    assert assembly.preview_runtime_provenance == provenance
    assert all(item.preview_config.enabled for item in assembly.binding.candidates)
    assert all(
        item.preview_config.renderer_id == preview.renderer_config.renderer_id
        for item in assembly.binding.candidates
    )

    spoofed_runtime = provenance.model_copy(
        update={
            "geometry": provenance.geometry.model_copy(
                update={"implementation_digest": _digest("spoofed-geometry")}
            )
        }
    )
    with pytest.raises(BowlArenaFactoryError, match="runtime preview provenance"):
        build_bowl_correction_arena_factory(
            config,
            protocol_config=protocol,
            risk_config=risk,
            coding_provider=provider,
            preview_geometry=_Geometry(),
            preview_output_root=tmp_path / "spoofed-previews",
            preview_runtime_provenance=spoofed_runtime,
        )


def test_disabled_preview_and_read_only_mode_are_closed(tmp_path: Path) -> None:
    protocol = _protocol()
    risk = _risk()
    config = _config(protocol=protocol, risk=risk)

    with pytest.raises(ValidationError, match="read_only"):
        BowlCorrectionArenaFactoryConfig.model_validate(
            {**config.model_dump(mode="python"), "execution_mode": "shadow_world"}
        )

    with pytest.raises(BowlArenaFactoryError, match="disabled preview"):
        build_bowl_correction_arena_factory(
            config,
            protocol_config=protocol,
            risk_config=risk,
            coding_provider=_coding_provider(tmp_path),
            preview_geometry=_Geometry(),
        )


def _arena_context(
    config: BowlCorrectionArenaFactoryConfig,
    *,
    run_id: str,
) -> ArenaContext:
    return ArenaContext(
        arena_run_id=run_id,
        episode_id="episode-1",
        workflow_id="workflow-1",
        graph_id=config.graph_id,
        graph_revision=config.graph_revision,
        graph_digest=_digest("graph"),
        slot_id="correction_arena",
        snapshot_ref=ResolvedArtifactRef("snapshot-1", _digest("snapshot-1")),
        expected_frame=config.planning.expected_frame,
        world_id="robot-world",
        resource_id="panda-arm",
        robot_model_digest=config.planning.robot_model_digest,
        config_digest=config.planning.robot_config_digest,
        candidate_budget_id=config.candidate_budget_id,
        candidate_budget_limit=config.total_candidate_budget_limit,
    )


def test_candidate_quota_persists_across_two_rounds_then_exhausts(
    tmp_path: Path,
) -> None:
    protocol = _protocol(iterations=2, count=3)
    config = _config(protocol=protocol, risk=_risk(count=3))
    ledger_path = tmp_path / "arena-ledger.jsonl"
    first_ledger = ArenaConsumptionLedger(ledger_path)

    first = first_ledger.reserve(
        context=_arena_context(config, run_id="round-1"),
        requested_candidates=3,
        caller_remaining=3,
    )
    assert first.reserved_candidates == 3

    # A fresh process-local object must recover the durable first reservation.
    restarted = ArenaConsumptionLedger(ledger_path)
    second = restarted.reserve(
        context=_arena_context(config, run_id="round-2"),
        requested_candidates=3,
        caller_remaining=3,
    )
    exhausted = restarted.reserve(
        context=_arena_context(config, run_id="round-3"),
        requested_candidates=3,
        caller_remaining=3,
    )

    assert second.reserved_candidates == 3
    assert exhausted.reserved_candidates == 0
    assert sum(item.reserved_candidates for item in restarted.records) == 6


def test_risk_expands_low_to_one_and_high_to_exact_k() -> None:
    policy = RiskPolicy(max_candidates=3)
    low = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.99,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.04,
            held_pose_uncertainty_m=0.001,
            monitor_observable=True,
        ),
        policy,
    )
    high = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.99,
            target_margin_m=0.05,
            ik_status=CheckStatus.FAIL,
            collision_status=CheckStatus.PASS,
            clearance_m=0.04,
            held_pose_uncertainty_m=0.001,
            monitor_observable=True,
        ),
        policy,
    )

    assert low.level is RiskLevel.LOW
    assert low.recommended_candidates == 1
    assert high.level is RiskLevel.HIGH
    assert high.recommended_candidates == 3
