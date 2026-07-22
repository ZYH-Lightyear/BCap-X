from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from robomex.data import (
    AttachmentStatus,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    PhysicalStateTrigger,
    StateTransitionProposal,
    core_schema_registry,
)
from robomex.evolution import ModelPin, PromptPin, RunBudgets, TaskSnapshot
from robomex.orchestration.arena import CheckStatus, RiskPolicy
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    BackendProvenance,
    ObservationBackendBinding,
)
from robomex.orchestration.bowl_application import (
    BowlPlaceHandoffError,
    FeasibilityCheckerPin,
    FixedBowlPlaceApplicationConfig,
)
from robomex.orchestration.bowl_arena import (
    BowlArenaComponentProvenance,
    BowlArenaFactoryError,
    BowlArenaPointCloudPreviewPin,
    BowlArenaPreviewRuntimeProvenance,
    BowlCorrectionArenaFactoryConfig,
    BowlMotionPlanningPin,
    pointcloud_preview_configuration_digest,
)
from robomex.orchestration.bowl_provider import BowlPlaceProviderConfig
from robomex.orchestration.bowl_swarm_application import (
    BowlSwarmApplicationAssemblyError,
    RiskAdaptiveBowlPlaceApplicationConfig,
    build_risk_adaptive_bowl_place_application,
)
from robomex.orchestration.intent import EntityRef, SubgoalIntent
from robomex.orchestration.motion_preview import PointCloudPreviewConfig
from robomex.orchestration.risk_provider import (
    DETERMINISTIC_MOTION_RISK_RUNNER_REF,
    MOTION_RISK_PROVIDER_ID,
    MotionRiskProviderConfig,
)
from robomex.protocols.risk_adaptive_bowl_place import (
    CORRECTION_ARENA_ACTIVATION_ID,
    RiskAdaptiveBowlPlaceProtocolConfig,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ControllerState,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
    WorldKind,
)
from robomex.runtime.observation import InMemoryObservationBackend


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


class _Checker:
    checker_id = "checker-main"

    def certify(self, _spec, _snapshot):  # pragma: no cover - assembly only
        raise AssertionError("assembly must not execute feasibility checks")


class _ActionBackend:
    def __init__(self, *, config_digest: str) -> None:
        self.config_digest = config_digest
        self._descriptor = BackendDescriptor(
            backend_id="robot-backend",
            motion_interface=BackendMotionInterface.EXACT_JOINT_PATH,
            watchdog_stop_thread_safe=True,
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            world_id=world_id,
            world_kind=WorldKind.AUTHORITATIVE,
            resource_id=resource_id,
            robot_revision=7,
            scene_revision=9,
            attachment_revision=4,
            config_revision=3,
            joint_names=("joint-1",),
            joint_positions_rad=(0.0,),
            config_digest=self.config_digest,
            collision_world_digest=_digest("collision-world"),
            attachment_status=AttachmentStatus.ATTEMPTED,
            controller_state=ControllerState.READY,
        )

    def execute_joint_path(self, **_kwargs) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def execute_joint_path_cooperative(self, **_kwargs) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def set_gripper(self, **_kwargs) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def wait(self, **_kwargs) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def stop_and_wait_quiescent(
        self,
        *,
        world_id: str,
        resource_id: str,
        timeout_s: float,
    ) -> AdmissionSnapshot:
        del timeout_s
        return self.snapshot(world_id, resource_id)

    def monitor_telemetry_capabilities(
        self,
        *,
        world_id: str,
        resource_id: str,
    ) -> MonitorTelemetryCapabilities:
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=(
                "attachment_status",
                "held_entity_visible",
                "identity_match",
            ),
            supported_hooks=(MonitorTelemetryHook.CONTROL,),
            cooperative_stop_guaranteed=True,
        )

    def monitor_sample(self, **_kwargs):
        return {
            "attachment_status": "verified_held",
            "held_entity_visible": True,
            "identity_match": True,
        }


class _Policy:
    def complete(self, _messages):  # pragma: no cover - assembly only
        raise AssertionError("assembly must not call the coding model")


class _Executor:
    def run_block(self, _block):  # pragma: no cover - assembly only
        raise AssertionError("assembly must not execute authored code")


class _Geometry:
    provider_id = "curobo-fk-scene-v1"

    def scene_points(self, *, context):
        del context
        return np.zeros((32, 3), dtype=float)

    def tcp_positions(self, *, plan, context):
        del context
        return np.zeros((len(plan.motion.positions_rad), 3), dtype=float)


def _provenance(backend_id: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        implementation_digest=_digest(f"{backend_id}-implementation"),
        configuration_digest=_digest(f"{backend_id}-configuration"),
        version="1.0.0",
    )


def _bindings(
    *,
    config_digest: str,
) -> tuple[tuple[ActionBackendBinding, ...], tuple[ObservationBackendBinding, ...]]:
    backend = _ActionBackend(config_digest=config_digest)
    checker = _Checker()
    provenance = _provenance("robot-backend")
    action = tuple(
        ActionBackendBinding(
            world_id="authoritative",
            resource_id=resource_id,
            backend=backend,
            feasibility_checker=checker,
            provenance=provenance,
        )
        for resource_id in ("robot.arm", "robot.gripper", "robot.controller")
    )
    observation_backend = InMemoryObservationBackend("bowl-observation-backend")
    observation = (
        ObservationBackendBinding(
            backend=observation_backend,
            provenance=_provenance("bowl-observation-backend"),
        ),
    )
    return action, observation


def _seed_pick_handoff(root: Path) -> None:
    plane = EpisodeDataPlane(
        root,
        episode_id="episode-1",
        schema_registry=core_schema_registry(),
        strict_schema_prefixes=("robomex.",),
    )
    plane.open_workflow("prior-pick")
    evidence = plane.publish(
        workflow_id="prior-pick",
        activation_id="grasp",
        attempt=1,
        port="receipt",
        schema="test.prior_pick_receipt.v1",
        payload={"action_id": "grasp-action-1"},
    )
    reducer = EmbodiedStateReducer(
        root,
        episode_id="episode-1",
        resolver=plane.resolver,
        strict_evidence=True,
    )
    for entity_id, label, track_id in (
        ("bowl-1", "bowl", "track-bowl-1"),
        ("plate-1", "plate", "track-plate-1"),
    ):
        reducer.commit(
            StateTransitionProposal.register_entity(
                episode_id="episode-1",
                effect_id=f"register-{entity_id}",
                before_revision=reducer.state.revision,
                source="prior-grounding",
                evidence_refs=(evidence.ref,),
                entity_id=entity_id,
                semantic_label=label,
                track_id=track_id,
            )
        )
    reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id="episode-1",
            effect_id="attempt-grasp-1",
            before_revision=reducer.state.revision,
            source="prior-pick",
            evidence_refs=(evidence.ref,),
            entity_id="bowl-1",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp-action-1",
            trigger=PhysicalStateTrigger.EVIDENCE,
            track_id="track-bowl-1",
        )
    )
    plane.close_workflow("prior-pick")


def _protocol(*, iterations: int = 2, candidates: int = 3):
    return RiskAdaptiveBowlPlaceProtocolConfig(
        max_alignment_iterations=iterations,
        max_arena_candidates=candidates,
    )


def _risk(
    *,
    policy: RiskPolicy | None = None,
    bowl_entity_id: str = "bowl-1",
    plate_entity_id: str = "plate-1",
) -> MotionRiskProviderConfig:
    return MotionRiskProviderConfig(
        expected_bowl_entity_id=bowl_entity_id,
        expected_target_entity_id=plate_entity_id,
        risk_policy=policy or RiskPolicy(max_candidates=3),
        fixed_ik_status=CheckStatus.PASS,
        fixed_collision_status=CheckStatus.PASS,
        fixed_clearance_m=0.03,
    )


def _planning(
    *,
    expected_frame: str = "world",
    robot_model_digest: str | None = None,
    robot_config_digest: str | None = None,
    tcp_frame_id: str = "panda_hand",
    planner_backend: str = "curobo",
) -> BowlMotionPlanningPin:
    return BowlMotionPlanningPin(
        robot_model_digest=robot_model_digest or _digest("robot-model"),
        robot_config_digest=robot_config_digest or _digest("robot-runtime-config"),
        expected_frame=expected_frame,
        tcp_frame_id=tcp_frame_id,
        planner_backend=planner_backend,
        planner_configuration_digest=_digest("curobo-production-config"),
    )


def _base_config(
    root: Path,
    *,
    protocol: RiskAdaptiveBowlPlaceProtocolConfig,
    budgets: RunBudgets | None = None,
    coding_provider_id: str = "skill_coding",
) -> FixedBowlPlaceApplicationConfig:
    checker_pin = FeasibilityCheckerPin(
        checker_id="checker-main",
        implementation_digest=_digest("checker-implementation"),
        configuration_digest=_digest("checker-configuration"),
        version="1.0.0",
    )
    return FixedBowlPlaceApplicationConfig(
        run_id="run-1",
        episode_id="episode-1",
        episode_root=root,
        task=TaskSnapshot(
            task_id="place-bowl",
            instruction="Put the held bowl on the plate.",
            success_rubric="The bowl is stably supported by the plate.",
            episode_spec_digest=_digest("episode-spec"),
        ),
        intent=SubgoalIntent(
            intent_id="place-bowl-intent",
            instruction="Visually servo the held bowl onto the plate.",
            success_rubric="Fresh evidence confirms supported_by(bowl, plate).",
            entity_refs=(
                EntityRef(entity_id="bowl-1", role="held_object"),
                EntityRef(entity_id="plate-1", role="support_target"),
            ),
        ),
        seed=17,
        model=ModelPin(
            model_id="coding-model",
            provider_id="model-provider",
            weights_digest=_digest("weights"),
            generation_config_digest=_digest("generation-config"),
            tokenizer_digest=_digest("tokenizer"),
        ),
        prompts=(
            PromptPin(
                prompt_id="bowl-coding-prompt",
                content_digest=_digest("prompt"),
            ),
        ),
        budgets=budgets
        or RunBudgets(
            max_model_calls=100_000,
            max_tokens=1_000_000_000,
            max_wall_time_s=1_000_000.0,
            max_physical_actions=100_000,
            max_shadow_rollouts=100_000,
            max_candidates=protocol.correction_candidate_budget_limit,
            max_recoveries=10,
        ),
        runtime_code_digest=_digest("runtime-code"),
        robot_model_digest=_digest("robot-model"),
        feasibility_checker_pins=dict.fromkeys(
            ("robot.arm", "robot.gripper", "robot.controller"),
            checker_pin,
        ),
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        protocol=protocol,
        provider=BowlPlaceProviderConfig(
            correction_limits=BowlPlaceProviderConfig().correction_limits.model_copy(
                update={"max_iterations": protocol.max_alignment_iterations}
            )
        ),
        coding_provider_id=coding_provider_id,
    )


def _swarm_config(
    root: Path,
    *,
    protocol: RiskAdaptiveBowlPlaceProtocolConfig | None = None,
    risk: MotionRiskProviderConfig | None = None,
    planning: BowlMotionPlanningPin | None = None,
    budgets: RunBudgets | None = None,
    coding_provider_id: str = "skill_coding",
    candidate_models: tuple[str, ...] | None = None,
    candidate_model_pins: dict[str, ModelPin] | None = None,
    total_candidate_budget_limit: int | None = None,
    preview: BowlArenaPointCloudPreviewPin | None = None,
    risk_provider_id: str = MOTION_RISK_PROVIDER_ID,
) -> RiskAdaptiveBowlPlaceApplicationConfig:
    protocol = protocol or _protocol()
    risk = risk or _risk(policy=RiskPolicy(max_candidates=protocol.max_arena_candidates))
    base = _base_config(
        root,
        protocol=protocol,
        budgets=budgets,
        coding_provider_id=coding_provider_id,
    )
    arena = BowlCorrectionArenaFactoryConfig.from_protocol(
        protocol_config=protocol,
        risk_config=risk,
        planning=planning or _planning(),
        candidate_models=candidate_models,
        total_candidate_budget_limit=total_candidate_budget_limit,
        preview=preview,
    )
    return RiskAdaptiveBowlPlaceApplicationConfig(
        base=base,
        risk=risk,
        arena=arena,
        candidate_model_pins=candidate_model_pins or {},
        risk_provider_id=risk_provider_id,
    )


def _build(config: RiskAdaptiveBowlPlaceApplicationConfig, *, action=None, observation=None, **kwargs):
    if action is None or observation is None:
        action, observation = _bindings(
            config_digest=config.arena.planning.robot_config_digest
        )
    return build_risk_adaptive_bowl_place_application(
        config=config,
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy=_Policy(),
        **kwargs,
    )


def _preview_pin() -> tuple[
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
            component_id=_Geometry.provider_id,
            implementation_digest=_digest("curobo-geometry-wheel"),
            configuration_digest=_digest("curobo-geometry-config"),
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


def test_complete_swarm_build_shares_one_data_plane_and_pins_closed_inventory(
    tmp_path: Path,
) -> None:
    _seed_pick_handoff(tmp_path)
    config = _swarm_config(tmp_path)

    assembled = _build(config)

    fixed = assembled.fixed
    plane = fixed.application.episode.data_plane
    assert assembled.agent is fixed.agent
    assert assembled.application is fixed.application
    assert assembled.manifest is fixed.manifest
    assert fixed.coding_provider.data_plane is plane
    assert assembled.arena.provider.data_plane is plane
    assert assembled.arena.provider.coding_provider is fixed.coding_provider
    assert assembled.risk_provider.data_plane is plane

    assert set(assembled.risk_bindings.providers) == {MOTION_RISK_PROVIDER_ID}
    assert set(assembled.risk_bindings.profiles) == {
        DETERMINISTIC_MOTION_RISK_RUNNER_REF
    }
    assert len(assembled.arena.profiles) == config.protocol.max_arena_candidates
    assert set(assembled.extension_providers) == {
        MOTION_RISK_PROVIDER_ID,
        config.arena.provider_id,
    }
    assert set(assembled.extension_profiles) == {
        *assembled.risk_bindings.profiles,
        *assembled.arena.profiles,
    }
    for provider_id, provider in assembled.extension_providers.items():
        assert fixed.dependencies.actor_providers[provider_id] is provider
    assert fixed.dependencies.arena_bindings == (assembled.arena.binding,)

    arena_profile_ids = {
        profile.profile_id for profile in assembled.arena.profiles.values()
    }
    motion_skill = next(
        skill
        for skill in fixed.contract_catalog.skills
        if skill.skill_id == "author_sealed_phase_motion"
    )
    assert arena_profile_ids <= set(motion_skill.compatible_actor_profiles)
    assert all(
        candidate.profile.profile_id in arena_profile_ids
        for candidate in assembled.arena.binding.candidates
    )

    manifest = assembled.manifest
    metadata = manifest.metadata
    expected_total = (
        config.protocol.max_alignment_iterations
        * config.protocol.max_arena_candidates
    )
    assert manifest.graph_digest == f"sha256:{assembled.protocol.compiled.digest}"
    assert manifest.candidate_config_digest == assembled.arena.candidate_config_digest
    assert manifest.mutation_policy == "disabled"
    assert metadata["baseline_mutation_policy"] == "disabled"
    assert metadata["swarm_application_config_digest"] == config.content_digest
    assert metadata["risk_provider_config_digest"] == config.risk.content_digest
    assert metadata["arena_factory_config_digest"] == config.arena.content_digest
    assert metadata["arena_candidate_config_digest"] == manifest.candidate_config_digest
    assert metadata["arena_binding_digest"] == assembled.arena.binding.content_digest
    assert metadata["arena_single_round_k"] == config.protocol.max_arena_candidates
    assert metadata["arena_total_candidate_budget"] == expected_total
    assert metadata["compiled_swarm_graph_digest"] == manifest.graph_digest
    assert metadata["authoritative_correction_activation"] == "execute_correction"
    assert metadata["candidate_effect_authority"] == "read_only"
    assert metadata["physical_writer_policy"] == "runtime_sealed_action_only"
    assert assembled.arena.binding.candidate_budget_limit == expected_total
    assert fixed.graph_budget_envelope.candidates == expected_total
    assert manifest.budgets.max_candidates == expected_total

    arena_node = next(
        node
        for node in assembled.protocol.spec.activations
        if node.activation_id == CORRECTION_ARENA_ACTIVATION_ID
    )
    assert arena_node.estimated_budget.actor_spawns == config.protocol.max_arena_candidates
    assert {
        pin.component_id for pin in manifest.actor_profile_pins
    }.issuperset(arena_profile_ids)


def test_rebuilding_exact_config_preserves_run_identity(tmp_path: Path) -> None:
    _seed_pick_handoff(tmp_path)
    config = _swarm_config(tmp_path)
    action, observation = _bindings(
        config_digest=config.arena.planning.robot_config_digest
    )

    first = _build(config, action=action, observation=observation)
    second = _build(config, action=action, observation=observation)

    assert second.manifest == first.manifest
    assert second.manifest.content_digest == first.manifest.content_digest
    assert (
        second.fixed.contract_catalog.content_digest
        == first.fixed.contract_catalog.content_digest
    )
    assert second.arena.binding.content_digest == first.arena.binding.content_digest
    assert second.arena.candidate_config_digest == first.arena.candidate_config_digest
    assert second.config.content_digest == first.config.content_digest


def test_config_rejects_risk_entities_outside_fixed_bowl_contract(tmp_path: Path) -> None:
    wrong_risk = _risk(bowl_entity_id="another-bowl")

    with pytest.raises(ValueError, match="risk provider entities"):
        _swarm_config(tmp_path, risk=wrong_risk)


def test_config_rejects_risk_policy_drift_from_arena(tmp_path: Path) -> None:
    protocol = _protocol()
    admitted_risk = _risk(policy=RiskPolicy(max_candidates=3))
    changed_risk = _risk(
        policy=RiskPolicy(max_candidates=3, high_risk_score=0.9)
    )
    base = _base_config(tmp_path, protocol=protocol)
    arena = BowlCorrectionArenaFactoryConfig.from_protocol(
        protocol_config=protocol,
        risk_config=admitted_risk,
        planning=_planning(),
    )

    with pytest.raises(ValueError, match="share one exact RiskPolicy"):
        RiskAdaptiveBowlPlaceApplicationConfig(
            base=base,
            risk=changed_risk,
            arena=arena,
        )


def test_config_rejects_planning_frame_and_robot_model_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="planning frame"):
        _swarm_config(tmp_path, planning=_planning(expected_frame="map"))

    with pytest.raises(ValueError, match="robot model pin"):
        _swarm_config(
            tmp_path,
            planning=_planning(robot_model_digest=_digest("another-robot-model")),
        )


@pytest.mark.parametrize(
    "planning",
    [
        _planning(tcp_frame_id="another_tcp"),
        _planning(planner_backend="another_planner"),
    ],
)
def test_build_rejects_motion_tcp_or_planner_split_brain(
    tmp_path: Path,
    planning: BowlMotionPlanningPin,
) -> None:
    _seed_pick_handoff(tmp_path)
    config = _swarm_config(tmp_path, planning=planning)

    with pytest.raises(BowlSwarmApplicationAssemblyError, match="differs"):
        _build(config)

    assert not (tmp_path / "run_identity.v2").exists()


def test_config_rejects_unpinned_or_mismatched_candidate_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model differs"):
        _swarm_config(
            tmp_path,
            candidate_models=("specialized-small-model", "", ""),
        )

    wrong_pin = ModelPin(
        model_id="wrong-model",
        provider_id="model-provider",
        weights_digest=_digest("wrong-weights"),
        generation_config_digest=_digest("wrong-generation-config"),
        tokenizer_digest=_digest("wrong-tokenizer"),
    )
    with pytest.raises(ValueError, match="model differs"):
        _swarm_config(
            tmp_path,
            candidate_models=("specialized-small-model", "", ""),
            candidate_model_pins={"direct": wrong_pin},
        )


def test_heterogeneous_small_model_is_pinned_and_requires_per_actor_policy(
    tmp_path: Path,
) -> None:
    small_pin = ModelPin(
        model_id="specialized-small-model",
        provider_id="small-model-provider",
        weights_digest=_digest("small-model-weights"),
        generation_config_digest=_digest("small-model-generation-config"),
        tokenizer_digest=_digest("small-model-tokenizer"),
    )
    config = _swarm_config(
        tmp_path,
        candidate_models=(small_pin.model_id, "", ""),
        candidate_model_pins={"direct": small_pin},
    )
    _seed_pick_handoff(tmp_path)
    action, observation = _bindings(
        config_digest=config.arena.planning.robot_config_digest
    )

    with pytest.raises(BowlSwarmApplicationAssemblyError, match="per-actor"):
        _build(config, action=action, observation=observation)

    assembled = build_risk_adaptive_bowl_place_application(
        config=config,
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy_factory=lambda *_args: _Policy(),
    )

    direct = next(
        candidate
        for candidate in assembled.arena.binding.candidates
        if candidate.candidate_id == "direct"
    )
    assert direct.profile.model == small_pin.model_id
    assert assembled.manifest.metadata["arena_candidate_model_pins"]["direct"] == (
        small_pin.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("coding_provider_id", "risk_provider_id"),
    [
        ("bowl_correction_arena_coding", MOTION_RISK_PROVIDER_ID),
        ("skill_coding", "bowl_place"),
    ],
)
def test_config_rejects_provider_id_collisions(
    tmp_path: Path,
    coding_provider_id: str,
    risk_provider_id: str,
) -> None:
    with pytest.raises(ValueError, match="provider IDs must be distinct"):
        _swarm_config(
            tmp_path,
            coding_provider_id=coding_provider_id,
            risk_provider_id=risk_provider_id,
        )


def test_config_rejects_one_round_quota_for_multi_round_loop(tmp_path: Path) -> None:
    protocol = _protocol(iterations=2, candidates=3)

    with pytest.raises(ValueError, match="full durable candidate quota"):
        _swarm_config(
            tmp_path,
            protocol=protocol,
            total_candidate_budget_limit=protocol.max_arena_candidates,
        )


def test_config_rejects_run_candidate_budget_below_iterations_times_k(
    tmp_path: Path,
) -> None:
    protocol = _protocol(iterations=2, candidates=3)
    budgets = RunBudgets(
        max_model_calls=100_000,
        max_tokens=1_000_000_000,
        max_wall_time_s=1_000_000.0,
        max_physical_actions=100_000,
        max_shadow_rollouts=100_000,
        max_candidates=5,
        max_recoveries=10,
    )

    with pytest.raises(ValueError, match="max_candidates"):
        _swarm_config(tmp_path, protocol=protocol, budgets=budgets)


def test_build_rejects_authoritative_arm_snapshot_config_drift(
    tmp_path: Path,
) -> None:
    _seed_pick_handoff(tmp_path)
    config = _swarm_config(tmp_path)
    action, observation = _bindings(config_digest=_digest("wrong-runtime-config"))

    with pytest.raises(BowlSwarmApplicationAssemblyError, match="config digest"):
        _build(config, action=action, observation=observation)

    assert not (tmp_path / "run_identity.v2").exists()


def test_build_rejects_missing_durable_pick_handoff_without_forging_state(
    tmp_path: Path,
) -> None:
    config = _swarm_config(tmp_path)

    with pytest.raises(BowlPlaceHandoffError, match="existing durable reducer"):
        _build(config)

    assert not (tmp_path / "embodied_state_events.jsonl").exists()
    assert not (tmp_path / "run_identity.v2").exists()
    assert not (tmp_path / config.base.skill_library_subdirectory).exists()


def test_enabled_preview_rejects_runtime_provenance_spoof_before_admission(
    tmp_path: Path,
) -> None:
    _seed_pick_handoff(tmp_path)
    preview, admitted = _preview_pin()
    config = _swarm_config(tmp_path, preview=preview)
    spoofed = BowlArenaPreviewRuntimeProvenance(
        renderer=admitted.renderer,
        geometry=BowlArenaComponentProvenance(
            component_id=admitted.geometry.component_id,
            implementation_digest=_digest("spoofed-geometry-wheel"),
            configuration_digest=admitted.geometry.configuration_digest,
            version=admitted.geometry.version,
        ),
    )

    with pytest.raises(BowlArenaFactoryError, match="runtime preview provenance"):
        _build(
            config,
            preview_geometry=_Geometry(),
            preview_runtime_provenance=spoofed,
        )

    assert not (tmp_path / "run_identity.v2").exists()


def test_enabled_preview_is_constructed_and_manifest_pinned(tmp_path: Path) -> None:
    _seed_pick_handoff(tmp_path)
    preview, provenance = _preview_pin()
    config = _swarm_config(tmp_path, preview=preview)

    assembled = _build(
        config,
        preview_geometry=_Geometry(),
        preview_runtime_provenance=provenance,
    )

    assert assembled.arena.renderer is not None
    assert assembled.arena.preview_runtime_provenance == provenance
    assert assembled.manifest.metadata["arena_preview_provenance"] == (
        provenance.model_dump(mode="json")
    )
    assert (tmp_path / config.preview_artifacts_subdirectory).is_dir()


def test_enabled_preview_rejects_missing_runtime_provenance(tmp_path: Path) -> None:
    _seed_pick_handoff(tmp_path)
    preview, _admitted = _preview_pin()
    config = _swarm_config(tmp_path, preview=preview)

    with pytest.raises(BowlArenaFactoryError, match="enabled preview requires"):
        _build(config, preview_geometry=_Geometry())

    assert not (tmp_path / "run_identity.v2").exists()
