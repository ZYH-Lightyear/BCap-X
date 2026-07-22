from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timezone
from pathlib import Path

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
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    BackendProvenance,
    ObservationBackendBinding,
)
from robomex.orchestration.bowl_application import (
    BOWL_PLACE_FUNCTION_IDS,
    BOWL_PLACE_SKILL_IDS,
    BowlApplicationAssemblyError,
    BowlPlaceHandoffError,
    FeasibilityCheckerPin,
    FixedBowlPlaceApplicationConfig,
    TrustedBowlApplicationExtensions,
    build_bowl_place_contract_catalog,
    build_episode_bowl_skill_library,
    build_fixed_bowl_place_application,
    verify_bowl_place_handoff,
)
from robomex.orchestration.bowl_provider import (
    BowlPlaceProviderConfig,
    build_bowl_place_coding_profiles,
)
from robomex.orchestration.intent import EntityRef, SubgoalIntent
from robomex.orchestration.production_manifest import ProductionManifestError
from robomex.protocols.bowl_place import build_fixed_bowl_place_protocol
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
    def __init__(self, checker_id: str = "checker-main") -> None:
        self.checker_id = checker_id

    def certify(self, _spec, _snapshot):  # pragma: no cover - assembly does not execute
        raise AssertionError("physical feasibility is not called during assembly")


class _ActionBackend:
    def __init__(self, *, signals: tuple[str, ...] | None = None) -> None:
        self._descriptor = BackendDescriptor(
            backend_id="robot-backend",
            motion_interface=BackendMotionInterface.EXACT_JOINT_PATH,
            watchdog_stop_thread_safe=True,
        )
        self.signals = signals or (
            "attachment_status",
            "held_entity_visible",
            "identity_match",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            world_id=world_id,
            world_kind=WorldKind.AUTHORITATIVE,
            resource_id=resource_id,
            robot_revision=1,
            scene_revision=1,
            attachment_revision=1,
            config_revision=1,
            joint_names=("joint-1",),
            joint_positions_rad=(0.0,),
            config_digest=_digest("robot-model"),
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
        self, *, world_id: str, resource_id: str, timeout_s: float
    ) -> AdmissionSnapshot:
        del timeout_s
        return self.snapshot(world_id, resource_id)

    def monitor_telemetry_capabilities(
        self, *, world_id: str, resource_id: str
    ) -> MonitorTelemetryCapabilities:
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=self.signals,
            supported_hooks=(MonitorTelemetryHook.CONTROL,),
            cooperative_stop_guaranteed=True,
        )

    def monitor_sample(self, **_kwargs):
        return dict.fromkeys(self.signals, True)


class _Policy:
    def complete(self, _messages):  # pragma: no cover - assembly does not spawn a worker
        raise AssertionError("coding policy is not called during assembly")


class _Executor:
    def run_block(self, _block):  # pragma: no cover - assembly does not spawn a worker
        raise AssertionError("CapX executor is not called during assembly")


def _provenance(backend_id: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=backend_id,
        implementation_digest=_digest(f"{backend_id}-implementation"),
        configuration_digest=_digest(f"{backend_id}-configuration"),
        version="1.0.0",
    )


def _bindings(
    *, backend: _ActionBackend | None = None
) -> tuple[tuple[ActionBackendBinding, ...], tuple[ObservationBackendBinding, ...]]:
    action = backend or _ActionBackend()
    checker = _Checker()
    action_provenance = _provenance("robot-backend")
    action_bindings = tuple(
        ActionBackendBinding(
            world_id="authoritative",
            resource_id=resource_id,
            backend=action,
            feasibility_checker=checker,
            provenance=action_provenance,
        )
        for resource_id in ("robot.arm", "robot.gripper", "robot.controller")
    )
    observation = InMemoryObservationBackend("bowl-observation-backend")
    observation_bindings = (
        ObservationBackendBinding(
            backend=observation,
            provenance=_provenance("bowl-observation-backend"),
        ),
    )
    return action_bindings, observation_bindings


def _seed_pick_handoff(root: Path, *, status: AttachmentStatus = AttachmentStatus.ATTEMPTED):
    schemas = core_schema_registry()
    plane = EpisodeDataPlane(
        root,
        episode_id="episode-1",
        schema_registry=schemas,
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
        # The production attempted handoff is strict.  The verified-held
        # branch below is a narrow reducer-transition fixture; production
        # verification evidence is covered by the state evidence suite.
        strict_evidence=status is AttachmentStatus.ATTEMPTED,
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
    if status is AttachmentStatus.VERIFIED_HELD:
        reducer.commit(
            StateTransitionProposal.set_attachment(
                episode_id="episode-1",
                effect_id="verify-grasp-1",
                before_revision=reducer.state.revision,
                source="prior-grasp-verifier",
                evidence_refs=(evidence.ref,),
                entity_id="bowl-1",
                status=AttachmentStatus.VERIFIED_HELD,
                action_id="grasp-action-1",
                trigger=PhysicalStateTrigger.EVIDENCE,
                track_id="track-bowl-1",
            )
        )
    plane.close_workflow("prior-pick")
    return reducer


def _config(root: Path, *, budgets: RunBudgets | None = None):
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
        prompts=(PromptPin(prompt_id="bowl-coding-prompt", content_digest=_digest("prompt")),),
        budgets=budgets
        or RunBudgets(
            max_model_calls=1_000,
            max_tokens=10_000_000,
            max_wall_time_s=10_000.0,
            max_physical_actions=1_000,
            max_shadow_rollouts=0,
            max_candidates=1,
            max_recoveries=1,
        ),
        runtime_code_digest=_digest("runtime-code"),
        robot_model_digest=_digest("robot-model"),
        feasibility_checker_pins=dict.fromkeys(
            ("robot.arm", "robot.gripper", "robot.controller"), checker_pin
        ),
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def test_episode_skill_view_and_catalog_pin_exact_five_packages(tmp_path: Path) -> None:
    protocol = build_fixed_bowl_place_protocol()
    coding = build_bowl_place_coding_profiles(protocol.spec)
    library = build_episode_bowl_skill_library(tmp_path)

    catalog = build_bowl_place_contract_catalog(library, coding)

    assert tuple(sorted(skill.skill_id for skill in catalog.skills)) == BOWL_PLACE_SKILL_IDS
    assert (
        tuple(sorted(item.function_id for skill in catalog.skills for item in skill.functions))
        == BOWL_PLACE_FUNCTION_IDS
    )


@pytest.mark.parametrize("status", [AttachmentStatus.ATTEMPTED, AttachmentStatus.VERIFIED_HELD])
def test_place_handoff_accepts_durable_attempted_or_verified_held(
    tmp_path: Path, status: AttachmentStatus
) -> None:
    reducer = _seed_pick_handoff(tmp_path, status=status)

    handoff = verify_bowl_place_handoff(
        reducer,
        provider_config=BowlPlaceProviderConfig(),
    )

    assert handoff.attachment_status is status
    assert handoff.action_id == "grasp-action-1"
    assert handoff.attempted_effect_id == "attempt-grasp-1"


def test_complete_fixed_application_assembles_real_runtime_inventory(tmp_path: Path) -> None:
    _seed_pick_handoff(tmp_path)
    action, observation = _bindings()

    assembled = build_fixed_bowl_place_application(
        config=_config(tmp_path),
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy=_Policy(),
    )

    assert assembled.agent.config.application is assembled.application
    assert assembled.manifest == assembled.application.manifest
    assert len(assembled.manifest.skills) == 5
    assert len(assembled.manifest.functions) == 5
    assert assembled.manifest.metadata["robot_model_digest"] == _digest("robot-model")
    assert set(assembled.manifest.metadata["feasibility_checker_config"]["identities"]) == {
        "checker-main"
    }
    assert set(assembled.manifest.metadata["feasibility_checker_config"]["resource_bindings"]) == {
        "authoritative/robot.arm",
        "authoritative/robot.gripper",
        "authoritative/robot.controller",
    }
    for key in ("protocol", "provider", "tracking", "coding", "task", "manager", "planner"):
        assert f"{key}_config" in assembled.manifest.metadata
        assert f"{key}_config_digest" in assembled.manifest.metadata
    assert assembled.skill_library.root.is_relative_to(tmp_path)
    assert assembled.dependencies.freshness_context_provider_id == "bowl_place"


def test_operator_created_at_makes_restart_manifest_identity_stable(tmp_path: Path) -> None:
    _seed_pick_handoff(tmp_path)
    action, observation = _bindings()
    config = _config(tmp_path)

    first = build_fixed_bowl_place_application(
        config=config,
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy=_Policy(),
    )
    second = build_fixed_bowl_place_application(
        config=config,
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy=_Policy(),
    )

    assert first.manifest.created_at == config.created_at
    assert second.manifest.content_digest == first.manifest.content_digest
    assert second.application.manifest == first.application.manifest


def test_trusted_extension_metadata_is_sealed_without_overwriting_fixed_keys(
    tmp_path: Path,
) -> None:
    _seed_pick_handoff(tmp_path)
    action, observation = _bindings()
    candidate_digest = _digest("arena-candidate-config")

    assembled = build_fixed_bowl_place_application(
        config=_config(tmp_path),
        action_backends=action,
        observation_backends=observation,
        capx_executor_factory=lambda *_args: _Executor(),
        coding_policy=_Policy(),
        _trusted_extensions=TrustedBowlApplicationExtensions(
            candidate_config_digest=candidate_digest,
            manifest_metadata={
                "swarm_extension_config": {"schema_version": "test.swarm_config.v1"}
            },
        ),
    )

    assert assembled.manifest.candidate_config_digest == candidate_digest
    assert assembled.manifest.metadata["swarm_extension_config"] == {
        "schema_version": "test.swarm_config.v1"
    }

    with pytest.raises(BowlApplicationAssemblyError, match="cannot overwrite"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path),
            action_backends=action,
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
            _trusted_extensions=TrustedBowlApplicationExtensions(
                manifest_metadata={"protocol_config": {"forged": True}}
            ),
        )


def test_assembly_rejects_missing_handoff_without_forging_state(tmp_path: Path) -> None:
    action, observation = _bindings()

    with pytest.raises(BowlPlaceHandoffError, match="existing durable reducer"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path),
            action_backends=action,
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
        )

    assert not (tmp_path / "embodied_state_events.jsonl").exists()
    assert not (tmp_path / "episode_manifest.v2.json").exists()


def test_assembly_rejects_non_exact_authoritative_resource_coverage(tmp_path: Path) -> None:
    action, observation = _bindings()

    with pytest.raises(BowlApplicationAssemblyError, match="exact authoritative resource"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path),
            action_backends=action[:-1],
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
        )


def test_assembly_rejects_monitor_signal_gap_before_handoff(tmp_path: Path) -> None:
    action, observation = _bindings(
        backend=_ActionBackend(signals=("attachment_status", "held_entity_visible"))
    )

    with pytest.raises(BowlApplicationAssemblyError, match="identity_match"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path),
            action_backends=action,
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
        )


def test_assembly_rejects_budget_below_worst_case_envelope(tmp_path: Path) -> None:
    action, observation = _bindings()
    insufficient = RunBudgets(
        max_model_calls=0,
        max_tokens=0,
        max_wall_time_s=0,
        max_physical_actions=0,
        max_shadow_rollouts=0,
        max_candidates=1,
        max_recoveries=0,
    )

    with pytest.raises(ProductionManifestError, match="run budgets do not cover"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path, budgets=insufficient),
            action_backends=action,
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
        )


def test_assembly_rejects_checker_identity_drift(tmp_path: Path) -> None:
    action, observation = _bindings()
    changed = list(action)
    changed[0] = ActionBackendBinding(
        world_id=changed[0].world_id,
        resource_id=changed[0].resource_id,
        backend=changed[0].backend,
        feasibility_checker=_Checker("different-checker"),
        provenance=changed[0].provenance,
    )

    with pytest.raises(BowlApplicationAssemblyError, match="checker_id"):
        build_fixed_bowl_place_application(
            config=_config(tmp_path),
            action_backends=changed,
            observation_backends=observation,
            capx_executor_factory=lambda *_args: _Executor(),
            coding_policy=_Policy(),
        )
