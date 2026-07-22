"""Production contract tests for the Arena-to-SkillCoding bridge."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from robomex.core.coder import ScriptedCodePolicy
from robomex.data import EpisodeDataPlane, ResolvedArtifactRef, core_schema_registry
from robomex.orchestration.actors import (
    ActorConflictError,
    ActorIsolation,
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InvocationSpec,
    WorkspaceMode,
)
from robomex.orchestration.arena import (
    ARENA_RUNTIME_CONTEXT_METADATA_KEY,
    ArenaBinding,
    ArenaCandidateSpec,
    ArenaConsumptionLedger,
    ArenaContext,
    ArenaHypothesisConfig,
    ArenaPolicy,
    ArenaPolicyError,
    ArenaPreviewConfig,
    ArenaRuntimeContextV1,
    BasicHypothesisGate,
    CheckStatus,
    RiskInputs,
    RiskPolicy,
    RiskReport,
    RuntimeArenaContextGuard,
    RuntimeMotionPromotionAuthority,
    ShadowBackendRegistry,
    SwarmArena,
)
from robomex.orchestration.arena_coding_provider import (
    ArenaCodingAgentProvider,
    ArenaCodingBridgeContractError,
    MotionPreviewFrame,
    MotionPreviewResult,
)
from robomex.orchestration.coding_provider import SkillCodingAgentProvider
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    install_runtime_schemas,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    ExecutionPolicy,
    FeasibilityStatus,
    JointPath,
    MotionPlan,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ControlOutcome
from robomex.test.test_coding_agent_provider_v2 import (
    _authoring_responses,
    _library,
    _NamespaceExecutor,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _snapshot(*, robot_revision: int = 3, world_id: str = "live_world") -> AdmissionSnapshot:
    return AdmissionSnapshot(
        world_id=world_id,
        resource_id="arm",
        robot_revision=robot_revision,
        scene_revision=7,
        attachment_revision=2,
        config_revision=5,
        joint_names=("j1", "j2", "j3"),
        joint_positions_rad=(0.0, 0.0, 0.0),
        config_digest=_digest("a"),
        collision_world_digest=_digest("b"),
        attachment_status="verified_held",
        controller_state="ready",
        captured_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def _plan(
    snapshot: AdmissionSnapshot,
    *,
    tcp_frame_id: str = "tool0",
    plan_kind: str = "bounded_correction",
    planner_backend: str = "curobo",
    endpoint: tuple[float, float, float] = (3.0, 4.0, 0.0),
) -> MotionPlan:
    return MotionPlan(
        plan_id="coding-authored-plan",
        plan_kind=plan_kind,
        tcp_frame_id=tcp_frame_id,
        planner_backend=planner_backend,
        robot_model_digest=_digest("c"),
        expected_snapshot=snapshot,
        max_start_deviation_rad=0.02,
        possibly_affected_revisions=("robot.arm", "scene", "attachment"),
        motion=JointPath(
            joint_names=snapshot.joint_names,
            positions_rad=(snapshot.joint_positions_rad, endpoint),
            execution_policy=ExecutionPolicy(subsample=1, timeout_s=5.0),
        ),
    )


def _success_result(plan: MotionPlan, *, lineage: tuple[ResolvedArtifactRef, ...] = ()):
    return ActivationExecutionResult(
        outcome=ControlOutcome.SUCCESS,
        artifacts=(
            ArtifactEmission(
                port="motion_plan",
                schema_id="robomex.motion_plan.v2",
                payload=plan.model_dump(mode="json"),
                lineage=lineage,
            ),
        ),
        reason="sealed plan authored",
    )


class _DelegateProvider:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.bound_planes: list[EpisodeDataPlane] = []
        self.calls: list[tuple[str, str]] = []
        self.invocation_specs: list[InvocationSpec] = []

    def bind_episode_data_plane(self, data_plane: EpisodeDataPlane) -> None:
        self.bound_planes.append(data_plane)

    def spawn(self, profile: ActorProfile, isolation: ActorIsolation) -> dict[str, Any]:
        self.calls.append(("spawn", isolation.owner_actor_id))
        return {"profile": profile, "isolation": isolation, "suspended": False}

    def invoke(self, runtime: dict[str, Any], spec: InvocationSpec) -> Any:
        self.calls.append(("invoke", runtime["isolation"].owner_actor_id))
        self.invocation_specs.append(spec)
        return self.result(spec) if callable(self.result) else self.result

    def suspend(self, runtime: dict[str, Any]) -> None:
        runtime["suspended"] = True
        self.calls.append(("suspend", runtime["isolation"].owner_actor_id))

    def resume(self, runtime: dict[str, Any]) -> None:
        runtime["suspended"] = False
        self.calls.append(("resume", runtime["isolation"].owner_actor_id))

    def retire(self, runtime: dict[str, Any]) -> None:
        self.calls.append(("retire", runtime["isolation"].owner_actor_id))


class _CapturingProvider:
    """Transparent AgentProvider used to inspect the Arena-authored outer spec."""

    def __init__(self, inner: ArenaCodingAgentProvider) -> None:
        self.inner = inner
        self.invocation_specs: list[InvocationSpec] = []

    def spawn(self, profile: ActorProfile, isolation: ActorIsolation):
        return self.inner.spawn(profile, isolation)

    def invoke(self, runtime: Any, spec: InvocationSpec):
        self.invocation_specs.append(spec)
        return self.inner.invoke(runtime, spec)

    def suspend(self, runtime: Any) -> None:
        self.inner.suspend(runtime)

    def resume(self, runtime: Any) -> None:
        self.inner.resume(runtime)

    def retire(self, runtime: Any) -> None:
        self.inner.retire(runtime)


class _PreviewRenderer:
    renderer_id = "pointcloud-preview-v1"

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def render(
        self,
        *,
        plan: MotionPlan,
        context: ArenaRuntimeContextV1,
    ) -> MotionPreviewResult:
        self.calls.append((plan.content_digest, context.candidate_id))
        if self.failure is not None:
            raise self.failure
        return MotionPreviewResult(
            frames=(
                MotionPreviewFrame(
                    view_id="camera_front",
                    media_type="application/x.robomex.pointcloud-preview+json",
                    media_digest=_digest("d"),
                    payload={"camera": "front", "waypoints": 2},
                ),
                MotionPreviewFrame(
                    view_id="camera_side",
                    media_type="application/x.robomex.pointcloud-preview+json",
                    media_digest=_digest("e"),
                    payload={"camera": "side", "waypoints": 2},
                ),
            ),
            terminal_position_m=(0.11, 0.22, 0.33),
        )


class _PassingChecker:
    def certify(self, spec: MotionPlan, snapshot: AdmissionSnapshot):
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="arena-bridge-test-checker",
            checks={
                "kinematic_feasibility": FeasibilityStatus.PASS,
                "collision": FeasibilityStatus.PASS,
                "joint_limits": FeasibilityStatus.PASS,
            },
        )


def _plane(path: Path) -> tuple[EpisodeDataPlane, AdmissionSnapshot, Any]:
    plane = EpisodeDataPlane(
        path,
        episode_id="episode_arena_bridge",
        schema_registry=install_runtime_schemas(core_schema_registry()),
        strict_schema_prefixes=("robomex.",),
    )
    plane.open_workflow("workflow_1")
    snapshot = _snapshot()
    snapshot_record = plane.publish_once(
        workflow_id="workflow_1",
        activation_id="observe",
        attempt=1,
        port="snapshot",
        schema="robomex.admission_snapshot.v1",
        payload=snapshot.model_dump(mode="json"),
    )
    return plane, snapshot, snapshot_record


def _profile() -> ActorProfile:
    return ActorProfile(
        profile_id="arena-motion-author",
        provider_id="arena_coding",
        runner_kind="coding_worker",
        lifecycle=ActorLifecycle.EPHEMERAL,
        metadata={
            "strict_runtime_context": True,
            "node_config_v1": (
                (
                    "arena_motion_node",
                    '{"plan_kind":"bounded_correction","planner_backend":"curobo",'
                    '"tcp_frame_id":"tool0"}',
                ),
            ),
        },
    )


def _context(
    snapshot_ref: ResolvedArtifactRef,
    *,
    candidate_id: str = "candidate_a",
    strategy: str = "wide-clearance",
    run_id: str = "arena_run_1",
    utility: float = 7.5,
    preview: ArenaPreviewConfig | None = None,
    required_tcp_frame_id: str | None = "tool0",
) -> ArenaRuntimeContextV1:
    return ArenaRuntimeContextV1(
        episode_id="episode_arena_bridge",
        workflow_id="workflow_1",
        graph_id="place-bowl",
        graph_revision=4,
        graph_digest=_digest("f"),
        command_attempt=2,
        slot_id="transport-proposals",
        arena_run_id=run_id,
        candidate_id=candidate_id,
        strategy=strategy,
        snapshot_ref=snapshot_ref,
        expected_frame="world",
        world_id="live_world",
        resource_id="arm",
        robot_model_digest=_digest("c"),
        config_digest=_digest("a"),
        coding_activation_id="arena_motion_node",
        coding_node_params={
            "plan_kind": "bounded_correction",
            "planner_backend": "curobo",
            "tcp_frame_id": "tool0",
        },
        hypothesis_config=ArenaHypothesisConfig(
            expected_effect="move held bowl over plate",
            preconditions=("attachment_verified", "target_visible"),
            estimated_risk=0.2,
            utility=utility,
            clearance_m=0.04,
            required_tcp_frame_id=required_tcp_frame_id,
        ),
        preview_config=preview or ArenaPreviewConfig(),
    )


def _spec(context: ArenaRuntimeContextV1) -> InvocationSpec:
    actor_id = f"{context.arena_run_id}-{context.candidate_id}"
    return InvocationSpec(
        invocation_id=f"{actor_id}-invoke",
        idempotency_key=actor_id,
        objective="Author one exact joint-space motion plan.",
        inputs={"snapshot": context.snapshot_ref.to_mapping()},
        output_contract={"motion_plan": "robomex.motion_plan.v2"},
        metadata={
            "prompt_variant": "collision-aware",
            ARENA_RUNTIME_CONTEXT_METADATA_KEY: context.model_dump(mode="json"),
        },
    )


def _isolation(context: ArenaRuntimeContextV1) -> ActorIsolation:
    actor_id = f"{context.arena_run_id}-{context.candidate_id}"
    return ActorIsolation(
        owner_actor_id=actor_id,
        namespace_id=f"episode/{actor_id}",
        workspace_id=f"workspace/{actor_id}",
        workspace_mode=WorkspaceMode.ISOLATED,
    )


def _invoke_direct(
    *,
    plane: EpisodeDataPlane,
    context: ArenaRuntimeContextV1,
    delegate: _DelegateProvider,
    renderer: _PreviewRenderer | None = None,
):
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane, renderer=renderer)
    runtime = bridge.spawn(_profile(), _isolation(context))
    return bridge, runtime, bridge.invoke(runtime, _spec(context))


def test_full_swarm_arena_skill_coding_bridge_publishes_and_promotes(
    tmp_path: Path,
) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    plan = _plan(snapshot)
    delegate = _DelegateProvider(_success_result(plan, lineage=(snapshot_record.ref,)))
    renderer = _PreviewRenderer()
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane, renderer=renderer)
    outer_capture = _CapturingProvider(bridge)
    registry = ActorRegistry(
        {"arena_coding": outer_capture},
        namespace_root="episode_arena_bridge",
        workspace_root=tmp_path / "actors",
    )
    arena = SwarmArena(
        registry,
        promotion_authority=RuntimeMotionPromotionAuthority(
            episode_id=plane.episode_id,
            artifacts=plane,
            snapshot_providers={("live_world", "arm"): lambda *_: snapshot},
            feasibility_checkers={("live_world", "arm"): _PassingChecker()},
        ),
        context_guard=RuntimeArenaContextGuard(
            episode_id=plane.episode_id,
            current_revision=lambda context: (context.graph_id, context.graph_revision),
        ),
        consumption_ledger=ArenaConsumptionLedger(),
        shadow_backends=ShadowBackendRegistry(),
        gates=(
            BasicHypothesisGate(
                expected_frame="world",
                expected_snapshot_ref=snapshot_record.ref,
            ),
        ),
        policy=ArenaPolicy(max_candidates=1),
    )
    candidate = ArenaCandidateSpec(
        candidate_id="candidate_a",
        strategy="wide-clearance",
        profile=_profile(),
        objective="Author one exact joint-space motion plan.",
        inputs={"snapshot": snapshot_record.ref.to_mapping()},
        output_contract={"motion_plan": "robomex.motion_plan.v2"},
        invocation_metadata={"prompt_variant": "collision-aware"},
        coding_activation_id="arena_motion_node",
        hypothesis_config=ArenaHypothesisConfig(
            expected_effect="move held bowl over plate",
            preconditions=("attachment_verified", "target_visible"),
            estimated_risk=0.2,
            utility=7.5,
            clearance_m=0.04,
            required_tcp_frame_id="tool0",
        ),
        preview_config=ArenaPreviewConfig(
            enabled=True,
            renderer_id=renderer.renderer_id,
            failure_mode="fail_closed",
        ),
    )
    arena_context = ArenaContext(
        arena_run_id="arena_run_1",
        episode_id=plane.episode_id,
        workflow_id="workflow_1",
        graph_id="place-bowl",
        graph_revision=4,
        graph_digest=_digest("f"),
        command_attempt=2,
        slot_id="transport-proposals",
        snapshot_ref=snapshot_record.ref,
        expected_frame="world",
        world_id="live_world",
        resource_id="arm",
        robot_model_digest=_digest("c"),
        config_digest=_digest("a"),
        candidate_budget_id="transport-candidate-budget",
        candidate_budget_limit=1,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.95,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.04,
            held_pose_uncertainty_m=0.002,
        ),
        RiskPolicy(max_candidates=1),
    )

    result = arena.run(
        context=arena_context,
        risk_report=risk,
        candidates=(candidate,),
        candidate_budget_remaining=1,
    )

    assert result.selected_candidate_id == "candidate_a"
    hypothesis = result.selected_hypothesis
    assert hypothesis is not None
    assert hypothesis.candidate_id == "candidate_a"
    assert hypothesis.strategy == "wide-clearance"
    assert hypothesis.utility == 7.5
    assert hypothesis.estimated_risk == 0.2
    assert hypothesis.path_length == 5.0
    assert hypothesis.terminal_position_m == (0.11, 0.22, 0.33)
    assert len(hypothesis.render_refs) == 2
    assert hypothesis.evidence_refs == (snapshot_record.ref,)
    assert plane.resolve(hypothesis.plan_ref).payload == plan.model_dump(mode="json")
    for ref in hypothesis.render_refs:
        preview = plane.resolve(ref)
        assert preview.schema == "robomex.motion_preview.v1"
        assert preview.payload["candidate_id"] == "candidate_a"
        assert preview.payload["plan_digest"] == plan.content_digest
        assert preview.payload["terminal_position_m"] == [0.11, 0.22, 0.33]
    assert [call[0] for call in delegate.calls] == ["spawn", "invoke", "retire"]
    assert renderer.calls == [(plan.content_digest, "candidate_a")]

    outer = outer_capture.invocation_specs[0]
    trusted = ArenaRuntimeContextV1.model_validate(
        outer.metadata[ARENA_RUNTIME_CONTEXT_METADATA_KEY]
    )
    assert trusted.episode_id == plane.episode_id
    assert trusted.workflow_id == "workflow_1"
    assert trusted.graph_id == "place-bowl"
    assert trusted.graph_revision == 4
    assert trusted.graph_digest == _digest("f")
    assert trusted.command_attempt == 2
    assert trusted.slot_id == "transport-proposals"
    assert trusted.arena_run_id == "arena_run_1"
    assert trusted.candidate_id == "candidate_a"
    assert trusted.strategy == "wide-clearance"
    assert trusted.snapshot_ref == snapshot_record.ref
    changed_outer = replace(
        outer,
        metadata={
            **dict(outer.metadata),
            ARENA_RUNTIME_CONTEXT_METADATA_KEY: {
                **trusted.model_dump(mode="json"),
                "command_attempt": 3,
            },
        },
    )
    assert changed_outer.fingerprint() != outer.fingerprint()

    captured = delegate.invocation_specs[0]
    assert ARENA_RUNTIME_CONTEXT_METADATA_KEY not in captured.metadata
    assert captured.metadata["episode_id"] == plane.episode_id
    assert captured.metadata["workflow_id"] == "workflow_1"
    assert captured.metadata["graph_id"] == "place-bowl"
    assert captured.metadata["graph_revision"] == 4
    assert captured.metadata["graph_digest"] == _digest("f")
    assert captured.metadata["attempt"] == 2
    assert captured.metadata["activation_id"] == "arena_motion_node"
    assert captured.metadata["node_params"] == {
        "plan_kind": "bounded_correction",
        "planner_backend": "curobo",
        "tcp_frame_id": "tool0",
    }
    assert captured.metadata["prompt_variant"] == "collision-aware"
    changed = replace(
        captured,
        metadata={
            **dict(captured.metadata),
            "attempt": 99,
        },
    )
    assert changed.fingerprint() != captured.fingerprint()


def test_committed_replay_after_restart_reuses_no_model_render_or_artifact(
    tmp_path: Path,
) -> None:
    episode_root = tmp_path / "episode"
    plane, snapshot, snapshot_record = _plane(episode_root)
    context = _context(
        snapshot_record.ref,
        preview=ArenaPreviewConfig(
            enabled=True,
            renderer_id="pointcloud-preview-v1",
            failure_mode="fail_closed",
        ),
    )
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    renderer = _PreviewRenderer()
    _, _, first = _invoke_direct(
        plane=plane,
        context=context,
        delegate=delegate,
        renderer=renderer,
    )
    artifact_count = len(plane.artifacts)

    reopened = EpisodeDataPlane(
        episode_root,
        episode_id=plane.episode_id,
        schema_registry=install_runtime_schemas(core_schema_registry()),
        strict_schema_prefixes=("robomex.",),
    )
    replay_delegate = _DelegateProvider(RuntimeError("model must not run during replay"))
    replay_renderer = _PreviewRenderer(failure=RuntimeError("renderer must not rerun"))
    _, _, second = _invoke_direct(
        plane=reopened,
        context=context,
        delegate=replay_delegate,
        renderer=replay_renderer,
    )

    assert second == first
    assert [call[0] for call in replay_delegate.calls] == ["spawn"]
    assert replay_delegate.invocation_specs == []
    assert replay_renderer.calls == []
    assert len(reopened.artifacts) == artifact_count


def test_replay_rejects_manifest_scoring_rebind_before_model_entry(tmp_path: Path) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    context = _context(snapshot_record.ref, utility=1.0)
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    _invoke_direct(plane=plane, context=context, delegate=delegate)

    rebound = _context(snapshot_record.ref, utility=99.0)
    replay_delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    bridge = ArenaCodingAgentProvider(replay_delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(rebound))
    with pytest.raises(ActorConflictError, match="rebind"):
        bridge.invoke(runtime, _spec(rebound))
    assert [call[0] for call in replay_delegate.calls] == ["spawn"]


@pytest.mark.parametrize(
    "result",
    [
        ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="motion_plan",
                    schema_id="robomex.action_hypothesis.v1",
                    payload={},
                ),
            ),
        ),
        ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="motion_plan",
                    schema_id="robomex.motion_plan.v2",
                    payload={},
                ),
                ArtifactEmission(
                    port="second_plan",
                    schema_id="robomex.motion_plan.v2",
                    payload={},
                ),
            ),
        ),
    ],
    ids=("wrong_schema", "multiple_emissions"),
)
def test_bridge_rejects_wrong_schema_or_multiple_emissions(
    tmp_path: Path,
    result: ActivationExecutionResult,
) -> None:
    plane, _, snapshot_record = _plane(tmp_path / "episode")
    context = _context(snapshot_record.ref)
    delegate = _DelegateProvider(result)
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    with pytest.raises(ArenaCodingBridgeContractError):
        bridge.invoke(runtime, _spec(context))
    assert len(plane.artifacts) == 1


@pytest.mark.parametrize(
    ("plan_factory", "match"),
    [
        (lambda: _plan(_snapshot(robot_revision=99)), "expected_snapshot"),
        (lambda: _plan(_snapshot(), tcp_frame_id="wrong_tcp"), "tcp_frame_id"),
        (lambda: _plan(_snapshot(), plan_kind="unbounded"), "plan_kind"),
        (
            lambda: _plan(_snapshot(), planner_backend="model_chosen_backend"),
            "planner_backend",
        ),
        (lambda: _plan(_snapshot(world_id="another_world")), "expected_snapshot, world_id"),
    ],
    ids=("wrong_snapshot", "wrong_frame", "wrong_plan_kind", "wrong_backend", "wrong_world"),
)
def test_bridge_rejects_plan_outside_trusted_physical_context(
    tmp_path: Path,
    plan_factory,
    match: str,
) -> None:
    plane, _, snapshot_record = _plane(tmp_path / "episode")
    context = _context(snapshot_record.ref)
    delegate = _DelegateProvider(_success_result(plan_factory()))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    with pytest.raises(ArenaCodingBridgeContractError, match=match):
        bridge.invoke(runtime, _spec(context))
    assert len(plane.artifacts) == 1


def test_model_cannot_smuggle_arena_identity_inside_motion_plan(tmp_path: Path) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    payload = _plan(snapshot).model_dump(mode="json")
    payload["candidate_id"] = "attacker_selected_candidate"
    payload["utility"] = 1_000_000
    delegate = _DelegateProvider(
        ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="motion_plan",
                    schema_id="robomex.motion_plan.v2",
                    payload=payload,
                ),
            ),
        )
    )
    context = _context(snapshot_record.ref)
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    with pytest.raises(ArenaCodingBridgeContractError, match="payload is invalid"):
        bridge.invoke(runtime, _spec(context))
    assert len(plane.artifacts) == 1


@pytest.mark.parametrize("reserved_key", [ARENA_RUNTIME_CONTEXT_METADATA_KEY, "candidate_id"])
def test_candidate_template_cannot_override_runtime_metadata(reserved_key: str) -> None:
    with pytest.raises(ValueError, match="cannot override runtime context"):
        ArenaCandidateSpec(
            candidate_id="candidate_a",
            strategy="wide-clearance",
            profile=_profile(),
            objective="author plan",
            invocation_metadata={reserved_key: "spoofed"},
        )


def test_candidate_runs_are_artifact_and_idempotency_isolated(tmp_path: Path) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    hypotheses = []
    for candidate_id, utility in (("candidate_a", 1.0), ("candidate_b", 2.0)):
        context = _context(
            snapshot_record.ref,
            candidate_id=candidate_id,
            strategy=f"strategy-{candidate_id}",
            utility=utility,
        )
        runtime = bridge.spawn(_profile(), _isolation(context))
        hypotheses.append(bridge.invoke(runtime, _spec(context)))

    first, second = hypotheses
    assert first.candidate_id == "candidate_a"
    assert second.candidate_id == "candidate_b"
    assert first.strategy == "strategy-candidate_a"
    assert second.strategy == "strategy-candidate_b"
    assert first.utility == 1.0
    assert second.utility == 2.0
    assert first.plan_ref.artifact_id != second.plan_ref.artifact_id
    assert first.plan_ref.content_digest == second.plan_ref.content_digest
    assert len([call for call in delegate.calls if call[0] == "invoke"]) == 2
    assert len(plane.artifacts) == 3


@pytest.mark.parametrize(
    ("failure_mode", "raises"),
    [("fail_closed", True), ("omit", False)],
)
def test_preview_failure_policy_is_manifest_pinned(
    tmp_path: Path,
    failure_mode: str,
    raises: bool,
) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / failure_mode)
    context = _context(
        snapshot_record.ref,
        preview=ArenaPreviewConfig(
            enabled=True,
            renderer_id="pointcloud-preview-v1",
            failure_mode=failure_mode,
        ),
    )
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    renderer = _PreviewRenderer(failure=RuntimeError("render unavailable"))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane, renderer=renderer)
    runtime = bridge.spawn(_profile(), _isolation(context))

    if raises:
        with pytest.raises(ArenaCodingBridgeContractError, match="preview failed"):
            bridge.invoke(runtime, _spec(context))
        assert len(plane.artifacts) == 2  # snapshot + durably published motion plan
    else:
        hypothesis = bridge.invoke(runtime, _spec(context))
        assert hypothesis.render_refs == ()
        assert hypothesis.terminal_position_m is None
        assert hypothesis.metadata["terminal_position_source"] == "unavailable"


def test_binding_injects_exact_manifest_pinned_context_refs() -> None:
    candidate = ArenaCandidateSpec(
        candidate_id="candidate_a",
        strategy="wide-clearance",
        profile=_profile(),
        objective="author plan",
        output_contract={"motion_plan": "robomex.motion_plan.v2"},
        coding_activation_id="arena_motion_node",
    )
    binding = ArenaBinding(
        binding_id="bowl_transport_arena",
        candidates=(candidate,),
        expected_frame="world",
        robot_model_digest=_digest("c"),
        candidate_budget_limit=1,
        policy=ArenaPolicy(max_candidates=1),
        context_input_schemas={
            "attachment_evidence": "robomex.attachment_evidence.v1",
            "observation": "robomex.observation_snapshot.v1",
            "servo_decision": "robomex.visual_servo_decision.v1",
        },
    )
    snapshot_ref = ResolvedArtifactRef("art:snapshot", _digest("1"))
    risk_ref = ResolvedArtifactRef("art:risk", _digest("2"))
    context_refs = {
        "attachment_evidence": ResolvedArtifactRef("art:attachment", _digest("3")),
        "observation": ResolvedArtifactRef("art:observation", _digest("4")),
        "servo_decision": ResolvedArtifactRef("art:servo", _digest("5")),
    }

    materialized = binding.candidates_for(
        snapshot_ref=snapshot_ref,
        risk_ref=risk_ref,
        context_refs=context_refs,
    )[0]

    assert materialized.inputs["snapshot"] == snapshot_ref.to_mapping()
    assert materialized.inputs["risk"] == risk_ref.to_mapping()
    for name, ref in context_refs.items():
        assert materialized.inputs[name] == ref.to_mapping()
    assert dict(materialized.context_input_schemas) == dict(
        binding.context_input_schemas
    )
    changed_binding = ArenaBinding(
        binding_id="bowl_transport_arena",
        candidates=(candidate,),
        expected_frame="world",
        robot_model_digest=_digest("c"),
        candidate_budget_limit=1,
        policy=ArenaPolicy(max_candidates=1),
        context_input_schemas={
            **dict(binding.context_input_schemas),
            "servo_decision": "robomex.visual_servo_decision.v2",
        },
    )
    assert changed_binding.content_digest != binding.content_digest
    with pytest.raises(ArenaPolicyError, match="exactly match"):
        binding.candidates_for(
            snapshot_ref=snapshot_ref,
            risk_ref=risk_ref,
            context_refs={"observation": context_refs["observation"]},
        )
    with pytest.raises(ArenaPolicyError, match="unknown"):
        binding.candidates_for(
            snapshot_ref=snapshot_ref,
            risk_ref=risk_ref,
            context_refs={**context_refs, "unadmitted": context_refs["observation"]},
        )

    overriding = replace(
        candidate,
        inputs={"observation": context_refs["observation"].to_mapping()},
    )
    overriding_binding = replace(binding, candidates=(overriding,))
    with pytest.raises(ArenaPolicyError, match="cannot override"):
        overriding_binding.candidates_for(
            snapshot_ref=snapshot_ref,
            risk_ref=risk_ref,
            context_refs=context_refs,
        )


def test_bridge_resolves_context_refs_by_exact_identity_and_schema(tmp_path: Path) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.95,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.04,
            held_pose_uncertainty_m=0.002,
        ),
        RiskPolicy(max_candidates=1),
    )
    context_record = plane.publish_once(
        workflow_id="workflow_1",
        activation_id="servo_evidence",
        attempt=1,
        port="servo_decision",
        schema="robomex.risk_report.v1",
        payload=risk.model_dump(mode="json"),
    )
    context_payload = _context(snapshot_record.ref).model_dump(mode="json")
    context_payload.update(
        {
            "context_input_schemas": {
                "servo_decision": "robomex.risk_report.v1"
            },
            "context_input_refs": {
                "servo_decision": context_record.ref.to_mapping()
            },
        }
    )
    context = ArenaRuntimeContextV1.model_validate(context_payload)
    spec = replace(
        _spec(context),
        inputs={
            "snapshot": snapshot_record.ref.to_mapping(),
            "servo_decision": context_record.ref.to_mapping(),
        },
    )
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    hypothesis = bridge.invoke(runtime, spec)

    assert hypothesis.candidate_id == "candidate_a"
    assert delegate.invocation_specs[0].inputs["servo_decision"] == (
        context_record.ref.to_mapping()
    )

    wrong_payload = context.model_dump(mode="json")
    wrong_payload["context_input_refs"] = {
        "servo_decision": snapshot_record.ref.to_mapping()
    }
    wrong_context = ArenaRuntimeContextV1.model_validate(wrong_payload)
    wrong_spec = replace(
        _spec(wrong_context),
        inputs={
            "snapshot": snapshot_record.ref.to_mapping(),
            "servo_decision": snapshot_record.ref.to_mapping(),
        },
    )
    wrong_delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    wrong_bridge = ArenaCodingAgentProvider(
        wrong_delegate,
        data_plane=plane,
        artifacts_root=tmp_path / "wrong-bridge",
    )
    wrong_runtime = wrong_bridge.spawn(_profile(), _isolation(wrong_context))
    with pytest.raises(ArenaCodingBridgeContractError, match="schema differs"):
        wrong_bridge.invoke(wrong_runtime, wrong_spec)
    assert [call[0] for call in wrong_delegate.calls] == ["spawn"]


def test_bridge_rejects_outer_runtime_identity_spoof_before_delegate(
    tmp_path: Path,
) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    context = _context(snapshot_record.ref)
    spoofed = replace(
        _spec(context),
        metadata={
            **dict(_spec(context).metadata),
            "attempt": 999,
            "activation_id": "attacker_node",
        },
    )
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    with pytest.raises(ArenaCodingBridgeContractError, match="inject delegate"):
        bridge.invoke(runtime, spoofed)
    assert [call[0] for call in delegate.calls] == ["spawn"]


def test_bridge_refuses_nonstrict_coding_profile(tmp_path: Path) -> None:
    plane, snapshot, _ = _plane(tmp_path / "episode")
    delegate = _DelegateProvider(_success_result(_plan(snapshot)))
    bridge = ArenaCodingAgentProvider(delegate, data_plane=plane)
    nonstrict = ActorProfile(
        profile_id="unsafe-arena-author",
        provider_id="arena_coding",
        runner_kind="coding_worker",
        lifecycle=ActorLifecycle.EPHEMERAL,
    )
    isolation = ActorIsolation(
        owner_actor_id="arena_run_1-candidate_a",
        namespace_id="episode/unsafe",
        workspace_id="workspace/unsafe",
        workspace_mode=WorkspaceMode.ISOLATED,
    )

    with pytest.raises(ArenaCodingBridgeContractError, match="strict_runtime_context"):
        bridge.spawn(nonstrict, isolation)
    assert delegate.calls == []


def test_bridge_drives_real_strict_skill_coding_provider(tmp_path: Path) -> None:
    plane, snapshot, snapshot_record = _plane(tmp_path / "episode")
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.95,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.04,
            held_pose_uncertainty_m=0.002,
        ),
        RiskPolicy(max_candidates=1),
    )
    servo_record = plane.publish_once(
        workflow_id="workflow_1",
        activation_id="servo_evidence",
        attempt=1,
        port="servo_decision",
        schema="robomex.risk_report.v1",
        payload=risk.model_dump(mode="json"),
    )
    executors: list[_NamespaceExecutor] = []

    def executor_factory(*_args: object) -> _NamespaceExecutor:
        executor = _NamespaceExecutor()
        executors.append(executor)
        return executor

    coding_provider = SkillCodingAgentProvider(
        data_plane=plane,
        executor_factory=executor_factory,
        library=_library(tmp_path / "skills"),
        policy=ScriptedCodePolicy(_authoring_responses(_plan(snapshot))),
        artifacts_root=tmp_path / "coding-provider",
    )
    raw_context = _context(snapshot_record.ref).model_dump(mode="json")
    raw_context.update(
        {
            "context_input_schemas": {
                "servo_decision": "robomex.risk_report.v1"
            },
            "context_input_refs": {
                "servo_decision": servo_record.ref.to_mapping()
            },
        }
    )
    context = ArenaRuntimeContextV1.model_validate(raw_context)
    spec = replace(
        _spec(context),
        inputs={
            "snapshot": snapshot_record.ref.to_mapping(),
            "servo_decision": servo_record.ref.to_mapping(),
        },
    )
    bridge = ArenaCodingAgentProvider(coding_provider, data_plane=plane)
    runtime = bridge.spawn(_profile(), _isolation(context))

    hypothesis = bridge.invoke(runtime, spec)

    assert hypothesis.candidate_id == "candidate_a"
    assert hypothesis.evidence_refs == (snapshot_record.ref, servo_record.ref)
    assert plane.resolve(hypothesis.plan_ref).schema == "robomex.motion_plan.v2"
    assert coding_provider.invocation_audits[-1].context_digest.startswith("sha256:")
    assert executors[0].sandbox_namespace["RUNTIME_CONTEXT_V1"] == {
        "schema_version": "robomex.coding_runtime_context.v1",
        "episode_id": plane.episode_id,
        "run_id": "workflow_1",
        "node_id": "arena_motion_node",
        "attempt": 2,
        "invocation_id": "arena_run_1-candidate_a-invoke",
        "idempotency_key": "arena_run_1-candidate_a",
        "graph_id": "place-bowl",
        "graph_revision": 4,
        "graph_digest": _digest("f"),
    }
    assert executors[0].physical_calls == []
    assert executors[0].sandbox_namespace["INPUTS"]["servo_decision"]["schema"] == (
        "robomex.risk_report.v1"
    )
