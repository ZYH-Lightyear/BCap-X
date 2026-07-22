from __future__ import annotations

from datetime import datetime, timezone

import pytest

from robomex.elastic import (
    ActivationSpec,
    ArtifactBinding,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    ExternalBinding,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
    IsolationPolicy,
)
from robomex.orchestration.arena import (
    ArenaBinding,
    ArenaCandidateSpec,
    ArenaContext,
    ArenaPolicy,
    CheckStatus,
    RegisteredShadowBackend,
    RiskInputs,
    RiskPolicy,
    RiskReport,
    ShadowBackendRegistry,
)
from robomex.orchestration.bootstrap import (
    DependencyAdmissionError,
    V2RuntimeDependencies,
    V2RuntimeFactory,
)
from robomex.orchestration.episode import EpisodeRuntime, EpisodeRuntimeError
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    FeasibilityStatus,
    MotionPlan,
    WorldKind,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ControlOutcome, NodeOutcomeEvent
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion, _snapshot

_ROBOT_DIGEST = "sha256:" + "c" * 64


class _Crash(BaseException):
    pass


class _Checker:
    def __init__(self, *, passed: bool = True) -> None:
        self.passed = passed

    def certify(self, spec, snapshot):
        status = FeasibilityStatus.PASS if self.passed else FeasibilityStatus.FAIL
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="arena-e2e-checker",
            checks={
                "exact_joint_path_interface": status,
                "controller_admissible": status,
                "resource_binding": status,
                "joint_order": status,
                "collision": status,
                "joint_limits": status,
                "robot_model": status,
            },
        )


def _risk(*, high: bool, count: int) -> RiskReport:
    return RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.45 if high else 0.99,
            target_margin_m=0.003 if high else 0.05,
            ik_status=CheckStatus.UNKNOWN if high else CheckStatus.PASS,
            collision_status=CheckStatus.UNKNOWN if high else CheckStatus.PASS,
            clearance_m=0.002 if high else 0.05,
            held_pose_uncertainty_m=0.03 if high else 0.001,
            prior_failures=1 if high else 0,
        ),
        RiskPolicy(max_candidates=count),
    )


def _arena_node(*, effect_scope: EffectScope, count: int) -> ActivationSpec:
    return ActivationSpec(
        activation_id="arena",
        runner_kind=RunnerKind.ARENA,
        runner_ref="arena.motion",
        effect_scope=effect_scope,
        inputs=(
            PortSpecV2(name="snapshot", schema_id="robomex.admission_snapshot.v1"),
            PortSpecV2(name="risk", schema_id="robomex.risk_report.v1"),
        ),
        outputs=(
            PortSpecV2(name="result", schema_id="robomex.arena_result.v1"),
            PortSpecV2(name="hypotheses", schema_id="robomex.arena_hypotheses.v1"),
            PortSpecV2(
                name="promotion_receipt",
                schema_id="robomex.arena_promotion_receipt.v1",
                required=False,
            ),
            PortSpecV2(
                name="selected_action_spec",
                schema_id="robomex.motion_plan.v2",
                required=False,
            ),
        ),
        bindings=(
            ExternalBinding(
                input_port="snapshot",
                ref="snapshot",
                schema_id="robomex.admission_snapshot.v1",
            ),
            ExternalBinding(
                input_port="risk",
                ref="risk",
                schema_id="robomex.risk_report.v1",
            ),
        ),
        estimated_budget=ExecutionBudget(
            actor_spawns=count,
            shadow_rollouts=(count if effect_scope is EffectScope.SHADOW_WORLD else 0),
        ),
    )


def _graph(*, effect_scope: EffectScope, count: int, execute: bool):
    arena = _arena_node(effect_scope=effect_scope, count=count)
    if not execute:
        return ElasticGraphCompiler().compile(
            ElasticGraphSpec(
                graph_id="arena_graph",
                entry_activation="arena",
                terminal_activations=("arena",),
                activations=(arena,),
            )
        )
    action = ActivationSpec(
        activation_id="execute",
        runner_kind=RunnerKind.SYSTEM_ACTION,
        runner_ref="runtime.sealed_action",
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        authority_world_id="live-world",
        authoritative_resource="arm",
        inputs=(PortSpecV2(name="action_spec", schema_id="robomex.motion_plan.v2"),),
        outputs=(PortSpecV2(name="receipt", schema_id="robomex.execution_receipt.v2"),),
        bindings=(
            ArtifactBinding(
                input_port="action_spec",
                source_activation="arena",
                source_port="selected_action_spec",
            ),
        ),
        estimated_budget=ExecutionBudget(authoritative_actions=1),
    )
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="arena_graph",
            entry_activation="arena",
            terminal_activations=("execute",),
            activations=(arena, action),
            transitions=(
                TransitionSpec(source="arena", outcome=ControlOutcome.SUCCESS, target="execute"),
            ),
        )
    )


def _build(
    tmp_path,
    *,
    count: int,
    high: bool,
    shadow: bool,
    passed: bool = True,
    episode_root=None,
    rejecting_provider: bool = False,
):
    snapshot = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    plans = tuple(
        _motion(snapshot, plan_id=f"candidate-{index}", endpoint=(0.2 + index, 0.3))
        for index in range(count)
    )
    plan_refs = {}

    def handler(profile, invocation, isolation):
        if rejecting_provider:
            raise AssertionError("completed Arena recovery re-entered candidate provider")
        index = int(profile.profile_id.rsplit("-", 1)[1])
        return {
            "candidate_id": f"candidate-{index}",
            "strategy": f"strategy-{index}",
            "plan_ref": plan_refs[index],
            "snapshot_ref": invocation.inputs["snapshot"],
            "frame": "world",
            "expected_effect": "bounded alignment",
            "utility": float(index),
            "estimated_risk": 0.1,
            "clearance_m": 0.03,
            "path_length": 0.2,
            "terminal_position_m": (0.1 + index, 0.2, 0.3),
        }

    provider = InMemoryAgentProvider(handler)
    actors = ActorRegistry(
        {"memory": provider},
        namespace_root="ep",
        workspace_root=tmp_path / "actors",
    )
    live_backend = FakeBackend(snapshot)
    shadow_registry = ShadowBackendRegistry()
    shadow_backend = None
    if shadow:
        shadow_snapshots = tuple(
            snapshot.model_copy(
                update={
                    "world_id": f"shadow-{index}",
                    "world_kind": WorldKind.SHADOW,
                }
            )
            for index in range(count)
        )
        shadow_backend = FakeBackend(shadow_snapshots)
        shadow_registry = ShadowBackendRegistry(
            (
                RegisteredShadowBackend(
                    backend_id="fake-backend",
                    backend=shadow_backend,
                    world_resource_bindings=frozenset(
                        (item.world_id, item.resource_id) for item in shadow_snapshots
                    ),
                ),
            )
        )
    candidates = tuple(
        ArenaCandidateSpec(
            candidate_id=f"candidate-{index}",
            strategy=f"strategy-{index}",
            profile=ActorProfile(
                profile_id=f"candidate-profile-{index}",
                provider_id="memory",
                lifecycle=ActorLifecycle.EPHEMERAL,
                effect_ceiling=(frozenset({"shadow_world.write"}) if shadow else frozenset()),
                isolation=IsolationPolicy(world_id=(f"shadow-{index}" if shadow else None)),
            ),
            objective="propose one sealed motion",
            effect_scope=(EffectScope.SHADOW_WORLD if shadow else EffectScope.READ_ONLY),
            requested_effects=(frozenset({"shadow_world.write"}) if shadow else frozenset()),
            shadow_backend_id=("fake-backend" if shadow else None),
            shadow_resource_id=("arm" if shadow else None),
        )
        for index in range(count)
    )
    risk_policy = RiskPolicy(max_candidates=count)
    binding = ArenaBinding(
        binding_id="arena.motion",
        candidates=candidates,
        expected_frame="world",
        robot_model_digest=_ROBOT_DIGEST,
        candidate_budget_limit=count,
        policy=ArenaPolicy(max_candidates=count),
        risk_policy=risk_policy,
    )
    runtime = EpisodeRuntime(
        episode_id="ep",
        episode_root=episode_root or tmp_path / "episode",
        actors=actors,
        action_backends={(snapshot.world_id, snapshot.resource_id): live_backend},
        feasibility_checkers={(snapshot.world_id, snapshot.resource_id): _Checker(passed=passed)},
        shadow_backends=shadow_registry,
        arena_bindings=(binding,),
    )
    if not rejecting_provider:
        runtime.data_plane.open_workflow("inputs")
        snapshot_record = runtime.data_plane.publish(
            workflow_id="inputs",
            activation_id="capture",
            attempt=1,
            port="snapshot",
            schema=snapshot.schema_version,
            payload=snapshot.model_dump(mode="json"),
        )
        risk = _risk(high=high, count=count)
        risk_record = runtime.data_plane.publish(
            workflow_id="inputs",
            activation_id="risk",
            attempt=1,
            port="risk",
            schema="robomex.risk_report.v1",
            payload=risk.model_dump(mode="json"),
        )
        for index, plan in enumerate(plans):
            plan_refs[index] = runtime.data_plane.publish(
                workflow_id="inputs",
                activation_id=f"plan-{index}",
                attempt=1,
                port="plan",
                schema=plan.schema_version,
                payload=plan.model_dump(mode="json"),
                lineage=(snapshot_record.ref,),
            ).ref
        refs = {"snapshot": snapshot_record.ref, "risk": risk_record.ref}
    else:
        refs = None
    return runtime, provider, live_backend, shadow_backend, binding, refs


@pytest.mark.parametrize(("high", "expected"), [(False, 1), (True, 3)])
def test_graph_arena_risk_expansion_and_only_selected_is_authoritative(
    tmp_path, high: bool, expected: int
) -> None:
    runtime, provider, live, shadow_backend, _binding, refs = _build(
        tmp_path, count=3, high=high, shadow=True
    )
    graph = _graph(effect_scope=EffectScope.SHADOW_WORLD, count=3, execute=True)
    workflow = runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="arena-motion",
            instruction="select and execute one safe alignment",
            success_rubric="one selected action converges",
        ),
        graph=graph,
        external_refs=refs,
    )

    terminal = runtime.run_until_terminal(workflow)

    assert terminal.status.value == "succeeded"
    assert len([call for call in provider.calls if call[0] == "invoke"]) == expected
    assert shadow_backend is not None
    assert len(shadow_backend.calls) == expected
    assert len(live.calls) == 1
    selected = runtime.data_plane.resolve(
        next(
            record.ref
            for record in runtime.data_plane.artifacts
            if record.workflow_id == "wf"
            and record.activation_id == "arena"
            and record.port == "selected_action_spec"
        )
    )
    plan = MotionPlan.model_validate(selected.payload)
    expected_id = "candidate-0" if not high else "candidate-2"
    assert plan.plan_id == expected_id
    assert all(
        receipt["receipt_authority"] == "shadow_only"
        for result in runtime.data_plane.artifacts
        if result.workflow_id == "wf" and result.port == "result"
        for receipt in (
            item.get("shadow_rollout_receipt")
            for item in runtime.data_plane.resolve(result.ref).payload["candidate_results"]
        )
        if receipt is not None
    )


def test_graph_arena_all_rejected_never_reaches_authoritative_action(tmp_path) -> None:
    runtime, provider, live, _shadow, _binding, refs = _build(
        tmp_path, count=2, high=True, shadow=False, passed=False
    )
    graph = _graph(effect_scope=EffectScope.READ_ONLY, count=2, execute=True)
    workflow = runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="reject",
            instruction="reject unsafe actions",
            success_rubric="no physical call occurs",
        ),
        graph=graph,
        external_refs=refs,
    )

    terminal = runtime.run_until_terminal(workflow)

    assert terminal.status.value == "failed"
    assert terminal.terminal_outcome is ControlOutcome.INFEASIBLE
    assert len([call for call in provider.calls if call[0] == "invoke"]) == 2
    assert live.calls == []
    assert not any(
        record.port == "selected_action_spec"
        for record in runtime.data_plane.artifacts
        if record.workflow_id == "wf"
    )


def test_graph_arena_zero_durable_reservation_is_exhausted_not_infeasible(
    tmp_path,
) -> None:
    runtime, provider, live, _shadow, binding, refs = _build(
        tmp_path, count=1, high=False, shadow=False
    )
    assert refs is not None
    assert binding.candidate_budget_id is not None
    snapshot_ref = refs["snapshot"]
    snapshot = AdmissionSnapshot.model_validate(
        runtime.data_plane.resolve(snapshot_ref).payload
    )
    graph = _graph(effect_scope=EffectScope.READ_ONLY, count=1, execute=False)
    runtime.arena_consumption_ledger.reserve(
        context=ArenaContext(
            arena_run_id="prior-crashed-arena-run",
            episode_id="ep",
            workflow_id="wf",
            graph_id=graph.spec.graph_id,
            graph_revision=graph.spec.revision,
            graph_digest=f"sha256:{graph.digest}",
            slot_id="arena",
            snapshot_ref=snapshot_ref,
            expected_frame=binding.expected_frame,
            world_id=snapshot.world_id,
            resource_id=snapshot.resource_id,
            robot_model_digest=binding.robot_model_digest,
            config_digest=snapshot.config_digest,
            candidate_budget_id=binding.candidate_budget_id,
            candidate_budget_limit=binding.candidate_budget_limit,
        ),
        requested_candidates=1,
        caller_remaining=1,
    )
    workflow = runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="quota-exhaustion",
            instruction="stop when the durable Arena quota is consumed",
            success_rubric="no proposal worker is entered",
        ),
        graph=graph,
        external_refs=refs,
    )

    terminal = runtime.run_until_terminal(workflow)

    assert terminal.status.value == "exhausted"
    assert terminal.terminal_outcome is ControlOutcome.EXHAUSTED
    assert provider.calls == ()
    assert live.calls == []
    result_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.workflow_id == "wf"
        and record.activation_id == "arena"
        and record.port == "result"
    )
    payload = runtime.data_plane.resolve(result_record.ref).payload
    assert payload["quota_exhausted"] is True
    assert payload["selection_reason"] == (
        "durable candidate quota exhausted: reservation admitted 0 of 1 "
        "requested candidates"
    )
    outcome = next(
        event
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent) and event.source == "graph_arena"
    )
    assert outcome.outcome is ControlOutcome.EXHAUSTED


def test_graph_arena_budget_must_cover_trusted_candidate_model_aggregate(
    tmp_path,
) -> None:
    candidate = ArenaCandidateSpec(
        candidate_id="bounded",
        strategy="bounded-candidate",
        profile=ActorProfile(
            profile_id="bounded-candidate-profile",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
        ),
        objective="propose one bounded motion",
        estimated_budget=ExecutionBudget(
            model_calls=2,
            tokens=40,
            wall_time_ms=500,
        ),
    )
    binding = ArenaBinding(
        binding_id="arena.motion",
        candidates=(candidate,),
        expected_frame="world",
        robot_model_digest=_ROBOT_DIGEST,
        candidate_budget_limit=1,
        policy=ArenaPolicy(max_candidates=1),
        risk_policy=RiskPolicy(max_candidates=1),
    )
    runtime = EpisodeRuntime(
        episode_id="ep",
        episode_root=tmp_path / "episode",
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        arena_bindings=(binding,),
    )
    underdeclared = _graph(
        effect_scope=EffectScope.READ_ONLY,
        count=1,
        execute=False,
    )

    with pytest.raises(EpisodeRuntimeError, match="model_calls"):
        runtime.open_workflow(
            workflow_id="wf",
            intent=SubgoalIntent(
                intent_id="underdeclared-arena",
                instruction="evaluate one proposal",
                success_rubric="candidate budget is closed",
            ),
            graph=underdeclared,
        )


def test_graph_arena_completed_result_replays_without_provider_reentry(tmp_path) -> None:
    root = tmp_path / "episode"
    runtime, _provider, _live, _shadow, binding, refs = _build(
        tmp_path, count=1, high=False, shadow=False, episode_root=root
    )
    graph = _graph(effect_scope=EffectScope.READ_ONLY, count=1, execute=False)
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="restart",
            instruction="select once",
            success_rubric="selection is durable",
        ),
        graph=graph,
        external_refs=refs,
    )
    scheduler = runtime._workflow("wf").scheduler
    on_event = scheduler.on_event

    def crash_before_node_commit(event):
        if isinstance(event, NodeOutcomeEvent) and event.source == "graph_arena":
            raise _Crash()
        return on_event(event)

    scheduler.on_event = crash_before_node_commit  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        runtime.execute_ready("wf")

    restarted, provider, _live2, _shadow2, _binding2, _refs2 = _build(
        tmp_path,
        count=1,
        high=False,
        shadow=False,
        episode_root=root,
        rejecting_provider=True,
    )
    assert binding.content_digest == _binding2.content_digest
    restarted.recover_workflow("wf")

    terminal = restarted.run_until_terminal("wf")

    assert terminal.status.value == "succeeded"
    assert not [call for call in provider.calls if call[0] == "invoke"]
    assert [record.record_kind for record in restarted.arena_consumption_ledger.records] == [
        "reservation",
        "completion",
    ]


def test_graph_arena_stale_context_and_params_authority_fail_closed(tmp_path) -> None:
    runtime, provider, live, _shadow, _binding, refs = _build(
        tmp_path, count=1, high=False, shadow=False
    )
    graph = _graph(effect_scope=EffectScope.READ_ONLY, count=1, execute=False)
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="stale",
            instruction="do not promote stale work",
            success_rubric="stale context fails",
        ),
        graph=graph,
        external_refs=refs,
    )
    runtime._arena_graph_revision = lambda _context: ("arena_graph", 2)  # type: ignore[method-assign]

    terminal = runtime.run_until_terminal("wf")

    assert terminal.status.value == "failed"
    assert terminal.terminal_outcome is ControlOutcome.STALE_INPUT
    assert not [call for call in provider.calls if call[0] == "invoke"]
    assert live.calls == []

    illegal = _arena_node(effect_scope=EffectScope.READ_ONLY, count=1).model_copy(
        update={"params": {"candidates": ["untrusted"]}}
    )
    illegal_graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="illegal_arena",
            entry_activation="arena",
            terminal_activations=("arena",),
            activations=(illegal,),
        )
    )
    with pytest.raises(EpisodeRuntimeError, match="do not accept params"):
        runtime.open_workflow(
            workflow_id="illegal",
            intent=SubgoalIntent(
                intent_id="illegal",
                instruction="must fail",
                success_rubric="untrusted params rejected",
            ),
            graph=illegal_graph,
            external_refs=refs,
        )


def test_runtime_factory_requires_exact_manifest_pin_for_arena_binding(tmp_path) -> None:
    from robomex.test.test_v2_bootstrap_manifest import _fixtures, _rebuild_manifest

    config, manifest, dependencies = _fixtures(tmp_path)
    profile = dependencies.actor_profiles["worker"]
    binding = ArenaBinding(
        binding_id="arena.bootstrap",
        candidates=(
            ArenaCandidateSpec(
                candidate_id="one",
                strategy="read-only-proposal",
                profile=profile,
                objective="propose one exact motion",
            ),
        ),
        expected_frame="world",
        robot_model_digest=_ROBOT_DIGEST,
        candidate_budget_limit=1,
        policy=ArenaPolicy(max_candidates=1),
        risk_policy=RiskPolicy(max_candidates=1),
    )
    admitted = V2RuntimeDependencies(
        contract_catalog=dependencies.contract_catalog,
        actor_providers=dependencies.actor_providers,
        actor_profiles=dependencies.actor_profiles,
        action_backends=dependencies.action_backends,
        observation_backends=dependencies.observation_backends,
        shadow_backends=dependencies.shadow_backends,
        arena_bindings=(binding,),
    )
    wrong = _rebuild_manifest(
        manifest,
        metadata={"arena_binding_digests": {binding.binding_id: "sha256:" + "0" * 64}},
    )
    with pytest.raises(DependencyAdmissionError, match="content hash drift"):
        V2RuntimeFactory().build(
            config=config.model_copy(update={"run_id": wrong.run_id}),
            manifest=wrong,
            dependencies=admitted,
        )

    pinned = _rebuild_manifest(
        manifest,
        metadata={
            "arena_binding_digests": {
                binding.binding_id: binding.content_digest,
            }
        },
    )
    app = V2RuntimeFactory().build(
        config=config,
        manifest=pinned,
        dependencies=admitted,
    )
    assert app.episode.arena_bindings.resolve(binding.binding_id) == binding
