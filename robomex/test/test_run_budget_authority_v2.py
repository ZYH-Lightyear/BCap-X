from __future__ import annotations

import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from robomex.data import ResolvedArtifactRef
from robomex.elastic import (
    ActivationSpec,
    ArtifactBinding,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)
from robomex.evolution import RunBudgets
from robomex.orchestration.actors import (
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.arena import (
    ArenaCandidateResult,
    ArenaCandidateSpec,
    ArenaContext,
    ArenaResult,
    CandidateStatus,
    CheckStatus,
    RiskInputs,
    RiskPolicy,
    RiskReport,
)
from robomex.orchestration.bootstrap import V2RuntimeFactory
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerDecision,
    ManagerLimits,
    ManagerSignal,
    ScriptedManagerInvoker,
    SwarmManagerSession,
)
from robomex.orchestration.run_budget import (
    RunBudgetAuthority,
    RunBudgetExceededError,
    RunBudgetIdentityConflictError,
    RunBudgetOperationStatus,
    RunBudgetVector,
)
from robomex.orchestration.task_orchestrator import (
    EpisodeOrchestrator,
    PlannerCallBudget,
    PlannerDecision,
    PlannerDecisionKind,
    ScriptedIntentPlanner,
    ScriptedWorkflowAuthor,
    TaskRunStatus,
)
from robomex.runtime.action_protocol import FeasibilityStatus
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import (
    ArenaDecision,
    ControlOutcome,
    ManagerInvocationOutcome,
    NodeOutcomeEvent,
)
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion, _snapshot
from robomex.test.test_v2_bootstrap_manifest import _fixtures


def _budgets(**overrides: int | float) -> RunBudgets:
    values: dict[str, int | float] = {
        "max_model_calls": 10,
        "max_tokens": 10_000,
        "max_wall_time_s": 300.0,
        "max_physical_actions": 10,
        "max_shadow_rollouts": 10,
        "max_candidates": 10,
        "max_recoveries": 10,
    }
    values.update(overrides)
    return RunBudgets.model_validate(values)


def _concurrent_reserve_worker(path: str, gate, results, operation_id: str) -> None:
    authority = RunBudgetAuthority(
        path,
        budgets=_budgets(max_model_calls=1),
    )
    gate.wait()
    try:
        reservation = authority.reserve(
            operation_id=operation_id,
            requested=RunBudgetVector(model_calls=1),
            binding={"worker": operation_id},
        )
        authority.complete(reservation)
        results.put("completed")
    except RunBudgetExceededError:
        results.put("denied")


def _one_node_graph(*, budget: ExecutionBudget | None = None):
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="budget-graph",
            entry_activation="worker",
            terminal_activations=("worker",),
            activations=(
                ActivationSpec(
                    activation_id="worker",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.worker",
                    estimated_budget=budget
                    or ExecutionBudget(
                        model_calls=1,
                        tokens=5,
                        wall_time_ms=1_000,
                    ),
                ),
            ),
        )
    )


def _open_one_node(runtime: EpisodeRuntime, *, workflow_id: str = "wf") -> str:
    runtime.register_actor_profile(
        "test.worker",
        ActorProfile(profile_id="worker", runner_kind="coding_worker"),
    )
    return runtime.open_workflow(
        workflow_id=workflow_id,
        intent=SubgoalIntent(
            intent_id=f"intent-{workflow_id}",
            instruction="perform one bounded proposal",
            success_rubric="the proposal is closed",
        ),
        graph=_one_node_graph(),
    )


def test_reserve_complete_release_are_durable_idempotent_and_bound(tmp_path: Path) -> None:
    clock = [100.0]
    path = tmp_path / "budget.jsonl"
    authority = RunBudgetAuthority(
        path, budgets=_budgets(max_model_calls=2), clock=lambda: clock[0]
    )
    request = RunBudgetVector(model_calls=1, tokens=20, wall_time_s=1.0)
    first = authority.reserve(
        operation_id="provider:one",
        requested=request,
        binding={"invocation": "sha256:a"},
    )

    restarted = RunBudgetAuthority(
        path, budgets=_budgets(max_model_calls=2), clock=lambda: clock[0]
    )
    retry = restarted.reserve(
        operation_id="provider:one",
        requested=request,
        binding={"invocation": "sha256:a"},
    )
    assert retry == first
    assert restarted.snapshot().committed.model_calls == 1

    completed = restarted.complete(first)
    assert completed.status is RunBudgetOperationStatus.COMPLETED
    assert restarted.complete(first) == completed
    assert restarted.snapshot().committed.model_calls == 1

    with pytest.raises(RunBudgetIdentityConflictError, match="rebound"):
        restarted.reserve(
            operation_id="provider:one",
            requested=RunBudgetVector(model_calls=1, tokens=21, wall_time_s=1.0),
            binding={"invocation": "sha256:a"},
        )
    with pytest.raises(RunBudgetIdentityConflictError, match="rebound"):
        restarted.reserve(
            operation_id="provider:one",
            requested=request,
            binding={"invocation": "sha256:b"},
        )

    released = restarted.reserve(
        operation_id="provider:not-entered",
        requested=RunBudgetVector(model_calls=1),
        binding="not-entered",
    )
    restarted.release(released)
    assert restarted.snapshot().committed.model_calls == 1


def test_budget_identity_and_wall_deadline_fail_closed(tmp_path: Path) -> None:
    clock = [10.0]
    path = tmp_path / "budget.jsonl"
    RunBudgetAuthority(path, budgets=_budgets(), clock=lambda: clock[0])
    with pytest.raises(RunBudgetIdentityConflictError, match="RunBudgets"):
        RunBudgetAuthority(
            path,
            budgets=_budgets(max_tokens=9_999),
            clock=lambda: clock[0],
        )

    deadline_path = tmp_path / "deadline.jsonl"
    authority = RunBudgetAuthority(
        deadline_path,
        budgets=_budgets(max_wall_time_s=2.0),
        clock=lambda: clock[0],
    )
    clock[0] = 12.0
    with pytest.raises(RunBudgetExceededError, match="wall_deadline"):
        authority.reserve(
            operation_id="late",
            requested=RunBudgetVector(model_calls=1),
            binding="late",
        )


def test_concurrent_processes_cannot_overreserve_one_global_call(tmp_path: Path) -> None:
    path = tmp_path / "concurrent-budget.jsonl"
    RunBudgetAuthority(path, budgets=_budgets(max_model_calls=1))
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_concurrent_reserve_worker,
            args=(str(path), gate, results, f"worker-{index}"),
        )
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    gate.set()
    outcomes = sorted(results.get(timeout=10) for _ in workers)
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    assert outcomes == ["completed", "denied"]
    restarted = RunBudgetAuthority(path, budgets=_budgets(max_model_calls=1))
    assert restarted.snapshot().committed.model_calls == 1


def test_factory_rejects_manifest_budget_identity_conflict(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    conflicting = manifest.budgets.model_copy(
        update={"max_model_calls": manifest.budgets.max_model_calls + 1}
    )
    RunBudgetAuthority(config.episode_root / "run_budget.v1.jsonl", budgets=conflicting)

    with pytest.raises(RunBudgetIdentityConflictError, match="RunBudgets"):
        V2RuntimeFactory().build(
            config=config,
            manifest=manifest,
            dependencies=dependencies,
        )


def test_planner_budget_exhaustion_never_enters_bounded_planner(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode"
    authority = RunBudgetAuthority(
        episode_root / "run_budget.v1.jsonl",
        budgets=_budgets(max_model_calls=0),
    )
    runtime = EpisodeRuntime(
        episode_id="planner-budget",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    planner = ScriptedIntentPlanner((PlannerDecision(kind=PlannerDecisionKind.DONE),))
    orchestrator = EpisodeOrchestrator(
        runtime,
        planner=planner,
        author=ScriptedWorkflowAuthor({}),
        planner_budget=PlannerCallBudget(token_grant=32, wall_time_ms=100),
    )
    orchestrator.start(task_run_id="planner_zero", task="do nothing")

    result = orchestrator.run("planner_zero")

    assert result.status is TaskRunStatus.EXHAUSTED
    assert planner.requests == []
    assert authority.snapshot().committed.model_calls == 0


def test_planner_completed_response_replays_after_precommit_crash(
    tmp_path: Path,
) -> None:
    episode_root = tmp_path / "episode"
    budgets = _budgets(max_model_calls=1, max_tokens=64)
    authority = RunBudgetAuthority(episode_root / "run_budget.v1.jsonl", budgets=budgets)
    runtime = EpisodeRuntime(
        episode_id="planner-replay",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors-first"),
        run_budget_authority=authority,
    )
    planner = ScriptedIntentPlanner(
        (PlannerDecision(kind=PlannerDecisionKind.DONE, reason="complete"),)
    )
    planner_budget = PlannerCallBudget(token_grant=64, wall_time_ms=100)
    first = EpisodeOrchestrator(
        runtime,
        planner=planner,
        author=ScriptedWorkflowAuthor({}),
        planner_budget=planner_budget,
    )
    first.start(task_run_id="planner_replay", task="already complete")
    original_commit = first._commit

    def crash_before_task_commit(state, *, reason, **updates):
        if reason == "planner_done":
            raise RuntimeError("synthetic crash after durable planner response")
        return original_commit(state, reason=reason, **updates)

    first._commit = crash_before_task_commit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synthetic crash"):
        first.run("planner_replay")
    assert len(planner.requests) == 1
    assert authority.snapshot().committed.model_calls == 1

    class NoCallPlanner:
        calls = 0

        def next_intent(self, _context):
            raise AssertionError("durable planner decision must replay")

        def next_intent_bounded(
            self,
            _context,
            *,
            max_tokens,
            max_model_calls,
            deadline_monotonic_s,
        ):
            del max_tokens, max_model_calls, deadline_monotonic_s
            self.calls += 1
            raise AssertionError("durable planner decision must replay")

    restarted_authority = RunBudgetAuthority(episode_root / "run_budget.v1.jsonl", budgets=budgets)
    restarted_runtime = EpisodeRuntime(
        episode_id="planner-replay",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors-restarted"),
        run_budget_authority=restarted_authority,
    )
    no_call = NoCallPlanner()
    restarted = EpisodeOrchestrator(
        restarted_runtime,
        planner=no_call,
        author=ScriptedWorkflowAuthor({}),
        planner_budget=planner_budget,
    )

    result = restarted.run("planner_replay")

    assert result.status is TaskRunStatus.SUCCEEDED
    assert no_call.calls == 0
    assert restarted_authority.snapshot().committed.model_calls == 1


def test_model_exhaustion_closes_node_without_provider_call(tmp_path: Path) -> None:
    provider = InMemoryAgentProvider(
        lambda *_: ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)
    )
    authority = RunBudgetAuthority(
        tmp_path / "episode" / "run_budget.v1.jsonl",
        budgets=_budgets(max_model_calls=0),
    )
    runtime = EpisodeRuntime(
        episode_id="model-budget",
        episode_root=tmp_path / "episode",
        actors=ActorRegistry({"in_memory": provider}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    workflow_id = _open_one_node(runtime)

    terminal = runtime.run_until_terminal(workflow_id)

    assert not provider.calls
    assert terminal.status.value == "exhausted"
    event = next(item for item in runtime.event_bus.history if isinstance(item, NodeOutcomeEvent))
    assert event.outcome is ControlOutcome.EXHAUSTED
    assert event.source == "run_budget_authority"


def test_actor_invocation_receives_typed_run_clamped_deadline(tmp_path: Path) -> None:
    observed_remaining_s: list[float | None] = []

    def invoke(_profile, spec, _isolation):
        observed_remaining_s.append(
            None
            if spec.deadline_monotonic_s is None
            else spec.deadline_monotonic_s - time.monotonic()
        )
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    provider = InMemoryAgentProvider(invoke)
    episode_root = tmp_path / "episode"
    runtime = EpisodeRuntime(
        episode_id="actor-deadline",
        episode_root=episode_root,
        actors=ActorRegistry({"in_memory": provider}, workspace_root=tmp_path / "actors"),
        run_budget_authority=RunBudgetAuthority(
            episode_root / "run_budget.v1.jsonl",
            budgets=_budgets(max_wall_time_s=5.0),
        ),
    )
    workflow_id = _open_one_node(runtime)
    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "succeeded"
    assert len(observed_remaining_s) == 1
    remaining_s = observed_remaining_s[0]
    assert remaining_s is not None
    assert 0 < remaining_s <= 1.0


def _two_action_graph(plan):
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="two-actions",
            entry_activation="author",
            terminal_activations=("execute_two",),
            activations=(
                ActivationSpec(
                    activation_id="author",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.author",
                    outputs=(PortSpecV2(name="action_spec", schema_id=plan.schema_version),),
                    estimated_budget=ExecutionBudget(
                        model_calls=1,
                        tokens=10,
                        wall_time_ms=1_000,
                    ),
                ),
                *(
                    ActivationSpec(
                        activation_id=activation_id,
                        runner_kind=RunnerKind.SYSTEM_ACTION,
                        runner_ref="runtime.sealed_action",
                        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
                        authority_world_id=plan.world_id,
                        authoritative_resource=plan.resource_id,
                        inputs=(PortSpecV2(name="action_spec", schema_id=plan.schema_version),),
                        outputs=(
                            PortSpecV2(
                                name="receipt",
                                schema_id="robomex.execution_receipt.v2",
                            ),
                        ),
                        bindings=(
                            ArtifactBinding(
                                input_port="action_spec",
                                source_activation="author",
                                source_port="action_spec",
                            ),
                        ),
                        estimated_budget=ExecutionBudget(authoritative_actions=1),
                    )
                    for activation_id in ("execute_one", "execute_two")
                ),
            ),
            transitions=(
                TransitionSpec(
                    source="author",
                    outcome=ControlOutcome.SUCCESS,
                    target="execute_one",
                ),
                TransitionSpec(
                    source="execute_one",
                    outcome=ControlOutcome.SUCCESS,
                    target="execute_two",
                ),
            ),
        )
    )


def test_second_physical_action_is_denied_before_backend_call(tmp_path: Path) -> None:
    plan = _motion(_snapshot(captured_at=datetime.now(timezone.utc)))  # noqa: UP017
    backend = FakeBackend(plan.expected_snapshot)

    class Checker:
        def certify(self, spec, snapshot):
            return build_feasibility_certificate(
                spec=spec,
                snapshot=snapshot,
                checker_id="budget-test-checker",
                checks=dict.fromkeys(
                    (
                        "exact_joint_path_interface",
                        "controller_admissible",
                        "resource_binding",
                        "joint_order",
                        "collision",
                        "joint_limits",
                        "robot_model",
                    ),
                    FeasibilityStatus.PASS,
                ),
            )

    provider = InMemoryAgentProvider(
        lambda *_: ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="action_spec",
                    schema_id=plan.schema_version,
                    payload=plan.model_dump(mode="json"),
                ),
            ),
        )
    )
    episode_root = tmp_path / "episode"
    runtime = EpisodeRuntime(
        episode_id="physical-budget",
        episode_root=episode_root,
        actors=ActorRegistry({"in_memory": provider}, workspace_root=tmp_path / "actors"),
        action_backends={(plan.world_id, plan.resource_id): backend},
        feasibility_checkers={(plan.world_id, plan.resource_id): Checker()},
        run_budget_authority=RunBudgetAuthority(
            episode_root / "run_budget.v1.jsonl",
            budgets=_budgets(max_physical_actions=1),
        ),
    )
    runtime.register_actor_profile(
        "test.author",
        ActorProfile(profile_id="author", runner_kind="coding_worker"),
    )
    workflow_id = runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="two-actions",
            instruction="attempt two actions",
            success_rubric="the budget stops the second action",
        ),
        graph=_two_action_graph(plan),
    )

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "exhausted"
    assert len(backend.calls) == 1
    denied = next(
        item
        for item in runtime.event_bus.history
        if isinstance(item, NodeOutcomeEvent) and item.activation_id == "execute_two"
    )
    assert denied.outcome is ControlOutcome.EXHAUSTED


def test_recovery_exhaustion_does_not_call_manager_invoker(tmp_path: Path) -> None:
    provider = InMemoryAgentProvider(
        lambda *_: ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)
    )
    episode_root = tmp_path / "episode"
    runtime = EpisodeRuntime(
        episode_id="manager-budget",
        episode_root=episode_root,
        actors=ActorRegistry({"in_memory": provider}, workspace_root=tmp_path / "actors"),
        run_budget_authority=RunBudgetAuthority(
            episode_root / "run_budget.v1.jsonl",
            budgets=_budgets(max_recoveries=0),
        ),
    )
    session = SwarmManagerSession(
        session_id="manager-session",
        episode_id="manager-budget",
        workflow_id="wf",
        intent_id="intent-wf",
        limits=ManagerLimits(max_reactivations=1, max_tokens=100, max_tokens_per_call=100),
    )
    invoker = ScriptedManagerInvoker((ManagerDecision(action=ManagerAction.NOOP, tokens_used=1),))
    runtime.register_actor_profile(
        "test.worker",
        ActorProfile(profile_id="worker", runner_kind="coding_worker"),
    )
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="intent-wf",
            instruction="open a managed workflow",
            success_rubric="recovery is bounded",
        ),
        graph=_one_node_graph(),
        manager_session=session,
        manager_invoker=invoker,
    )

    result = runtime.invoke_manager("wf", signal=ManagerSignal.RECOVERY)

    assert result.step.invoked is False
    assert not invoker.requests
    event = next(
        item for item in runtime.event_bus.history if isinstance(item, ManagerInvocationOutcome)
    )
    assert event.invoked is False
    assert event.source == "run_budget_authority"


def test_manager_invoker_owned_model_budget_is_not_double_charged_by_episode(
    tmp_path: Path,
) -> None:
    episode_root = tmp_path / "episode"
    authority = RunBudgetAuthority(
        episode_root / "run_budget.v1.jsonl",
        budgets=_budgets(max_model_calls=1, max_tokens=100),
    )
    runtime = EpisodeRuntime(
        episode_id="manager-owned-budget",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    session = SwarmManagerSession(
        session_id="manager-owned-session",
        episode_id="manager-owned-budget",
        workflow_id="wf",
        intent_id="intent-wf",
        limits=ManagerLimits(
            max_initial_calls=1,
            max_reactivations=0,
            max_tokens=100,
            max_tokens_per_call=100,
        ),
    )

    class OwningInvoker:
        manages_run_model_budget = True
        calls = 0

        def invoke(self, request):
            self.calls += 1
            reservation = authority.reserve(
                operation_id=f"owned-manager:{request.invocation_id}",
                requested=RunBudgetVector(
                    model_calls=1,
                    tokens=request.token_limit,
                ),
                binding=request.model_dump(mode="json"),
            )
            authority.complete(
                reservation,
                settlement=RunBudgetVector(model_calls=1, tokens=1),
            )
            return ManagerDecision(action=ManagerAction.NOOP, tokens_used=1)

    invoker = OwningInvoker()
    runtime.register_actor_profile(
        "test.worker",
        ActorProfile(profile_id="worker", runner_kind="coding_worker"),
    )
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="intent-wf",
            instruction="open a managed workflow",
            success_rubric="one Manager call is charged once",
        ),
        graph=_one_node_graph(),
        manager_session=session,
        manager_invoker=invoker,  # type: ignore[arg-type]
    )

    result = runtime.invoke_manager("wf", signal=ManagerSignal.INTENT_AUTHORING)

    assert result.step.invoked is True
    assert invoker.calls == 1
    snapshot = authority.snapshot()
    assert snapshot.committed.model_calls == 1
    charged_operations = [
        operation.operation_id
        for operation in snapshot.operations
        if operation.settlement and operation.settlement.model_calls
    ]
    assert len(charged_operations) == 1
    assert charged_operations[0].startswith("owned-manager:manager-owned-session-")


def test_candidate_exhaustion_records_decision_without_entering_arena(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode"
    authority = RunBudgetAuthority(
        episode_root / "run_budget.v1.jsonl",
        budgets=_budgets(max_candidates=1),
    )
    used = authority.reserve(
        operation_id="prior-arena",
        requested=RunBudgetVector(candidates=1),
        binding="prior-arena",
    )
    authority.complete(used)
    runtime = EpisodeRuntime(
        episode_id="arena-budget",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    runtime.register_actor_profile(
        "unused.worker",
        ActorProfile(profile_id="worker", runner_kind="coding_worker"),
    )
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="arena-intent",
            instruction="evaluate proposals",
            success_rubric="candidate use is bounded",
        ),
        graph=ElasticGraphCompiler().compile(
            ElasticGraphSpec(
                graph_id="arena-graph",
                entry_activation="arena",
                terminal_activations=("arena",),
                activations=(
                    ActivationSpec(
                        activation_id="arena",
                        runner_kind=RunnerKind.CODING_WORKER,
                        runner_ref="unused.worker",
                        estimated_budget=ExecutionBudget(
                            model_calls=1,
                            tokens=1,
                            wall_time_ms=1,
                        ),
                    ),
                ),
            )
        ),
    )

    class NeverRunArena:
        calls = 0

        def assert_bound_to(self, **_kwargs):
            return None

        def budget_request(self, **_kwargs):
            return 1, 1, 0

        def run(self, **_kwargs):
            self.calls += 1
            raise AssertionError("Arena external boundary must not be entered")

    arena = NeverRunArena()
    context = ArenaContext(
        arena_run_id="arena-two",
        episode_id="arena-budget",
        workflow_id="wf",
        graph_id="arena-graph",
        graph_revision=1,
        slot_id="slot",
        snapshot_ref=ResolvedArtifactRef("snapshot", "sha256:" + "a" * 64),
        expected_frame="world",
        world_id="shadow",
        resource_id="arm",
        robot_model_digest="sha256:" + "b" * 64,
        config_digest="sha256:" + "c" * 64,
        candidate_budget_id="candidate-budget",
        candidate_budget_limit=2,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=1.0,
            target_margin_m=0.1,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.1,
            held_pose_uncertainty_m=0.0,
        ),
        RiskPolicy(max_candidates=1),
    )

    with pytest.raises(RunBudgetExceededError, match="candidates"):
        runtime.run_arena(
            "wf",
            arena=arena,  # type: ignore[arg-type]
            context=context,
            risk_report=risk,
            candidates=(),
            candidate_budget_remaining=1,
        )

    assert arena.calls == 0
    decision = next(item for item in runtime.event_bus.history if isinstance(item, ArenaDecision))
    assert decision.source == "run_budget_authority"
    assert not decision.considered_candidate_ids


def test_arena_model_budget_denial_prevents_candidate_boundary(tmp_path: Path) -> None:
    episode_root = tmp_path / "episode"
    authority = RunBudgetAuthority(
        episode_root / "run_budget.v1.jsonl",
        budgets=_budgets(max_model_calls=0),
    )
    runtime = EpisodeRuntime(
        episode_id="arena-model-budget",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    _open_one_node(runtime)

    class NeverRunArena:
        calls = 0

        def assert_bound_to(self, **_kwargs):
            return None

        def budget_request(self, **_kwargs):
            return 1, 1, 0

        def run(self, **_kwargs):
            self.calls += 1
            raise AssertionError("Arena candidate boundary must not be entered")

    arena = NeverRunArena()
    context = ArenaContext(
        arena_run_id="arena-model-denied",
        episode_id="arena-model-budget",
        workflow_id="wf",
        graph_id="budget-graph",
        graph_revision=1,
        slot_id="slot",
        snapshot_ref=ResolvedArtifactRef("snapshot", "sha256:" + "a" * 64),
        expected_frame="world",
        world_id="shadow",
        resource_id="arm",
        robot_model_digest="sha256:" + "b" * 64,
        config_digest="sha256:" + "c" * 64,
        candidate_budget_id="candidate-budget",
        candidate_budget_limit=1,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=1.0,
            target_margin_m=0.1,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.1,
            held_pose_uncertainty_m=0.0,
        ),
        RiskPolicy(max_candidates=1),
    )
    candidate = ArenaCandidateSpec(
        candidate_id="model-candidate",
        strategy="bounded-model-proposal",
        profile=ActorProfile(
            profile_id="arena-candidate",
            runner_kind="coding_worker",
        ),
        objective="propose one candidate",
        estimated_budget=ExecutionBudget(
            model_calls=1,
            tokens=20,
            wall_time_ms=100,
        ),
    )

    with pytest.raises(RunBudgetExceededError, match="model_calls"):
        runtime.run_arena(
            "wf",
            arena=arena,  # type: ignore[arg-type]
            context=context,
            risk_report=risk,
            candidates=(candidate,),
            candidate_budget_remaining=1,
        )

    assert arena.calls == 0


def test_arena_settlement_charges_candidate_model_token_and_wall_grant(
    tmp_path: Path,
) -> None:
    episode_root = tmp_path / "episode"
    authority = RunBudgetAuthority(episode_root / "run_budget.v1.jsonl", budgets=_budgets())
    runtime = EpisodeRuntime(
        episode_id="arena-settlement",
        episode_root=episode_root,
        actors=ActorRegistry({}, workspace_root=tmp_path / "actors"),
        run_budget_authority=authority,
    )
    _open_one_node(runtime)
    context = ArenaContext(
        arena_run_id="arena-settled",
        episode_id="arena-settlement",
        workflow_id="wf",
        graph_id="budget-graph",
        graph_revision=1,
        slot_id="slot",
        snapshot_ref=ResolvedArtifactRef("snapshot", "sha256:" + "a" * 64),
        expected_frame="world",
        world_id="shadow",
        resource_id="arm",
        robot_model_digest="sha256:" + "b" * 64,
        config_digest="sha256:" + "c" * 64,
        candidate_budget_id="candidate-budget",
        candidate_budget_limit=1,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=1.0,
            target_margin_m=0.1,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.1,
            held_pose_uncertainty_m=0.0,
        ),
        RiskPolicy(max_candidates=1),
    )
    candidate = ArenaCandidateSpec(
        candidate_id="charged-candidate",
        strategy="bounded-model-proposal",
        profile=ActorProfile(
            profile_id="charged-arena-candidate",
            runner_kind="coding_worker",
        ),
        objective="propose one candidate",
        estimated_budget=ExecutionBudget(
            model_calls=2,
            tokens=30,
            wall_time_ms=250,
        ),
    )

    class CompletedArena:
        def assert_bound_to(self, **_kwargs):
            return None

        def budget_request(self, **_kwargs):
            return 1, 1, 0

        def run(self, **_kwargs):
            return ArenaResult(
                arena_run_id=context.arena_run_id,
                graph_id=context.graph_id,
                graph_revision=context.graph_revision,
                risk_report=risk,
                consumption_reservation_sequence=1,
                requested_candidates=1,
                candidate_budget_used=1,
                candidate_results=(
                    ArenaCandidateResult(
                        candidate_id=candidate.candidate_id,
                        actor_id="arena-actor",
                        strategy=candidate.strategy,
                        status=CandidateStatus.FAILED,
                        provider_entered=True,
                        error="synthetic rejected proposal",
                    ),
                ),
                selection_reason="all candidates failed",
            )

    runtime.run_arena(
        "wf",
        arena=CompletedArena(),  # type: ignore[arg-type]
        context=context,
        risk_report=risk,
        candidates=(candidate,),
        candidate_budget_remaining=1,
        publish_runtime_artifacts=False,
    )

    operation = next(
        item
        for item in authority.snapshot().operations
        if item.operation_id.endswith(":arena-settled")
    )
    assert operation.settlement == RunBudgetVector(
        model_calls=2,
        tokens=30,
        wall_time_s=0.25,
        candidates=1,
    )
