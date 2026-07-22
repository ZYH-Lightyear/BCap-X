from __future__ import annotations

from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    RunnerKind,
)
from robomex.evolution import RunBudgets
from robomex.orchestration.actors import ActorProfile, ActorRegistry, InMemoryAgentProvider
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    EpisodeRuntime,
    InvocationUsage,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.run_budget import RunBudgetAuthority
from robomex.runtime.events import ControlOutcome


def _run(tmp_path, *, usage_kind: str):
    def handler(_profile, invocation, _isolation):
        usage = None
        if usage_kind == "exact":
            usage = InvocationUsage(
                invocation_fingerprint=f"sha256:{invocation.fingerprint()}",
                model_calls=1,
                tokens=7,
                wall_time_s=0.01,
            )
        elif usage_kind == "wrong_binding":
            usage = InvocationUsage(
                invocation_fingerprint="sha256:" + "f" * 64,
                model_calls=1,
                tokens=7,
                wall_time_s=0.01,
            )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            usage=usage,
        )

    profile = ActorProfile(
        profile_id="usage-worker",
        provider_id="memory",
        runner_kind=RunnerKind.CODING_WORKER.value,
    )
    authority = RunBudgetAuthority(
        tmp_path / "budget.jsonl",
        budgets=RunBudgets(
            max_model_calls=3,
            max_tokens=100,
            max_wall_time_s=30,
            max_physical_actions=0,
        ),
    )
    runtime = EpisodeRuntime(
        episode_id="usage",
        episode_root=tmp_path / "episode",
        actors=ActorRegistry(
            {"memory": InMemoryAgentProvider(handler)},
            namespace_root="usage",
            workspace_root=tmp_path / "actors",
        ),
        run_budget_authority=authority,
    )
    runtime.register_actor_profile("usage.worker", profile)
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="usage",
            entry_activation="worker",
            terminal_activations=("worker",),
            activations=(
                ActivationSpec(
                    activation_id="worker",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="usage.worker",
                    estimated_budget=ExecutionBudget(
                        model_calls=3,
                        tokens=100,
                        wall_time_ms=5_000,
                    ),
                ),
            ),
        )
    )
    workflow = runtime.open_workflow(
        workflow_id="workflow",
        intent=SubgoalIntent(
            intent_id="usage",
            instruction="measure one provider",
            success_rubric="provider completes",
        ),
        graph=graph,
    )
    terminal = runtime.run_until_terminal(workflow)
    return terminal, authority.snapshot()


def test_exact_provider_usage_settles_below_reserved_grant(tmp_path) -> None:
    terminal, budget = _run(tmp_path, usage_kind="exact")
    assert terminal.status.value == "succeeded"
    assert budget.committed.model_calls == 1
    assert budget.committed.tokens == 7
    assert budget.committed.wall_time_s == 0.01


def test_absent_usage_conservatively_charges_full_grant(tmp_path) -> None:
    terminal, budget = _run(tmp_path, usage_kind="absent")
    assert terminal.status.value == "succeeded"
    assert budget.committed.model_calls == 3
    assert budget.committed.tokens == 100
    assert budget.committed.wall_time_s == 5.0


def test_rebound_usage_fails_activation_and_still_charges_full_grant(tmp_path) -> None:
    terminal, budget = _run(tmp_path, usage_kind="wrong_binding")
    assert terminal.status.value == "failed"
    assert budget.committed.model_calls == 3
    assert budget.committed.tokens == 100
    assert budget.committed.wall_time_s == 5.0
