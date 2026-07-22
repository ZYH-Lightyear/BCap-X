from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robomex.data import ResolvedArtifactRef
from robomex.elastic import ElasticGraphCompiler, ElasticGraphSpec
from robomex.evolution import RunBudgets
from robomex.orchestration.actors import InMemoryAgentProvider
from robomex.orchestration.application import (
    ManagerResponseError,
    ManagerWorkflowAuthor,
    ManifestPinnedWorkflowAuthor,
    PolicyManagerInvoker,
    RoboMExV2Agent,
    V2AgentConfig,
    V2ApplicationError,
)
from robomex.orchestration.bootstrap import (
    V2RuntimeDependencies,
    V2RuntimeFactory,
)
from robomex.orchestration.episode import ActivationExecutionResult
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.manager import (
    ManagerCallKind,
    ManagerInvocation,
    ManagerLimits,
    ManagerRemaining,
    ManagerSignal,
    ManagerSnapshot,
)
from robomex.orchestration.run_budget import (
    RunBudgetAuthority,
    RunBudgetExceededError,
)
from robomex.orchestration.task_orchestrator import (
    AvailableArtifact,
    PlannerCallBudget,
    PlannerDecision,
    PlannerDecisionKind,
    ScriptedIntentPlanner,
    ScriptedWorkflowAuthor,
    TaskRunStatus,
    WorkflowAuthoringContext,
    WorkflowBlueprint,
)
from robomex.runtime.events import ControlOutcome
from robomex.test.test_v2_bootstrap_manifest import _fixtures, _rebuild_manifest


class _Policy:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[list[dict]] = []
        self.token_ceilings: list[int] = []

    def complete(self, prompt: list[dict]) -> str:
        self.calls.append(prompt)
        return json.dumps(self.response)

    def complete_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float,
    ) -> str:
        assert max_tokens > 0
        assert deadline_monotonic_s > 0
        self.token_ceilings.append(max_tokens)
        return self.complete(prompt)


def _graph_payload(*, graph_id: str = "manager_graph", runner_ref: str = "worker.profile") -> dict:
    return {
        "schema_id": "robomex.elastic_graph.v2",
        "graph_id": graph_id,
        "revision": 1,
        "entry_activation": "work",
        "terminal_activations": ["work"],
        "activations": [
            {
                "activation_id": "work",
                "runner_kind": "coding_worker",
                "runner_ref": runner_ref,
                "estimated_budget": {
                    "model_calls": 1,
                    "tokens": 100,
                    "wall_time_ms": 1000,
                },
                "params": {"objective": "Produce a typed read-only proposal."},
            }
        ],
    }


def _snapshot(*, state_revision: int = 0) -> ManagerSnapshot:
    return ManagerSnapshot(
        snapshot_id=f"snapshot_{state_revision}",
        episode_id="episode_app",
        workflow_id="workflow_app",
        graph_id="pending_workflow_app",
        graph_revision=1,
        state_revision=state_revision,
    )


def _invocation(*, state_revision: int = 0, token_limit: int = 100_000) -> ManagerInvocation:
    return ManagerInvocation(
        invocation_id="manager-call-001",
        session_id="manager-session",
        session_revision=1,
        kind=ManagerCallKind.INITIAL,
        signal=ManagerSignal.INTENT_AUTHORING,
        snapshot=_snapshot(state_revision=state_revision),
        remaining_before=ManagerRemaining(
            initial_calls=1,
            reactivations=2,
            tokens=token_limit,
            candidates=3,
        ),
        token_limit=token_limit,
    )


def _context(
    *, available_artifacts: tuple[AvailableArtifact, ...] = ()
) -> WorkflowAuthoringContext:
    return WorkflowAuthoringContext(
        authoring_call_id="workflow_app_author",
        task_run_id="task_run_app",
        episode_id="episode_app",
        workflow_id="workflow_app",
        task="Place the bowl on the plate.",
        intent=SubgoalIntent(
            intent_id="intent_place",
            instruction="Place the bowl on the plate.",
            success_rubric="Fresh independent evidence shows the bowl on the plate.",
        ),
        embodied_state_revision=0,
    ).model_copy(update={"available_artifacts": available_artifacts})


def _decision(graph: dict, **payload_updates: object) -> dict:
    payload = {"graph": graph, **payload_updates}
    return {
        "schema_version": "robomex.manager_decision.v1",
        "action": "author_scaffold",
        "summary": "Bounded initial workflow scaffold.",
        "payload": payload,
        "candidate_delta": 0,
        "tokens_used": 0,
    }


def test_policy_manager_invoker_replays_call_id_without_second_model_call(
    tmp_path: Path,
) -> None:
    policy = _Policy(_decision(_graph_payload()))
    invoker = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")

    first = invoker.invoke(_invocation())
    second = invoker.invoke(_invocation())

    assert first == second
    assert len(policy.calls) == 1
    assert first.tokens_used > 0

    restarted = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")
    assert restarted.invoke(_invocation()) == first
    assert len(policy.calls) == 1


def test_policy_manager_invoker_denies_budget_and_oversized_prompt_before_api(
    tmp_path: Path,
) -> None:
    policy = _Policy(_decision(_graph_payload()))
    zero_authority = RunBudgetAuthority(
        tmp_path / "zero-budget.jsonl",
        budgets=RunBudgets(
            max_model_calls=0,
            max_tokens=100_000,
            max_wall_time_s=10,
            max_physical_actions=0,
            max_shadow_rollouts=0,
            max_candidates=1,
            max_recoveries=0,
        ),
    )
    denied = PolicyManagerInvoker(
        policy,
        ledger_root=tmp_path / "denied-manager-calls",
        run_budget_authority=zero_authority,
    )
    with pytest.raises(RunBudgetExceededError, match="model_calls"):
        denied.invoke(_invocation())
    assert policy.calls == []

    prompt_authority = RunBudgetAuthority(
        tmp_path / "prompt-budget.jsonl",
        budgets=RunBudgets(
            max_model_calls=1,
            max_tokens=10,
            max_wall_time_s=10,
            max_physical_actions=0,
            max_shadow_rollouts=0,
            max_candidates=1,
            max_recoveries=0,
        ),
    )
    oversized = PolicyManagerInvoker(
        policy,
        ledger_root=tmp_path / "oversized-manager-calls",
        run_budget_authority=prompt_authority,
    )
    with pytest.raises(ManagerResponseError, match="prompt exceeds"):
        oversized.invoke(_invocation(token_limit=10))
    assert policy.calls == []
    assert prompt_authority.snapshot().committed.model_calls == 0


def test_policy_manager_invoker_settles_total_grant_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    budgets = RunBudgets(
        max_model_calls=1,
        max_tokens=100_000,
        max_wall_time_s=10,
        max_physical_actions=0,
        max_shadow_rollouts=0,
        max_candidates=1,
        max_recoveries=0,
    )
    budget_path = tmp_path / "manager-budget.jsonl"
    authority = RunBudgetAuthority(budget_path, budgets=budgets)
    policy = _Policy(_decision(_graph_payload()))
    ledger_root = tmp_path / "manager-calls-budgeted"
    invoker = PolicyManagerInvoker(
        policy,
        ledger_root=ledger_root,
        run_budget_authority=authority,
        wall_time_ms=250,
    )

    first = invoker.invoke(_invocation())

    assert len(policy.calls) == 1
    assert 0 < policy.token_ceilings[0] < 100_000
    snapshot = authority.snapshot()
    assert snapshot.committed.model_calls == 1
    assert snapshot.committed.tokens == first.tokens_used
    assert snapshot.committed.wall_time_s == pytest.approx(0.25)

    replay_policy = _Policy(_decision(_graph_payload()))
    restarted = PolicyManagerInvoker(
        replay_policy,
        ledger_root=ledger_root,
        run_budget_authority=RunBudgetAuthority(budget_path, budgets=budgets),
        wall_time_ms=250,
    )
    assert restarted.invoke(_invocation()) == first
    assert replay_policy.calls == []


def test_policy_manager_failed_boundary_is_fully_charged_and_never_reentered(
    tmp_path: Path,
) -> None:
    budgets = RunBudgets(
        max_model_calls=1,
        max_tokens=100_000,
        max_wall_time_s=10,
        max_physical_actions=0,
        max_shadow_rollouts=0,
        max_candidates=1,
        max_recoveries=0,
    )
    budget_path = tmp_path / "failed-manager-budget.jsonl"
    ledger_root = tmp_path / "failed-manager-calls"
    authority = RunBudgetAuthority(budget_path, budgets=budgets)

    class FailingPolicy:
        calls = 0

        def complete_bounded(self, _prompt, *, max_tokens, deadline_monotonic_s):
            assert max_tokens > 0
            assert deadline_monotonic_s > 0
            self.calls += 1
            raise RuntimeError("synthetic provider crash")

    failing = FailingPolicy()
    invoker = PolicyManagerInvoker(
        failing,
        ledger_root=ledger_root,
        run_budget_authority=authority,
    )
    with pytest.raises(RuntimeError, match="synthetic provider crash"):
        invoker.invoke(_invocation())
    assert failing.calls == 1
    assert authority.snapshot().committed.tokens == 100_000

    replay_policy = _Policy(_decision(_graph_payload()))
    restarted = PolicyManagerInvoker(
        replay_policy,
        ledger_root=ledger_root,
        run_budget_authority=RunBudgetAuthority(budget_path, budgets=budgets),
    )
    with pytest.raises(ManagerResponseError, match="cannot be re-entered"):
        restarted.invoke(_invocation())
    assert replay_policy.calls == []


def test_policy_manager_timeout_receives_deadline_and_charges_full_grant(
    tmp_path: Path,
) -> None:
    budgets = RunBudgets(
        max_model_calls=1,
        max_tokens=100_000,
        max_wall_time_s=10,
        max_physical_actions=0,
        max_shadow_rollouts=0,
        max_candidates=1,
        max_recoveries=0,
    )
    authority = RunBudgetAuthority(tmp_path / "timeout-budget.jsonl", budgets=budgets)

    class TimeoutPolicy:
        deadline = None

        def complete_bounded(self, _prompt, *, max_tokens, deadline_monotonic_s):
            assert max_tokens > 0
            self.deadline = deadline_monotonic_s
            raise TimeoutError("synthetic bounded timeout")

    policy = TimeoutPolicy()
    invoker = PolicyManagerInvoker(
        policy,
        ledger_root=tmp_path / "timeout-manager-calls",
        run_budget_authority=authority,
        wall_time_ms=50,
    )

    with pytest.raises(TimeoutError, match="synthetic bounded timeout"):
        invoker.invoke(_invocation())

    assert policy.deadline is not None
    snapshot = authority.snapshot()
    assert snapshot.committed.model_calls == 1
    assert snapshot.committed.tokens == 100_000
    assert snapshot.committed.wall_time_s == pytest.approx(0.05)


def test_policy_manager_invoker_rejects_call_id_content_rebinding(
    tmp_path: Path,
) -> None:
    policy = _Policy(_decision(_graph_payload()))
    invoker = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")
    invoker.invoke(_invocation())

    with pytest.raises(ManagerResponseError, match="rebound"):
        invoker.invoke(_invocation(state_revision=1))
    assert len(policy.calls) == 1


def test_manager_workflow_author_compiles_closed_graph_and_serializes_session(
    tmp_path: Path,
) -> None:
    graph_payload = _graph_payload()
    policy = _Policy(_decision(graph_payload))
    invoker = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")
    author = ManagerWorkflowAuthor(
        invoker,
        limits=ManagerLimits(
            max_initial_calls=1,
            max_reactivations=2,
            max_tokens=100_000,
            max_tokens_per_call=100_000,
            max_candidates=3,
        ),
        catalog_refs=("worker.profile",),
    )

    blueprint = author.author(_context())

    expected = ElasticGraphCompiler().compile(ElasticGraphSpec.model_validate(graph_payload))
    assert blueprint.graph.digest == expected.digest
    assert blueprint.manager_session is not None
    assert blueprint.manager_session.usage.initial_calls == 1
    assert blueprint.manager_invoker is invoker
    assert blueprint.external_refs == {}


def test_initial_manager_workflow_author_uses_bound_global_model_budget(
    tmp_path: Path,
) -> None:
    authority = RunBudgetAuthority(
        tmp_path / "author-budget.jsonl",
        budgets=RunBudgets(
            max_model_calls=1,
            max_tokens=100_000,
            max_wall_time_s=10,
            max_physical_actions=0,
            max_shadow_rollouts=0,
            max_candidates=1,
            max_recoveries=0,
        ),
    )
    policy = _Policy(_decision(_graph_payload()))
    invoker = PolicyManagerInvoker(
        policy,
        ledger_root=tmp_path / "author-manager-calls",
    )
    author = ManagerWorkflowAuthor(
        invoker,
        limits=ManagerLimits(
            max_initial_calls=1,
            max_reactivations=0,
            max_tokens=100_000,
            max_tokens_per_call=100_000,
            max_candidates=0,
        ),
    )
    author.bind_run_budget_authority(authority)

    blueprint = author.author(_context())

    assert blueprint.manager_session is not None
    assert len(policy.calls) == 1
    assert authority.snapshot().committed.model_calls == 1


def test_manager_workflow_author_cannot_forge_external_artifact_reference(
    tmp_path: Path,
) -> None:
    forged = {
        "artifact_id": "artifact_forged",
        "content_digest": "sha256:" + "f" * 64,
    }
    policy = _Policy(_decision(_graph_payload(), external_refs={"scene": forged}))
    invoker = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")
    author = ManagerWorkflowAuthor(
        invoker,
        limits=ManagerLimits(
            max_initial_calls=1,
            max_reactivations=0,
            max_tokens=100_000,
            max_tokens_per_call=100_000,
            max_candidates=0,
        ),
    )
    admitted = AvailableArtifact(
        workflow_id="previous_workflow",
        activation_id="observe",
        attempt=1,
        port="scene",
        schema_id="robomex.scene.v1",
        artifact_id="artifact_admitted",
        content_digest="sha256:" + "a" * 64,
    )

    with pytest.raises(V2ApplicationError, match="admitted artifact inventory"):
        author.author(_context(available_artifacts=(admitted,)))


@pytest.mark.parametrize(
    ("metadata", "planner_budget", "match"),
    [
        (
            {},
            PlannerCallBudget(token_grant=257),
            "differs from the manifest-pinned grant",
        ),
        (
            {"planner_call_budget": {"token_grant": 0}},
            PlannerCallBudget(),
            "malformed",
        ),
    ],
)
def test_v2_agent_rejects_unpinned_or_malformed_planner_grant(
    tmp_path: Path,
    metadata: dict,
    planner_budget: PlannerCallBudget,
    match: str,
) -> None:
    config, base_manifest, dependencies = _fixtures(tmp_path)
    manifest = _rebuild_manifest(base_manifest, metadata=metadata)
    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )

    with pytest.raises(V2ApplicationError, match=match):
        RoboMExV2Agent(
            V2AgentConfig(
                application=application,
                planner=ScriptedIntentPlanner((PlannerDecision(kind=PlannerDecisionKind.DONE),)),
                author=ScriptedWorkflowAuthor({}),
                planner_budget=planner_budget,
            )
        )


@pytest.mark.parametrize(
    ("metadata", "config_updates", "match"),
    [
        ({}, {"max_intents": 9}, "task limits differ"),
        (
            {"task_orchestration_limits": {"max_intents": 0}},
            {},
            "task_orchestration_limits is malformed",
        ),
    ],
)
def test_v2_agent_rejects_unpinned_or_malformed_task_limits(
    tmp_path: Path,
    metadata: dict,
    config_updates: dict,
    match: str,
) -> None:
    config, base_manifest, dependencies = _fixtures(tmp_path)
    manifest = _rebuild_manifest(base_manifest, metadata=metadata)
    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    values = {
        "application": application,
        "planner": ScriptedIntentPlanner((PlannerDecision(kind=PlannerDecisionKind.DONE),)),
        "author": ScriptedWorkflowAuthor({}),
        **config_updates,
    }

    with pytest.raises(V2ApplicationError, match=match):
        RoboMExV2Agent(V2AgentConfig(**values))


def test_v2_agent_requires_manager_author_limits_to_match_manifest(
    tmp_path: Path,
) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    manager_author = ManagerWorkflowAuthor(
        PolicyManagerInvoker(
            _Policy(_decision(_graph_payload(runner_ref="worker"))),
            ledger_root=tmp_path / "manager-limit-calls",
        ),
        limits=ManagerLimits(max_tokens=5_000, max_tokens_per_call=1_024),
    )

    with pytest.raises(V2ApplicationError, match="ManagerWorkflowAuthor limits differ"):
        RoboMExV2Agent(
            V2AgentConfig(
                application=application,
                planner=ScriptedIntentPlanner((PlannerDecision(kind=PlannerDecisionKind.DONE),)),
                author=manager_author,
            )
        )


@pytest.mark.parametrize(
    ("metadata", "wall_time_ms", "match"),
    [
        ({}, 4_999, "call budget differs"),
        (
            {"manager_policy_call_budget": {"wall_time_ms": 0}},
            5_000,
            "manager_policy_call_budget is malformed",
        ),
    ],
)
def test_v2_agent_rejects_unpinned_or_malformed_manager_policy_wall_grant(
    tmp_path: Path,
    metadata: dict,
    wall_time_ms: int,
    match: str,
) -> None:
    config, base_manifest, dependencies = _fixtures(tmp_path)
    manifest = _rebuild_manifest(base_manifest, metadata=metadata)
    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    manager_author = ManagerWorkflowAuthor(
        PolicyManagerInvoker(
            _Policy(_decision(_graph_payload(runner_ref="worker"))),
            ledger_root=tmp_path / "manager-call-budget-calls",
            wall_time_ms=wall_time_ms,
        )
    )

    with pytest.raises(V2ApplicationError, match=match):
        RoboMExV2Agent(
            V2AgentConfig(
                application=application,
                planner=ScriptedIntentPlanner(
                    (PlannerDecision(kind=PlannerDecisionKind.DONE),)
                ),
                author=manager_author,
            )
        )


class _StaticAuthor:
    def __init__(self, blueprint: WorkflowBlueprint) -> None:
        self.blueprint = blueprint

    def author(self, _context: WorkflowAuthoringContext) -> WorkflowBlueprint:
        return self.blueprint


def test_manifest_pinned_author_rejects_unsealed_initial_graph() -> None:
    graph = ElasticGraphCompiler().compile(_graph_payload())
    author = _StaticAuthor(WorkflowBlueprint(graph=graph, external_refs={}))
    wrong_manifest = SimpleNamespace(
        graph_digest="sha256:" + "0" * 64,
        metadata={},
    )
    pinned = ManifestPinnedWorkflowAuthor(author, wrong_manifest)

    with pytest.raises(V2ApplicationError, match="not manifest-pinned"):
        pinned.author(_context())

    allowed_manifest = SimpleNamespace(
        graph_digest="sha256:" + "0" * 64,
        metadata={"allowed_initial_graph_digests": [f"sha256:{graph.digest}"]},
    )
    assert ManifestPinnedWorkflowAuthor(author, allowed_manifest).author(_context()).graph == graph


def test_manager_workflow_author_accepts_only_digest_bound_inventory_ref(
    tmp_path: Path,
) -> None:
    admitted = AvailableArtifact(
        workflow_id="previous_workflow",
        activation_id="observe",
        attempt=1,
        port="scene",
        schema_id="robomex.scene.v1",
        artifact_id="artifact_admitted",
        content_digest="sha256:" + "a" * 64,
    )
    policy = _Policy(
        _decision(
            _graph_payload(),
            external_refs={"scene": admitted.ref.to_mapping()},
        )
    )
    invoker = PolicyManagerInvoker(policy, ledger_root=tmp_path / "manager_calls")
    author = ManagerWorkflowAuthor(
        invoker,
        limits=ManagerLimits(
            max_initial_calls=1,
            max_reactivations=0,
            max_tokens=100_000,
            max_tokens_per_call=100_000,
            max_candidates=0,
        ),
    )

    blueprint = author.author(_context(available_artifacts=(admitted,)))

    assert blueprint.external_refs == {
        "scene": ResolvedArtifactRef(
            admitted.artifact_id,
            admitted.content_digest,
        )
    }


def test_v2_agent_runs_manifest_pinned_task_and_restarts_without_reexecution(
    tmp_path: Path,
) -> None:
    graph = ElasticGraphCompiler().compile(
        _graph_payload(graph_id="entrypoint_graph", runner_ref="worker")
    )
    config, base_manifest, base_dependencies = _fixtures(tmp_path)
    manifest = _rebuild_manifest(
        base_manifest,
        graph_digest=f"sha256:{graph.digest}",
        metadata={"task_orchestration_limits": {"max_intents": 2}},
    )
    config = config.model_copy(update={"graph_digest": manifest.graph_digest})
    provider = InMemoryAgentProvider(
        lambda _profile, _spec, _isolation: ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS
        )
    )
    dependencies = V2RuntimeDependencies(
        contract_catalog=base_dependencies.contract_catalog,
        actor_providers={"memory": provider},
        actor_profiles=base_dependencies.actor_profiles,
        action_backends=base_dependencies.action_backends,
        observation_backends=base_dependencies.observation_backends,
        shadow_backends=base_dependencies.shadow_backends,
    )
    intent = SubgoalIntent(
        intent_id="intent_entrypoint",
        instruction="Inspect the workspace.",
        success_rubric="The inspection workflow succeeds.",
    )
    planner = ScriptedIntentPlanner(
        (
            PlannerDecision(kind=PlannerDecisionKind.OPEN_INTENT, intent=intent),
            PlannerDecision(kind=PlannerDecisionKind.DONE, reason="task verified"),
        )
    )
    author = ScriptedWorkflowAuthor(
        {intent.intent_id: WorkflowBlueprint(graph=graph, external_refs={})}
    )

    first_application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    first_agent = RoboMExV2Agent(
        V2AgentConfig(
            application=first_application,
            planner=planner,
            author=author,
            max_intents=2,
        )
    )
    first = first_agent.run()

    assert first.status is TaskRunStatus.SUCCEEDED
    assert len(first.outcomes) == 1
    assert first.outcomes[0].status.value == "succeeded"
    calls_after_first = provider.calls
    assert [call[0] for call in calls_after_first].count("invoke") == 1
    first_agent.close()

    restarted_application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    restarted = RoboMExV2Agent(
        V2AgentConfig(
            application=restarted_application,
            planner=planner,
            author=author,
            max_intents=2,
        )
    ).run()

    assert restarted == first
    assert provider.calls == calls_after_first
