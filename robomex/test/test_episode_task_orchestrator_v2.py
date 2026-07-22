from __future__ import annotations

import pytest

from robomex.agents.planner import ReactivePlanner, ScriptedPlannerPolicy
from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExternalBinding,
    PortSpecV2,
    RunnerKind,
)
from robomex.orchestration.actors import (
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.task_orchestrator import (
    EpisodeOrchestrator,
    EpisodePlanningContext,
    PlannerDecision,
    PlannerDecisionKind,
    ReactivePlannerIntentAdapter,
    ScriptedIntentPlanner,
    TaskRunStatus,
    TaskStateConflictError,
    WorkflowBlueprint,
)
from robomex.runtime.events import ControlOutcome
from robomex.skills import SkillLibrary

VALUE_SCHEMA = "test.task_value.v1"


def _graph_one():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="task-first",
            entry_activation="produce",
            terminal_activations=("produce",),
            activations=(
                ActivationSpec(
                    activation_id="produce",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.produce",
                    outputs=(PortSpecV2(name="value", schema_id=VALUE_SCHEMA),),
                ),
            ),
        )
    )


def _graph_two():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="task-second",
            entry_activation="consume",
            terminal_activations=("consume",),
            activations=(
                ActivationSpec(
                    activation_id="consume",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="test.consume",
                    inputs=(PortSpecV2(name="value", schema_id=VALUE_SCHEMA),),
                    bindings=(
                        ExternalBinding(
                            input_port="value",
                            ref="episode://prior/value",
                            schema_id=VALUE_SCHEMA,
                        ),
                    ),
                ),
            ),
        )
    )


def _runtime(root, *, calls: list[str]) -> EpisodeRuntime:
    def provider(profile, invocation, isolation):
        del profile, isolation
        activation = str(invocation.metadata["activation_id"])
        calls.append(activation)
        if activation == "produce":
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    ArtifactEmission(
                        port="value",
                        schema_id=VALUE_SCHEMA,
                        payload={"value": 7},
                    ),
                ),
            )
        assert activation == "consume"
        assert set(invocation.inputs) == {"value"}
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    runtime = EpisodeRuntime(
        episode_id="task-episode",
        episode_root=root,
        actors=ActorRegistry(
            {"in_memory": InMemoryAgentProvider(provider)},
            namespace_root="task-episode",
            workspace_root=root / "actors",
        ),
    )
    runtime.register_actor_profile(
        "test.produce",
        ActorProfile(profile_id="produce", runner_kind="coding_worker"),
    )
    runtime.register_actor_profile(
        "test.consume",
        ActorProfile(profile_id="consume", runner_kind="coding_worker"),
    )
    return runtime


class _CrossWorkflowAuthor:
    def __init__(self) -> None:
        self.contexts = []

    def author(self, context):
        self.contexts.append(context)
        if context.intent.intent_id == "first":
            return WorkflowBlueprint(graph=_graph_one(), external_refs={})
        prior = [
            item for item in context.available_artifacts if item.schema_id == VALUE_SCHEMA
        ]
        assert len(prior) == 1
        return WorkflowBlueprint(
            graph=_graph_two(),
            external_refs={"episode://prior/value": prior[0].ref},
        )


def _intent(intent_id: str) -> SubgoalIntent:
    return SubgoalIntent(
        intent_id=intent_id,
        instruction=f"execute {intent_id}",
        success_rubric=f"{intent_id} is verified",
    )


def test_orchestrator_runs_multiple_intents_with_explicit_cross_workflow_ref(
    tmp_path,
) -> None:
    calls: list[str] = []
    runtime = _runtime(tmp_path / "episode", calls=calls)
    planner = ScriptedIntentPlanner(
        (
            PlannerDecision(kind=PlannerDecisionKind.OPEN_INTENT, intent=_intent("first")),
            PlannerDecision(kind=PlannerDecisionKind.OPEN_INTENT, intent=_intent("second")),
            PlannerDecision(kind=PlannerDecisionKind.DONE, reason="task verified"),
        )
    )
    author = _CrossWorkflowAuthor()
    orchestrator = EpisodeOrchestrator(runtime, planner=planner, author=author)
    orchestrator.start(task_run_id="task_run", task="do two causal steps")

    result = orchestrator.run("task_run")

    assert result.status is TaskRunStatus.SUCCEEDED
    assert [item.intent_id for item in result.intents] == ["first", "second"]
    assert [item.status.value for item in result.outcomes] == ["succeeded", "succeeded"]
    assert calls == ["produce", "consume"]
    assert author.contexts[1].prior_outcomes == (result.outcomes[0],)
    assert any(
        item.schema_id == VALUE_SCHEMA
        for item in author.contexts[1].available_artifacts
    )


class _CrashOncePlanner:
    def __init__(self) -> None:
        self.call_ids: list[str] = []

    def next_intent(self, context):
        self.call_ids.append(context.planner_call_id)
        raise KeyboardInterrupt("synthetic planner process crash")


def test_restart_reuses_pending_planner_call_identity_and_budget(tmp_path) -> None:
    root = tmp_path / "episode"
    runtime = _runtime(root, calls=[])
    crashing = _CrashOncePlanner()
    author = _CrossWorkflowAuthor()
    first = EpisodeOrchestrator(runtime, planner=crashing, author=author)
    first.start(
        task_run_id="restart_task",
        task="one step",
        max_intents=1,
        max_planner_calls=2,
    )

    with pytest.raises(KeyboardInterrupt):
        first.run("restart_task")

    pending = first.ledger.latest("restart_task")
    assert pending is not None
    assert pending.planner_calls_started == 1
    assert pending.pending_planner_call_id == crashing.call_ids[0]

    recovered_runtime = _runtime(root, calls=[])
    recovered_planner = ScriptedIntentPlanner(
        (
            PlannerDecision(
                kind=PlannerDecisionKind.OPEN_INTENT,
                intent=_intent("first"),
            ),
            PlannerDecision(kind=PlannerDecisionKind.DONE),
        )
    )
    recovered = EpisodeOrchestrator(
        recovered_runtime,
        planner=recovered_planner,
        author=_CrossWorkflowAuthor(),
    )

    result = recovered.run("restart_task")

    assert result.status is TaskRunStatus.SUCCEEDED
    assert recovered_planner.requests[0].planner_call_id == crashing.call_ids[0]
    terminal = recovered.ledger.latest("restart_task")
    assert terminal is not None and terminal.planner_calls_started == 2


def test_task_run_identity_is_immutable(tmp_path) -> None:
    runtime = _runtime(tmp_path / "episode", calls=[])
    orchestrator = EpisodeOrchestrator(
        runtime,
        planner=ScriptedIntentPlanner((PlannerDecision(kind="done"),)),
        author=_CrossWorkflowAuthor(),
    )
    orchestrator.start(task_run_id="immutable", task="original", max_intents=2)

    with pytest.raises(TaskStateConflictError, match="already bound"):
        orchestrator.start(task_run_id="immutable", task="different", max_intents=2)


def test_reactive_planner_adapter_emits_versioned_intent_and_replays_call_id(
    tmp_path,
) -> None:
    planner = ReactivePlanner(
        SkillLibrary(tmp_path / "skills"),
        ScriptedPlannerPolicy(
            [
                "Goal: pick the bowl\nPostcondition: the bowl is held",
                "DONE",
            ]
        ),
    )
    adapter = ReactivePlannerIntentAdapter(planner)
    context = EpisodePlanningContext(
        planner_call_id="planner-call-1",
        task_run_id="task",
        episode_id="task-episode",
        task="put the bowl on the plate",
        intent_index=0,
        embodied_state_revision=0,
        remaining_intents=2,
        remaining_planner_calls=2,
    )

    first = adapter.next_intent(context)
    replay = adapter.next_intent(context)
    done = adapter.next_intent(
        context.model_copy(
            update={
                "planner_call_id": "planner-call-2",
            }
        )
    )

    assert first.kind is PlannerDecisionKind.OPEN_INTENT
    assert first.intent is not None
    assert first.intent.intent_id.startswith("intent_000_")
    assert replay == first
    assert done.kind is PlannerDecisionKind.DONE
