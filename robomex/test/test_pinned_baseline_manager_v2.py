from __future__ import annotations

from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    RunnerKind,
)
from robomex.orchestration.application import (
    ManagerWorkflowAuthor,
    PinnedBaselineManagerInvoker,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerSignal,
    ManagerSnapshot,
)
from robomex.orchestration.task_orchestrator import WorkflowAuthoringContext


def _graph():
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="fixed_baseline_graph",
            entry_activation="complete",
            terminal_activations=("complete",),
            activations=(
                ActivationSpec(
                    activation_id="complete",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.complete",
                ),
            ),
        )
    )


def test_pinned_baseline_manager_authors_exact_graph_without_model_tokens() -> None:
    graph = _graph()
    invoker = PinnedBaselineManagerInvoker(graph)
    author = ManagerWorkflowAuthor(invoker)
    context = WorkflowAuthoringContext(
        authoring_call_id="author-call-1",
        task_run_id="task-run",
        episode_id="episode-1",
        workflow_id="workflow-1",
        task="finish safely",
        intent=SubgoalIntent(
            intent_id="intent-1",
            instruction="finish safely",
            success_rubric="the fixed workflow completes",
        ),
        embodied_state_revision=0,
    )

    blueprint = author.author(context)

    assert blueprint.graph.digest == graph.digest
    assert blueprint.external_refs == {}
    assert blueprint.manager_session is not None
    assert blueprint.manager_session.usage.initial_calls == 1
    assert blueprint.manager_session.usage.tokens == 0
    assert blueprint.manager_invoker is invoker


def test_pinned_baseline_manager_closes_on_exception_instead_of_inventing_patch() -> None:
    graph = _graph()
    invoker = PinnedBaselineManagerInvoker(graph)
    author = ManagerWorkflowAuthor(invoker)
    context = WorkflowAuthoringContext(
        authoring_call_id="author-call-1",
        task_run_id="task-run",
        episode_id="episode-1",
        workflow_id="workflow-1",
        task="finish safely",
        intent=SubgoalIntent(
            intent_id="intent-1",
            instruction="finish safely",
            success_rubric="the fixed workflow completes",
        ),
        embodied_state_revision=0,
    )
    blueprint = author.author(context)
    session = blueprint.manager_session
    assert session is not None
    step = session.wake(
        ManagerSignal.RECOVERY,
        ManagerSnapshot(
            snapshot_id="recovery-snapshot-1",
            episode_id="episode-1",
            workflow_id="workflow-1",
            graph_id=graph.spec.graph_id,
            graph_revision=graph.spec.revision,
            state_revision=0,
            triggering_event={"event_id": "recovery-event-1"},
        ),
        invoker,
    )

    assert step.invoked is True
    assert step.decision is not None
    assert step.decision.action is ManagerAction.CLOSE
    assert "graph_patch" not in step.decision.payload
    assert step.decision.tokens_used == 0
