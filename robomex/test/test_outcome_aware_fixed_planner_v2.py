from __future__ import annotations

from robomex.orchestration.intent import (
    IntentOutcome,
    IntentStatus,
    SubgoalIntent,
)
from robomex.orchestration.task_orchestrator import (
    EpisodePlanningContext,
    OutcomeAwareFixedIntentPlanner,
    PlannerDecisionKind,
)


def _intent() -> SubgoalIntent:
    return SubgoalIntent(
        intent_id="place-bowl",
        instruction="place the held bowl on the plate",
        success_rubric="independent evidence asserts bowl supported_by plate",
    )


def _context(*, outcome: IntentOutcome | None = None) -> EpisodePlanningContext:
    intent = _intent()
    return EpisodePlanningContext(
        planner_call_id="planner-1" if outcome is None else "planner-2",
        task_run_id="task-run",
        episode_id="episode",
        task="place the held bowl on the plate",
        intent_index=0 if outcome is None else 1,
        prior_intents=() if outcome is None else (intent,),
        prior_outcomes=() if outcome is None else (outcome,),
        embodied_state_revision=0,
        remaining_intents=1,
        remaining_planner_calls=2,
    )


def _outcome(status: IntentStatus) -> IntentOutcome:
    return IntentOutcome(
        episode_id="episode",
        workflow_id="workflow",
        intent_id="place-bowl",
        intent_revision=1,
        status=status,
        reason="fixture terminal reason",
    )


def test_fixed_planner_closes_only_after_success() -> None:
    planner = OutcomeAwareFixedIntentPlanner(_intent())
    assert planner.next_intent(_context()).kind is PlannerDecisionKind.OPEN_INTENT
    assert (
        planner.next_intent(_context(outcome=_outcome(IntentStatus.SUCCEEDED))).kind
        is PlannerDecisionKind.DONE
    )


def test_fixed_planner_does_not_turn_failed_workflow_into_done() -> None:
    planner = OutcomeAwareFixedIntentPlanner(_intent())
    decision = planner.next_intent(_context(outcome=_outcome(IntentStatus.FAILED)))
    assert decision.kind is PlannerDecisionKind.BLOCKED
    assert "failed" in decision.reason
