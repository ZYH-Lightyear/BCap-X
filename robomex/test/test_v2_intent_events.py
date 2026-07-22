from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from robomex.orchestration.intent import (
    BudgetHint,
    EntityRef,
    IntentOutcome,
    IntentStatus,
    SubgoalIntent,
)
from robomex.runtime.events import (
    ActionOutcome,
    ArenaDecision,
    ArtifactPublished,
    ControlOutcome,
    GraphPatchOutcome,
    LifecycleEvent,
    ManagerInvocationOutcome,
    MonitorFinding,
    NodeOutcomeEvent,
    PatchRequest,
    RosterUpdate,
    StateProposal,
    deserialize_runtime_event,
    runtime_event_to_json,
    serialize_runtime_event,
)


def _envelope(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "event_id": "evt-replay-0001",
        "episode_id": "episode-01",
        "workflow_id": "place-bowl-01",
        "timestamp": datetime(
            2026, 7, 22, 3, 0, tzinfo=timezone.utc  # noqa: UP017 - Python 3.10
        ),
        "source": "test.runtime",
    }
    values.update(overrides)
    return values


def test_subgoal_intent_keeps_open_language_and_typed_boundaries() -> None:
    intent = SubgoalIntent(
        intent_id="place_bowl",
        revision=2,
        instruction="微调并将 bowl_1 稳定放到 plate_1 上",
        success_rubric="释放后碗稳定留在盘中，夹爪安全离开",
        entity_refs=[
            "bowl_1",
            EntityRef(entity_id="plate_1", role="target", entity_type="plate"),
        ],
        protected_invariants=["authoritative_action_lane", "no_unsealed_motion"],
        evidence_types=["held_object_pose.v1", "support_relation.v1"],
        budget_hint=BudgetHint(
            max_model_calls=8,
            max_physical_actions=4,
            max_candidates=3,
        ),
        context={"risk": "high", "alignment_tolerance_m": 0.01},
    )

    assert intent.schema_version == "robomex.subgoal_intent.v1"
    assert intent.entity_refs[0].entity_id == "bowl_1"
    assert intent.entity_refs[1].role == "target"
    assert intent.budget_hint.max_candidates == 3
    assert SubgoalIntent.model_validate(intent.model_dump(mode="json")) == intent


def test_intent_contracts_fail_closed_on_version_extra_and_invalid_budget() -> None:
    payload = {
        "schema_version": "robomex.subgoal_intent.v2",
        "intent_id": "pick",
        "instruction": "pick the bowl",
        "success_rubric": "the bowl follows the gripper",
    }
    with pytest.raises(ValidationError, match="schema_version"):
        SubgoalIntent.model_validate(payload)

    payload["schema_version"] = "robomex.subgoal_intent.v1"
    payload["future_field"] = True
    with pytest.raises(ValidationError, match="future_field"):
        SubgoalIntent.model_validate(payload)

    with pytest.raises(ValidationError, match="max_physical_actions"):
        BudgetHint(max_physical_actions=-1)


def test_intent_outcome_is_versioned_and_replayable() -> None:
    outcome = IntentOutcome(
        episode_id="episode-01",
        workflow_id="pick-bowl-01",
        intent_id="pick_bowl",
        intent_revision=1,
        status=IntentStatus.SUCCEEDED,
        summary="attachment verified",
        evidence_refs=["artifact:evidence-7"],
        final_state_revision=3,
        metrics={"physical_actions": 1.0},
    )

    restored = IntentOutcome.model_validate_json(outcome.model_dump_json())
    assert restored == outcome
    assert restored.status is IntentStatus.SUCCEEDED


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [
        (
            NodeOutcomeEvent(
                **_envelope(),
                activation_id="activation-1",
                node_id="align",
                outcome=ControlOutcome.NEEDS_ADJUSTMENT,
                artifact_ids=["artifact:alignment-1"],
                graph_revision=1,
            ),
            NodeOutcomeEvent,
        ),
        (
            LifecycleEvent(
                **_envelope(event_id="evt-2"),
                actor_id="monitor-1",
                transition="spawned",
                actor_type="monitor",
            ),
            LifecycleEvent,
        ),
        (
            ArtifactPublished(
                **_envelope(event_id="evt-3"),
                artifact_id="artifact:pose-1",
                schema_id="robomex.held_object_pose.v1",
                digest="sha256:1234",
                producer_activation_id="activation-1",
            ),
            ArtifactPublished,
        ),
        (
            MonitorFinding(
                **_envelope(event_id="evt-4"),
                monitor_id="attachment-monitor",
                finding="attachment_anomaly",
                severity="critical",
                confidence=0.95,
                evidence_refs=["artifact:frame-7"],
            ),
            MonitorFinding,
        ),
        (
            ActionOutcome(
                **_envelope(event_id="evt-5"),
                action_id="action-3",
                status="interrupted",
                reason="attachment_anomaly",
                triggering_finding_id="finding-4",
            ),
            ActionOutcome,
        ),
        (
            StateProposal(
                **_envelope(event_id="evt-6"),
                base_state_revision=2,
                updates={"attachment.status": "unknown"},
                evidence_refs=["artifact:frame-7"],
            ),
            StateProposal,
        ),
        (
            PatchRequest(
                **_envelope(event_id="evt-7"),
                base_graph_revision=1,
                reason="all bounded candidates were rejected",
                requested_scope="alignment_recovery",
            ),
            PatchRequest,
        ),
        (
            RosterUpdate(
                **_envelope(event_id="evt-8"),
                operation="spawn",
                actor_id="motion-candidate-2",
                slot_id="placement-arena",
                actor_profile_ref="profile:motion-small-v1",
            ),
            RosterUpdate,
        ),
        (
            ArenaDecision(
                **_envelope(event_id="evt-9"),
                arena_run_id="arena-1",
                slot_id="placement-arena",
                graph_revision=1,
                selected_candidate_id="motion-candidate-2",
                considered_candidate_ids=("motion-candidate-1", "motion-candidate-2"),
                rejected_candidate_ids=("motion-candidate-1",),
                selection_reason="passed hard gates with lowest deterministic rank",
                selected_hypothesis_ref="artifact:hypothesis-2",
            ),
            ArenaDecision,
        ),
        (
            GraphPatchOutcome(
                **_envelope(event_id="evt-10"),
                patch_id="patch-1",
                slot_id="placement-recovery",
                operation="fill_slot",
                accepted=True,
                before_revision=1,
                after_revision=2,
                before_digest="digest-before",
                after_digest="digest-after",
            ),
            GraphPatchOutcome,
        ),
        (
            ManagerInvocationOutcome(
                **_envelope(event_id="evt-11"),
                session_id="manager-1",
                session_revision=2,
                signal="recovery",
                invoked=True,
                action="repair_frontier",
            ),
            ManagerInvocationOutcome,
        ),
    ],
)
def test_all_runtime_event_kinds_round_trip_through_closed_union(
    event: object,
    expected_type: type[object],
) -> None:
    payload = serialize_runtime_event(event)  # type: ignore[arg-type]
    restored = deserialize_runtime_event(payload)
    restored_from_json = deserialize_runtime_event(runtime_event_to_json(event))  # type: ignore[arg-type]

    assert type(restored) is expected_type
    assert restored == event
    assert restored_from_json == event
    assert payload["event_id"] == event.event_id  # type: ignore[attr-defined]
    assert payload["episode_id"] == "episode-01"
    assert payload["workflow_id"] == "place-bowl-01"
    assert payload["source"] == "test.runtime"


def test_runtime_event_unknown_kind_and_extra_payload_fail_closed() -> None:
    event = NodeOutcomeEvent(
        **_envelope(),
        activation_id="activation-1",
        node_id="align",
        outcome="success",
    )
    payload = serialize_runtime_event(event)
    payload["kind"] = "future_unregistered_event"
    with pytest.raises(ValidationError, match="future_unregistered_event"):
        deserialize_runtime_event(payload)

    payload = serialize_runtime_event(event)
    payload["untrusted_extension"] = {"execute": True}
    with pytest.raises(ValidationError, match="untrusted_extension"):
        deserialize_runtime_event(payload)


def test_node_outcome_vocabulary_and_schema_version_are_closed() -> None:
    payload = serialize_runtime_event(
        NodeOutcomeEvent(
            **_envelope(),
            activation_id="activation-1",
            node_id="align",
            outcome="success",
        )
    )
    payload["outcome"] = "looks_good"
    with pytest.raises(ValidationError, match="outcome"):
        deserialize_runtime_event(payload)

    payload["outcome"] = "success"
    payload["schema_version"] = "robomex.runtime_event.v2"
    with pytest.raises(ValidationError, match="schema_version"):
        deserialize_runtime_event(payload)


def test_event_timestamp_must_be_aware_and_is_normalized_to_utc() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        NodeOutcomeEvent(
            **_envelope(timestamp=datetime(2026, 7, 22, 11, 0)),
            activation_id="activation-1",
            node_id="align",
            outcome="success",
        )

    event = NodeOutcomeEvent(
        **_envelope(
            timestamp=datetime(
                2026,
                7,
                22,
                11,
                0,
                tzinfo=timezone(timedelta(hours=8)),
            )
        ),
        activation_id="activation-1",
        node_id="align",
        outcome="success",
    )
    assert event.timestamp == datetime(
        2026, 7, 22, 3, 0, tzinfo=timezone.utc  # noqa: UP017 - Python 3.10
    )
    assert event.timestamp.utcoffset() == timedelta(0)


def test_event_id_defaults_but_can_be_injected_for_replay() -> None:
    generated = NodeOutcomeEvent(
        episode_id="episode-01",
        workflow_id="workflow-01",
        source="scheduler",
        activation_id="activation-1",
        node_id="align",
        outcome="success",
    )
    injected = generated.model_copy(update={"event_id": "recorded-event-42"})

    assert len(generated.event_id) == 32
    assert deserialize_runtime_event(serialize_runtime_event(injected)).event_id == "recorded-event-42"
