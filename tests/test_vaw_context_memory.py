from __future__ import annotations

from tests.test_vaw_context_runtime import FakeContextApi, FailedMotionBackend
from vaw.context_runtime.context_projection import project_embodied_state
from vaw.context_runtime.memory import (
    InteractionEvent,
    InteractionMemory,
    interaction_outcome,
)
from vaw.context_runtime.protocol import main_function_registry
from vaw.context_runtime.workspace import ContextWorkspace


def _event(
    turn: int,
    kind: str,
    function: str,
    arguments: dict | None = None,
    *,
    outcome: str = "ok",
    revision_before: int = 1,
    revision_after: int = 1,
) -> InteractionEvent:
    return InteractionEvent(
        turn=turn,
        kind=kind,  # type: ignore[arg-type]
        function=function,
        arguments=arguments or {},
        outcome=outcome,
        revision_before=revision_before,
        revision_after=revision_after,
    )


def test_interaction_memory_keeps_full_trace_and_only_folds_prompt_repeats() -> None:
    memory = InteractionMemory()
    memory.record(_event(1, "call", "detect_region", {"query": "can"}))
    memory.record(_event(2, "call", "detect_region", {"query": "can"}))
    memory.record(
        _event(
            3,
            "action",
            "open_gripper",
            outcome="completed",
            revision_before=1,
            revision_after=2,
        ),
        effect_channel="gripper",
    )

    assert len(memory.snapshot()) == 3
    assert memory.prompt_lines() == [
        't1-t2 [函数调用] detect_region({"query":"can"}) -> 结果=ok (r1->r1) ×2',
        "t3 [物理动作] open_gripper({}) -> 结果=completed (r1->r2)",
    ]
    assert memory.last_action.function == "open_gripper"
    assert memory.last_gripper_action.function == "open_gripper"


def test_prompt_memory_uses_only_the_latest_five_committed_calls() -> None:
    memory = InteractionMemory()
    for turn in range(1, 8):
        memory.record(_event(turn, "call", f"function_{turn}"))

    assert len(memory.snapshot()) == 7
    projection = memory.prompt_lines()
    assert len(projection) == 5
    assert projection[0].startswith("t3 ")
    assert projection[-1].startswith("t7 ")


def test_outcome_projection_keeps_transaction_evidence_not_task_semantics() -> None:
    assert interaction_outcome({}, physical_outcome="completed") == "completed"
    assert interaction_outcome(
        {
            "status": "dispatched_unsettled",
            "error": "controller endpoint was not confirmed",
        }
    ) == (
        "dispatched_unsettled "
        "(error=controller endpoint was not confirmed)"
    )
    assert "grasp" not in interaction_outcome({}, physical_outcome="completed")


def test_function_registry_owns_effect_metadata_without_leaking_it_to_schema() -> None:
    registry = main_function_registry()

    assert registry.get("detect_region").world_effect == "none"
    assert registry.get("execute_action").world_effect == "physical"
    assert registry.get("execute_action").effect_channel == "arm"
    assert registry.get("close_gripper").effect_channel == "gripper"
    assert all("world_effect" not in definition for definition in registry.definitions)


def test_embodied_state_card_uses_live_robot_state_and_event_recency() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    memory = InteractionMemory()
    memory.record(
        _event(
            2,
            "action",
            "close_gripper",
            outcome="completed",
            revision_before=1,
            revision_after=2,
        ),
        effect_channel="gripper",
    )
    workspace.refresh_observation()

    card = project_embodied_state(
        workspace.state,
        memory,
        decision_turn=5,
    ).summary()

    assert card["tcp_pose"] == workspace.state.robot.tcp_pose.summary()
    assert "gripper_opening" not in card
    assert card["last_action"]["turns_ago"] == 3
    assert card["last_action"]["revisions_ago"] == max(
        0, workspace.state.observation_revision - 2
    )
    assert card["last_gripper_action"]["function"] == "close_gripper"
    assert "attachment" not in card


def test_workspace_no_longer_owns_agent_interaction_memory() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")

    assert not hasattr(workspace.state, "task_memory")
    assert workspace.execute("open_gripper").ok
    assert workspace.execute("detect_region", query="can").ok
    assert not hasattr(workspace.state, "task_memory")


def test_direct_controls_preserve_the_grasp_subject_across_revisions() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]
    assert workspace.execute("execute_action", action_id=action).ok
    assert workspace.execute("close_gripper").ok
    assert workspace.state.last_physical_action.source_query == "can"
    assert workspace._private.attachment_hypothesis.query == "can"

    assert workspace.execute(
        "move_tcp_delta",
        delta_xyz_m=[0.0, 0.0, 0.01],
        frame="base",
    ).ok
    assert workspace.state.last_physical_action.source_query == "can"


def test_pre_dispatch_planner_failure_does_not_create_a_physical_receipt() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    revision = workspace.state.observation_revision

    result = workspace.execute(
        "move_tcp_delta",
        delta_xyz_m=[0.0, 0.0, 0.01],
        frame="base",
    )

    assert not result.ok
    assert workspace.state.observation_revision == revision
    assert workspace.state.last_physical_action is None
