from __future__ import annotations

from tests.test_vaw_context_runtime import FakeContextApi, FailedMotionBackend
from vaw.context_runtime.memory import (
    PhysicalPrimitive,
    TaskMemory,
    project_function_event,
)
from vaw.context_runtime.workspace import ContextWorkspace


def test_task_memory_compacts_only_equivalent_physical_primitives() -> None:
    memory = TaskMemory(capacity=6)
    memory.record(
        PhysicalPrimitive(
            op="delta_move",
            status="executed",
            frame="base",
            delta_xyz_m=(0.0, 0.0, 0.01),
        )
    )
    memory.record(
        PhysicalPrimitive(
            op="delta_move",
            status="executed",
            frame="base",
            delta_xyz_m=(0.0, 0.01, 0.02),
        )
    )
    memory.record(PhysicalPrimitive(op="close_gripper", status="executed"))
    memory.record(PhysicalPrimitive(op="close_gripper", status="executed"))
    memory.record(PhysicalPrimitive(op="close_gripper", status="effect_unknown"))

    assert memory.summary() == [
        {
            "op": "delta_move",
            "status": "executed",
            "frame": "base",
            "delta_xyz_m": [0.0, 0.01, 0.03],
        },
        {"op": "close_gripper", "status": "executed"},
        {"op": "close_gripper", "status": "effect_unknown"},
    ]


def test_successful_retry_replaces_uncertain_move_with_the_same_intent() -> None:
    memory = TaskMemory()
    memory.record(
        PhysicalPrimitive(
            op="move_to",
            status="effect_unknown",
            intent="approach can for grasp",
        )
    )
    memory.record(
        PhysicalPrimitive(
            op="move_to",
            status="executed",
            intent="approach can for grasp",
        )
    )

    assert memory.summary() == [
        {
            "op": "move_to",
            "status": "executed",
            "intent": "approach can for grasp",
        }
    ]


def test_consecutive_absolute_moves_keep_only_the_latest_arm_state() -> None:
    memory = TaskMemory()
    memory.record(
        PhysicalPrimitive(
            op="move_to",
            status="executed",
            intent="approach can for grasp",
        )
    )
    memory.record(
        PhysicalPrimitive(
            op="move_to",
            status="executed",
            intent="move to can top center",
        )
    )

    assert memory.summary() == [
        {
            "op": "move_to",
            "status": "executed",
            "intent": "move to can top center",
        }
    ]


def test_perception_never_enters_or_erases_task_memory() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    assert workspace.execute("open_gripper").ok
    before = workspace.state.task_memory.summary()

    assert workspace.execute("detection_and_sam", query="can").ok
    assert workspace.execute("locate_point", query="can center").ok

    assert workspace.state.task_memory.summary() == before == [
        {"op": "open_gripper", "status": "executed"}
    ]


def test_direct_controls_preserve_the_grasp_subject_across_revisions() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action = workspace.execute("select", seed_id=seed).result["action_id"]
    assert workspace.execute("commit", action_id=action).ok
    assert workspace.execute("close_gripper").ok
    assert workspace.state.last_physical_action.source_query == "can"
    assert workspace._private.attachment_hypothesis.query == "can"

    assert workspace.execute(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, 0.01],
        frame="base",
    ).ok
    assert workspace.state.last_physical_action.source_query == "can"


def test_pre_dispatch_planner_failure_is_not_a_physical_memory_fact() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    revision = workspace.state.observation_revision

    result = workspace.execute(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, 0.01],
        frame="base",
    )

    assert not result.ok
    assert workspace.state.observation_revision == revision
    assert workspace.state.last_physical_action is None
    assert workspace.state.task_memory.summary() == []


def test_dispatched_physical_failure_is_recorded_as_effect_unknown() -> None:
    api = FakeContextApi()
    api.gripper_error = True
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("close_gripper")

    assert not result.ok
    assert workspace.state.task_memory.summary() == [
        {"op": "close_gripper", "status": "effect_unknown"}
    ]


def test_select_event_surfaces_measured_plan_detail() -> None:
    event = project_function_event(
        "select",
        {
            "action_id": "a2",
            "solve_ik": "error",
            "detail": "CuRobo found no collision-free trajectory",
            "plan_reused": True,
        },
    )

    assert event.summary() == {
        "function": "select",
        "status": "ok",
        "references": {"action_id": "a2"},
        "message": (
            "returned this seed's already-measured plan; the trajectory result "
            "is unchanged; CuRobo found no collision-free trajectory"
        ),
    }


def test_current_event_projects_references_and_hides_backend_telemetry() -> None:
    success = project_function_event(
        "select",
        {
            "action_id": "a2",
            "solve_ik": "returned",
            "trajectory_checked": True,
        },
    )
    failure = project_function_event(
        "commit",
        {
            "error": "CuRobo INVALID_START_STATE_JOINT_LIMITS",
            "position_error_m": 0.302,
        },
    )

    assert success.summary() == {
        "function": "select",
        "status": "ok",
        "references": {"action_id": "a2"},
    }
    assert failure.summary() == {
        "function": "commit",
        "status": "failed",
        "message": "the spatial command did not complete; its effect is uncertain",
    }


def test_pre_dispatch_physical_failure_does_not_claim_the_world_changed() -> None:
    event = project_function_event(
        "commit",
        {"error": "active action 'a1' has no executable cached plan"},
        world_changed=False,
    )

    assert event.summary() == {
        "function": "commit",
        "status": "failed",
        "message": (
            "rejected before dispatch; the real world is unchanged: "
            "active action 'a1' has no executable cached plan"
        ),
    }


def test_imagination_failure_event_carries_the_policy_reason() -> None:
    event = project_function_event(
        "call_imagination",
        {"status": "failed", "action_id": "a3", "reason": "geometry_unresolved"},
        world_changed=False,
    )

    assert event.summary() == {
        "function": "call_imagination",
        "status": "failed",
        "references": {"action_id": "a3", "reason": "geometry_unresolved"},
        "message": (
            "local imagination did not deliver a refined action; "
            "reason=geometry_unresolved"
        ),
    }


def test_imagination_partial_event_surfaces_the_review_advisory() -> None:
    event = project_function_event(
        "call_imagination",
        {
            "status": "partial",
            "action_id": "a3",
            "reason": "turn_limit",
            "advisory": "review the Preview, then commit, re-delegate or reject",
        },
        world_changed=False,
    )

    assert event.summary() == {
        "function": "call_imagination",
        "status": "partial",
        "references": {"action_id": "a3", "reason": "turn_limit"},
        "message": "review the Preview, then commit, re-delegate or reject",
    }
