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


def test_perception_never_enters_or_erases_task_memory() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    assert workspace.execute("open_gripper").ok
    before = workspace.state.task_memory.summary()

    assert workspace.execute("detection_and_sam", query="can").ok
    assert workspace.execute("locate_point", query="can center").ok

    assert workspace.state.task_memory.summary() == before == [
        {"op": "open_gripper", "status": "executed"}
    ]


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
