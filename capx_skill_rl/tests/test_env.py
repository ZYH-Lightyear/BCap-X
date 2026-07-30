from __future__ import annotations

import numpy as np
import pytest

from capx_skill_rl.env import Observation, ToolEnv
from capx_skill_rl.tests.fakes import FakeBackend


def action(name: str, **arguments):
    return {"name": name, "arguments": arguments}


def test_observation_is_rgb_and_task_only() -> None:
    observation = ToolEnv(FakeBackend()).reset(seed=7)
    assert isinstance(observation, Observation)
    assert observation.task == "pick up the object"
    assert observation.rgb.shape == (4, 4, 3)
    assert set(observation.__dataclass_fields__) == {"rgb", "task"}


def test_perception_chain_uses_hidden_frame_and_returns_small_values() -> None:
    backend = FakeBackend()
    env = ToolEnv(backend)
    env.reset()

    bbox = env.step(action("vlm_bbox_detection", query="object"))
    assert bbox.result == {"bbox": [0.0, 0.0, 3.0, 3.0]}
    assert bbox.observation is None

    mask = env.step(action("sam3", bbox=bbox.result["bbox"]))
    assert mask.result == {"mask_id": "m0"}

    obb = env.step(action("get_obb", mask_id="m0"))
    assert set(obb.result) == {"center", "extent", "quaternion"}
    assert backend.last_obb_points is not None
    assert np.all(backend.last_obb_points[:, 0] >= 1.0)
    assert np.all(backend.last_obb_points[:, 1] >= 2.0)
    assert np.allclose(backend.last_obb_points[:, 2], 4.0)

    grasp = env.step(action("plan_grasp", mask_id="m0"))
    assert grasp.result == {
        "position": [1.4, 2.5, 3.6],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }


def test_xyzw_is_converted_to_capx_wxyz_for_ik() -> None:
    backend = FakeBackend()
    env = ToolEnv(backend)
    env.reset()
    result = env.step(
        action(
            "solve_ik",
            position=[0.1, 0.2, 0.3],
            quaternion=[0.0, 0.0, 0.0, 2.0],
        )
    )
    assert result.result == {"joints": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    assert np.allclose(backend.last_ik_quaternion, [1.0, 0.0, 0.0, 0.0])


def test_physical_action_refreshes_rgb_and_invalidates_masks() -> None:
    backend = FakeBackend()
    env = ToolEnv(backend)
    env.reset()
    mask_id = env.step(action("sam3", text="object")).result["mask_id"]

    moved = env.step(action("move_to_joints", joints=[0] * 7))
    assert moved.result == {}
    assert moved.observation is not None
    assert moved.observation.rgb[0, 0, 0] == 1
    assert backend.capture_count == 1

    stale = env.step(action("get_obb", mask_id=mask_id))
    assert "unknown mask_id" in stale.result["error"]


def test_failed_physical_action_still_refreshes_observation() -> None:
    backend = FakeBackend()
    backend.fail_move = True
    env = ToolEnv(backend)
    env.reset()
    result = env.step(action("move_to_joints", joints=[0] * 7))
    assert result.result == {"error": "motion failed"}
    assert result.observation is not None
    assert backend.capture_count == 1


def test_sparse_success_reward_and_done_come_from_backend() -> None:
    env = ToolEnv(FakeBackend(success_on_move=True))
    env.reset()
    result = env.step(action("move_to_joints", joints=[0] * 7))
    assert result.reward == 1.0
    assert result.done is True


def test_invalid_actions_count_toward_horizon_without_execution() -> None:
    backend = FakeBackend()
    env = ToolEnv(backend, max_steps=2)
    env.reset()

    first = env.step([action("go_home")])
    assert first.result == {"error": "exactly one tool call is required per step"}
    assert first.done is False

    second = env.step({"name": "open_gripper", "arguments": {}, "id": "extra"})
    assert "unexpected" in second.result["error"]
    assert second.done is True
    assert backend.capture_count == 0


def test_step_rejects_calls_after_terminal() -> None:
    env = ToolEnv(FakeBackend(), max_steps=1)
    env.reset()
    assert env.step(action("vlm_point_detection", query="object")).done
    with pytest.raises(RuntimeError, match="already ended"):
        env.step(action("vlm_point_detection", query="object"))
