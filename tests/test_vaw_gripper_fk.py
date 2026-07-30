from __future__ import annotations

import numpy as np
import pytest

from vaw.gripper_mesh import PandaUrdfGripperFK
from vaw.preview import run_preview
from vaw.scripts.smoke_render import synthetic_obs
from vaw.state import ActionState
from vaw.types import Candidate, Pose, TOP_DOWN_QUAT_WXYZ
from vaw.workspace import Workspace


@pytest.fixture(scope="module")
def panda_fk() -> PandaUrdfGripperFK:
    pytest.importorskip("yourdfpy")
    pytest.importorskip("robot_descriptions")
    return PandaUrdfGripperFK()


def test_joint_fk_returns_actual_urdf_gripper_triangles(
    panda_fk: PandaUrdfGripperFK,
) -> None:
    joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

    triangles = panda_fk.triangles(joints, gripper_opening=1.0)
    hand = panda_fk.frame(joints, "panda_hand", gripper_opening=1.0)

    assert triangles.ndim == 3
    assert triangles.shape[1:] == (3, 3)
    assert len(triangles) > 100
    assert np.isfinite(triangles).all()
    # The mesh is transformed into robot/world coordinates by URDF FK, not left
    # in panda_hand-local coordinates near the origin.
    assert np.linalg.norm(np.median(triangles.reshape(-1, 3), axis=0) - hand[:3, 3]) < 0.2


def test_joint_fk_uses_observed_finger_opening(
    panda_fk: PandaUrdfGripperFK,
) -> None:
    joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

    closed = panda_fk.triangles(joints, gripper_opening=0.0)
    opened = panda_fk.triangles(joints, gripper_opening=1.0)

    assert closed.shape == opened.shape
    assert not np.allclose(closed, opened)


def test_workspace_keeps_authoritative_arm_joints() -> None:
    obs, _ = synthetic_obs()
    expected = np.linspace(-0.6, 0.6, 7)
    obs["robot_joint_pos"] = np.concatenate([expected, [0.75]])

    class Api:
        camera_name = "agentview"
        wrist_camera_name = "robot0_eye_in_hand"

        def get_observation(self):
            return obs

    workspace = Workspace(Api(), instruction="test")
    workspace.refresh_observation()

    np.testing.assert_allclose(workspace.state.arm_joint_positions_rad, expected)
    assert workspace.state.arm_joint_positions_rad.shape == (7,)


def test_preview_preserves_the_exact_ik_solution_for_rendering() -> None:
    obs, _ = synthetic_obs()
    expected = np.array([0.1, -0.2, 0.3, -1.7, 0.4, 1.2, 0.8])

    class Api:
        def solve_ik(self, position, quat_wxyz, *, return_info=False):
            assert return_info
            return expected.copy(), {"orientation_used": "requested"}

    state = ActionState(instruction="test")
    candidate = Candidate(
        candidate_id="p1",
        kind="waypoint",
        pose=Pose(np.array([0.55, 0.0, 0.25]), TOP_DOWN_QUAT_WXYZ.copy()),
    )

    preview = run_preview(Api(), state, candidate)

    np.testing.assert_allclose(preview.joint_positions_rad, expected)
    assert preview.ik_ok
    assert preview.notes == "endpoint IK solved; trajectory not planned"
    assert preview.summary()["trajectory_planned"] is False
    assert "path_world" not in vars(preview)
    assert "collision" not in preview.summary()
