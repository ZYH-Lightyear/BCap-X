from __future__ import annotations

import numpy as np
import pytest

from vaw.context_runtime.gripper_mesh import PandaUrdfGripperFK


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


def test_joint_fk_can_render_the_complete_robot_visual_mesh(
    panda_fk: PandaUrdfGripperFK,
) -> None:
    joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

    gripper = panda_fk.triangles(joints, gripper_opening=1.0)
    robot = panda_fk.robot_triangles(joints, gripper_opening=1.0)

    assert robot.shape[1:] == (3, 3)
    assert len(robot) > len(gripper)
    assert np.isfinite(robot).all()
