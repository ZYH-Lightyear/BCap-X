from __future__ import annotations

import numpy as np
import pytest

from vaw.context_runtime.gripper_mesh import (
    PandaUrdfGripperFK,
    overlay_projected_mesh_outline,
    semantic_parallel_jaw_triangles,
)


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


def test_joint_fk_exposes_palm_only_geometry_for_collision_occupancy(
    panda_fk: PandaUrdfGripperFK,
) -> None:
    joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])

    palm = panda_fk.hand_triangles(joints, gripper_opening=1.0)
    gripper = panda_fk.triangles(joints, gripper_opening=1.0)

    assert palm.shape[1:] == (3, 3)
    assert 0 < len(palm) < len(gripper)
    assert np.isfinite(palm).all()


def test_projected_mesh_outline_preserves_rgb_inside_silhouette() -> None:
    background = np.array([31, 47, 63], dtype=np.uint8)
    image = np.full((48, 48, 3), background, dtype=np.uint8)
    triangles = np.array(
        [
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]],
            [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    projected = np.array(
        [
            [[10.0, 10.0], [38.0, 10.0], [38.0, 38.0]],
            [[10.0, 10.0], [38.0, 38.0], [10.0, 38.0]],
        ],
        dtype=np.float64,
    )

    overlay_projected_mesh_outline(
        image,
        triangles,
        projected,
        np.ones((2, 3), dtype=np.float64),
        camera_position_base=np.zeros(3, dtype=np.float64),
    )

    line_color = np.array([216, 203, 255], dtype=np.uint8)
    assert np.any(np.all(image == line_color, axis=-1))
    assert np.array_equal(image[24, 24], background)


def test_semantic_parallel_jaw_is_metric_symmetric_wireframe_geometry() -> None:
    closed = semantic_parallel_jaw_triangles(
        (0.1, -0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
        0.0,
    )
    opened = semantic_parallel_jaw_triangles(
        (0.1, -0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
        1.0,
    )

    assert closed.shape == opened.shape == (36, 3, 3)
    assert np.isfinite(opened).all()
    closed_y = closed[..., 1] + 0.2
    opened_y = opened[..., 1] + 0.2
    assert np.isclose(closed_y.min(), -closed_y.max())
    assert np.isclose(opened_y.min(), -opened_y.max())
    assert np.ptp(opened_y) > np.ptp(closed_y)
