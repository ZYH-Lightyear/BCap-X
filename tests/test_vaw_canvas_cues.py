from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from vaw.context_runtime import near_field, plumb_line, scene_view
from vaw.context_runtime.base_frame_overlay import (
    BASE_FRAME_AXIS_COLORS,
    draw_agentview_base_frame,
)
from vaw.context_runtime.model import Pose


def _oblique_camera(width: int, height: int) -> dict[str, np.ndarray]:
    forward = np.array([0.691, 0.691, -0.207], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    origin = np.array([0.0, 0.0, 0.08], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = origin - forward
    focal = min(width, height) * 0.8
    return {
        "intrinsics": np.array(
            [
                [focal, 0.0, width / 2],
                [0.0, focal, height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "pose_mat": pose,
    }


def test_agentview_base_frame_uses_calibrated_xyz_projection() -> None:
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    drawn = draw_agentview_base_frame(image, _oblique_camera(480, 320))

    assert drawn is True
    for color in BASE_FRAME_AXIS_COLORS.values():
        assert np.count_nonzero(np.all(image == color, axis=2)) > 10


def test_auxiliary_compass_draws_xy_without_z() -> None:
    image = Image.new("RGB", (520, 320), color=(220, 220, 220))
    draw = ImageDraw.Draw(image, "RGBA")

    near_field._draw_direct_base_xy_axes(
        draw,
        _oblique_camera(520, 320),
        520,
    )

    rendered = np.asarray(image)
    assert (
        np.count_nonzero(np.all(rendered == near_field._AXIS_COLORS[0][:3], axis=2))
        > 10
    )
    assert (
        np.count_nonzero(np.all(rendered == near_field._AXIS_COLORS[1][:3], axis=2))
        > 10
    )
    assert (
        np.count_nonzero(np.all(rendered == near_field._AXIS_COLORS[2][:3], axis=2))
        == 0
    )


def test_contact_frame_uses_stable_tool_axis_for_horizontal_handle_pose() -> None:
    # 来自 drawer trace 的真实水平抓取姿态。此时 tool-Y 几乎竖直，旧实现
    # 会归一化其 0.006 的水平残量并把 IK 噪声放大成相机偏航。
    pose = Pose(
        (0.0, 0.0, 0.0),
        (0.707984, 0.000917, -0.003301, 0.706220),
    )
    tool_rotation = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()

    contact_quaternion = near_field.gravity_stable_contact_frame_quaternion(pose)
    contact_rotation = Rotation.from_quat(contact_quaternion).as_matrix()
    projected_tool_x = tool_rotation[:, 0].copy()
    projected_tool_x[2] = 0.0
    projected_tool_x /= np.linalg.norm(projected_tool_x)

    assert np.linalg.norm(tool_rotation[:2, 1]) < 0.01
    assert np.dot(contact_rotation[:, 0], projected_tool_x) > 0.999999
    assert np.allclose(contact_rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-8)


def test_contact_frame_reference_prevents_half_turn_sign_flip() -> None:
    pose = Pose(
        (0.0, 0.0, 0.0),
        tuple(Rotation.from_euler("z", 179.0, degrees=True).as_quat()),
    )
    reference = tuple(Rotation.from_euler("z", 1.0, degrees=True).as_quat())

    quaternion = near_field.gravity_stable_contact_frame_quaternion(
        pose,
        reference_quaternion_xyzw=reference,
    )

    rotation = Rotation.from_quat(quaternion).as_matrix()
    reference_x = Rotation.from_quat(reference).as_matrix()[:, 0]
    assert np.dot(rotation[:, 0], reference_x) > 0.0


def test_contact_frame_keeps_tool_y_semantics_when_axes_are_equally_stable() -> None:
    pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    quaternion = near_field.gravity_stable_contact_frame_quaternion(pose)
    rotation = Rotation.from_quat(quaternion).as_matrix()

    # Identity pose 的 tool-Y 仍是 Contact +Y；这保持了常见竖直抓取已有
    # 的 FRONT/SIDE 语义，而不是无缘无故旋转 90°。
    assert np.allclose(rotation, np.eye(3), atol=1e-8)


def test_gripper_occupancy_is_five_percentage_points_more_opaque() -> None:
    assert near_field._CURRENT_FINGER_ALPHA == pytest.approx(0.31)
    assert near_field._PREVIEW_OCCUPANCY_ALPHA == pytest.approx(0.25)
    assert scene_view._PREVIEW_OCCUPANCY_ALPHA == pytest.approx(0.23)


def test_plumb_overlay_keeps_direction_geometry_without_numeric_labels() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    camera = {
        "intrinsics": np.array(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "pose_mat": np.eye(4, dtype=np.float64),
    }
    cue = plumb_line.PlumbLine(
        kind="current",
        anchor_base_xyz=(0.0, 0.0, 2.0),
        landing_base_xyz=(0.0, 0.0, 1.0),
        height_m=1.0,
        footprint_base=None,
        target_center_base_xyz=(0.2, 0.0, 1.0),
        offset_xy_m=(0.2, 0.0),
    )

    plumb_line.draw_plumb_overlays(image, camera, [cue])

    assert np.any(np.all(image == plumb_line.OFFSET_ARROW_RGB, axis=2))
    assert not hasattr(plumb_line, "_pill_label")


def test_grasp_sweep_line_projects_the_observed_finger_channel() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    camera = {
        "intrinsics": np.array(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "pose_mat": np.eye(4, dtype=np.float64),
    }

    drawn = near_field.draw_grasp_sweep_overlay(
        image,
        camera,
        np.array([[-0.3, 0.0, 1.0], [0.3, 0.0, 1.0]], dtype=np.float64),
    )

    assert drawn is True
    assert np.any(np.all(image == near_field.GRASP_SWEEP_RGB, axis=2))


def test_closed_gripper_z_axis_draws_tcp_ball_and_base_z_line() -> None:
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    camera = {
        "intrinsics": np.array(
            [[120.0, 0.0, 90.0], [0.0, 120.0, 80.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        # camera X=BASE-X, camera Y=-BASE-Z, camera Z=BASE-Y，因此
        # BASE-Z 在图像中是一条竖直线。
        "pose_mat": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    }
    # [下端, TCP, 上端]；在该测试相机中 BASE-Z 投影为穿过 TCP 的竖线。
    axis = np.array(
        [[0.0, 1.0, -0.25], [0.0, 1.0, 0.0], [0.0, 1.0, 0.08]],
        dtype=np.float64,
    )

    drawn = near_field.draw_gripper_z_axis_overlay(image, camera, axis)

    assert drawn is True
    cue = np.all(image == near_field.GRIPPER_Z_AXIS_RGB, axis=2)
    assert np.count_nonzero(cue) > 20
    # TCP 小球比三像素轴线更宽。
    assert np.count_nonzero(cue[80]) >= 10
