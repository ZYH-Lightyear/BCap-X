from __future__ import annotations

import numpy as np
import pytest

from vaw.context_runtime import near_field, plumb_line, scene_view


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
