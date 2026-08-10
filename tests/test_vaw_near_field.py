from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import vaw.context_runtime.near_field as near_field_module
from vaw.context_runtime.model import Pose, RobotState
from vaw.context_runtime.near_field import NearFieldPreview, render_near_field
from vaw.context_runtime.private import VisualEdit


def _camera(
    color: tuple[int, int, int],
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    height, width = 16, 20
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = np.asarray(color, dtype=np.uint8)
    depth = np.full((height, width), 0.06, dtype=np.float64)
    intrinsics = np.array(
        [[100.0, 0.0, 9.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return {
        "images": {"rgb": rgb, "depth": depth},
        "intrinsics": intrinsics,
        "pose_mat": pose,
    }


def _robot() -> RobotState:
    pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    return RobotState(
        ee_pose=pose,
        tcp_pose=pose,
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=0.5,
        source_revision=1,
    )


def test_near_field_fuses_both_current_rgbd_views_deterministically(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    agentview = _camera((220, 30, 30))
    wrist = _camera((30, 210, 60), translation=(0.0, 0.035, 0.0))

    first = render_near_field(agentview, wrist, _robot())
    second = render_near_field(agentview, wrist, _robot())

    assert first is not None
    assert first.shape == (720, 640, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.any(np.all(first == np.array([220, 30, 30]), axis=-1))
    assert np.any(np.all(first == np.array([30, 210, 60]), axis=-1))


def test_near_field_places_contact_side_below_gripper(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    raster = render_near_field(_camera((220, 30, 30)), None, _robot())

    assert raster is not None
    # The synthetic surface lies at local +Z from the TCP. In the jaw-plane
    # panel, +Z is the contact/support side and must appear below its centre.
    jaw = raster[near_field_module._PANEL_HEIGHT + near_field_module._PANEL_GAP :]
    red_rows = np.nonzero(np.all(jaw == np.array([220, 30, 30]), axis=-1))[0]
    assert red_rows.size > 0
    assert float(np.median(red_rows)) > jaw.shape[0] / 2


def test_near_field_overlays_current_gripper_as_high_salience_blue(monkeypatch) -> None:
    triangle = np.array(
        [[[-0.04, -0.04, 0.0], [0.04, -0.04, 0.0], [0.0, 0.05, 0.04]]],
        dtype=np.float64,
    )

    class FakeFK:
        def triangles(self, joints, opening):
            del joints, opening
            return triangle

    monkeypatch.setattr(
        near_field_module, "load_panda_urdf_fk", lambda: FakeFK()
    )
    raster = render_near_field(_camera((180, 180, 180)), None, _robot())

    assert raster is not None
    blue_fill = (raster[:, :, 2] > 190) & (raster[:, :, 0] < 120)
    assert np.count_nonzero(blue_fill) > 100


def test_near_field_overlays_preview_gripper_on_same_current_cloud(monkeypatch) -> None:
    triangle = np.array(
        [[[-0.04, -0.04, 0.0], [0.04, -0.04, 0.0], [0.0, 0.05, 0.04]]],
        dtype=np.float64,
    )

    class FakeFK:
        def triangles(self, joints, opening):
            del opening
            shifted = triangle.copy()
            shifted[:, :, 0] += float(np.asarray(joints)[0])
            return shifted

    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: FakeFK())
    reference = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    target = Pose((0.03, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    preview = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.0,
        visual_edit=VisualEdit(
            kind="delta_move",
            frame="base",
            reference_pose=reference,
            delta_xyz_m=(0.03, 0.0, 0.0),
        ),
    )

    raster = render_near_field(
        _camera((180, 180, 180)), None, _robot(), preview
    )

    assert raster is not None
    blue = (raster[:, :, 2] > 180) & (raster[:, :, 0] < 100)
    violet = (raster[:, :, 0] > 90) & (raster[:, :, 2] > 170)
    assert np.count_nonzero(blue) > 100
    assert np.count_nonzero(violet) > 100


def test_near_field_rotate_preview_changes_visual_cue_deterministically(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    reference = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    target = Pose(
        (0.0, 0.0, 0.0),
        tuple(Rotation.from_euler("y", 15.0, degrees=True).as_quat()),
    )
    preview = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=None,
        gripper_opening=0.5,
        visual_edit=VisualEdit(
            kind="rotate",
            frame="tool",
            reference_pose=reference,
            axis="y",
            angle_deg=15.0,
        ),
    )

    first = render_near_field(_camera((180, 180, 180)), None, _robot(), preview)
    second = render_near_field(_camera((180, 180, 180)), None, _robot(), preview)

    assert first is not None
    assert np.array_equal(first, second)
    assert not np.array_equal(
        first,
        render_near_field(_camera((180, 180, 180)), None, _robot()),
    )


def test_near_field_requires_current_tcp_proprioception() -> None:
    robot = RobotState(
        ee_pose=None,
        tcp_pose=None,
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=1.0,
        source_revision=1,
    )
    assert render_near_field(_camera((10, 20, 30)), None, robot) is None
