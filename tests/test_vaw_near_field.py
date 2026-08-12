from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import vaw.context_runtime.near_field as near_field_module
from vaw.context_runtime.contact_camera import ContactCameraPair
from vaw.context_runtime.model import Pose, RobotState
from vaw.context_runtime.near_field import (
    CONTACT_FOCUS_HEIGHT,
    CONTACT_FOCUS_WIDTH,
    NearFieldPreview,
    render_contact_focus,
    render_near_field,
)
from vaw.context_runtime.private import VisualEdit
from vaw.context_runtime.rgbd_surface import reconstruct_rgbd_surface


def _camera(
    color: tuple[int, int, int],
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
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
    if rotation is not None:
        pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
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
    camera_rotation = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    agentview = _camera((220, 30, 30), rotation=camera_rotation)
    wrist = _camera(
        (30, 210, 60),
        translation=(0.0, 0.035, 0.0),
        rotation=camera_rotation,
    )

    first = render_near_field(agentview, wrist, _robot())
    second = render_near_field(agentview, wrist, _robot())

    assert first is not None
    assert first.shape == (720, 832, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.any(np.all(first == np.array([220, 30, 30]), axis=-1))
    assert np.any(np.all(first == np.array([30, 210, 60]), axis=-1))


def test_contact_focus_emphasizes_only_the_observed_source_surface() -> None:
    camera = _camera((100, 120, 140))
    mask = np.zeros((16, 20), dtype=bool)
    mask[:, :10] = True

    sampled = reconstruct_rgbd_surface(
        camera,
        frame_position_base=np.zeros(3),
        frame_rotation_base=np.eye(3),
        half_extent_m=(1.0, 1.0, 1.0),
        emphasis_mask=mask,
    )

    assert sampled is not None
    unique = {tuple(color) for color in sampled.colors_rgb}
    assert (59, 142, 185) in unique  # source blended toward cyan
    assert (142, 152, 162) in unique  # background retained but de-emphasized


def test_near_field_places_contact_side_below_gripper(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    camera_rotation = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    raster = render_near_field(
        _camera(
            (220, 30, 30),
            translation=(0.0, 0.0, 0.06),
            rotation=camera_rotation,
        ),
        None,
        _robot(),
    )

    assert raster is not None
    # The synthetic surface lies at WORLD +Z from the TCP.  Gravity-stable
    # Contact views keep WORLD +Z visually upward.
    front = raster[: near_field_module.CONTACT_PANEL_HEIGHT]
    red_rows = np.nonzero(np.all(front == np.array([220, 30, 30]), axis=-1))[0]
    assert red_rows.size > 0
    assert float(np.median(red_rows)) < front.shape[0] / 2


def test_near_field_overlays_current_gripper_as_white_outline(monkeypatch) -> None:
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
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    baseline = render_near_field(_camera((180, 180, 180)), None, _robot())

    assert raster is not None and baseline is not None
    white_outline = np.all(raster == np.array([255, 255, 255]), axis=-1)
    baseline_white = np.all(baseline == np.array([255, 255, 255]), axis=-1)
    assert np.count_nonzero(white_outline) > np.count_nonzero(baseline_white) + 100


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
    violet = (raster[:, :, 0] > 90) & (raster[:, :, 2] > 170)
    assert np.count_nonzero(violet) > 100

    focus = render_contact_focus(
        _camera((180, 180, 180)), None, _robot(), preview
    )
    assert focus is not None
    assert focus.shape == (CONTACT_FOCUS_HEIGHT, CONTACT_FOCUS_WIDTH, 3)
    assert np.array_equal(focus, raster)


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


def test_near_field_exposes_previous_preview_and_on_demand_rotation_gizmo(
    monkeypatch,
) -> None:
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
    previous = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    target = Pose(
        (0.02, 0.0, 0.0),
        tuple(Rotation.from_euler("y", 12.0, degrees=True).as_quat()),
    )
    preview = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.5,
        previous_target_pose=previous,
        previous_gripper_opening=0.5,
        rotation_gizmo_frame="tool",
        rotation_gizmo_axis="y",
    )
    without_memory = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.5,
    )

    first = render_near_field(_camera((180, 180, 180)), None, _robot(), preview)
    second = render_near_field(_camera((180, 180, 180)), None, _robot(), preview)
    baseline = render_near_field(
        _camera((180, 180, 180)), None, _robot(), without_memory
    )

    assert first is not None and baseline is not None
    assert np.array_equal(first, second)
    assert not np.array_equal(first, baseline)


def test_direct_contact_camera_keeps_dense_rgb_and_moves_rotation_guide_off_object(
    monkeypatch,
) -> None:
    triangle = np.array(
        [[[0.0, -0.03, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.06]]],
        dtype=np.float64,
    )

    class FakeFK:
        def triangles(self, joints, opening):
            del joints, opening
            return triangle.copy()

    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: FakeFK())
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["images"]["depth"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH),
        0.3,
        dtype=np.float64,
    )
    camera["intrinsics"] = np.array(
        [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera["pose_mat"] = np.eye(4, dtype=np.float64)
    preview = NearFieldPreview(
        target_pose=Pose((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=0.5,
        rotation_gizmo_frame="tool",
        rotation_gizmo_axis="y",
    )

    raster = render_contact_focus(
        camera,
        None,
        _robot(),
        preview,
        contact_cameras=ContactCameraPair(front=camera, side=camera),
    )

    assert raster is not None
    assert raster.shape == (CONTACT_FOCUS_HEIGHT, CONTACT_FOCUS_WIDTH, 3)
    scene_width = CONTACT_FOCUS_WIDTH - near_field_module._ROTATION_GUIDE_WIDTH
    # The direct scene remains a dense resized MuJoCo raster rather than a
    # point-splat background with holes.
    dense_color = np.all(
        raster[: near_field_module.CONTACT_PANEL_HEIGHT, :scene_width]
        == np.array([116, 137, 158]),
        axis=-1,
    )
    assert np.count_nonzero(dense_color) > dense_color.size * 0.65
    # The single-axis +/- guide occupies its own rail and cannot cover the
    # contact evidence in the left scene area.
    rail = raster[
        : near_field_module.CONTACT_PANEL_HEIGHT,
        scene_width:,
    ]
    assert np.any(rail[:, :, 0] > rail[:, :, 1])  # red -10° card
    assert np.any(rail[:, :, 1] > rail[:, :, 0])  # green +10° card


def test_direct_contact_camera_always_shows_calibrated_move_and_rotate_axes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["images"]["depth"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH),
        0.3,
        dtype=np.float64,
    )
    preview = NearFieldPreview(
        target_pose=None,
        joint_positions_rad=None,
        gripper_opening=0.5,
    )

    raster = render_contact_focus(
        camera,
        None,
        _robot(),
        preview,
        contact_cameras=ContactCameraPair(front=camera, side=camera),
    )

    assert raster is not None
    front = raster[: near_field_module.CONTACT_PANEL_HEIGHT]
    rotate_control = front[
        8 : 8 + near_field_module._ROTATION_CORNER_HEIGHT,
        8 : 8 + near_field_module._ROTATION_CORNER_WIDTH,
    ]
    move_control = front[
        8 : 8 + near_field_module._TRANSLATION_CORNER_HEIGHT,
        -8 - near_field_module._TRANSLATION_CORNER_WIDTH : -8,
    ]
    for color in near_field_module._AXIS_COLORS:
        rgb = np.asarray(color[:3], dtype=np.uint8)
        assert np.count_nonzero(np.all(rotate_control == rgb, axis=-1)) > 5
        assert np.count_nonzero(np.all(move_control == rgb, axis=-1)) > 5


def test_direct_translation_guide_separates_view_normal_axis_from_screen_axes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    front = _camera((116, 137, 158))
    side = _camera((116, 137, 158))
    front["pose_mat"][:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    side["pose_mat"][:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    preview = NearFieldPreview(
        target_pose=None,
        joint_positions_rad=None,
        gripper_opening=0.5,
    )

    raster = render_contact_focus(
        front,
        None,
        _robot(),
        preview,
        contact_cameras=ContactCameraPair(front=front, side=side),
    )

    assert raster is not None
    panel_height = near_field_module.CONTACT_PANEL_HEIGHT
    panels = (
        raster[:panel_height],
        raster[panel_height + near_field_module.CONTACT_PANEL_GAP :],
    )
    for panel, depth_axis_index in zip(panels, (0, 1), strict=True):
        control = panel[
            8 : 8 + near_field_module._TRANSLATION_CORNER_HEIGHT,
            -8 - near_field_module._TRANSLATION_CORNER_WIDTH : -8,
        ]
        screen_axes = control[:, :164]
        depth_card = control[:, 164:]
        depth_color = np.asarray(
            near_field_module._AXIS_COLORS[depth_axis_index][:3],
            dtype=np.uint8,
        )
        assert not np.any(np.all(screen_axes == depth_color, axis=-1))
        assert np.count_nonzero(np.all(depth_card == depth_color, axis=-1)) > 20


def test_base_axis_screen_directions_follow_camera_calibration() -> None:
    camera = _camera((0, 0, 0))
    camera["pose_mat"][:3, :3] = Rotation.from_euler(
        "z", 90.0, degrees=True
    ).as_matrix()

    directions = near_field_module._base_axis_screen_directions(camera)

    # With BASE-from-camera = Rz(+90°), BASE +X appears toward image -Y
    # and BASE +Y appears toward image +X.  BASE +Z is view-normal here.
    assert np.allclose(directions[0], [0.0, -1.0], atol=1e-8)
    assert np.allclose(directions[1], [1.0, 0.0], atol=1e-8)
    assert np.allclose(directions[2], [0.0, 0.0], atol=1e-8)


def test_contact_views_keep_observed_world_fixed_across_target_rotation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    locked_frame = near_field_module.gravity_stable_contact_frame_quaternion(
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    )
    identity = NearFieldPreview(
        target_pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=None,
        gripper_opening=0.5,
        contact_frame_quaternion_xyzw=locked_frame,
    )
    rotated = NearFieldPreview(
        target_pose=Pose(
            (0.0, 0.0, 0.0),
            tuple(Rotation.from_euler("xyz", [35.0, 50.0, 70.0], degrees=True).as_quat()),
        ),
        joint_positions_rad=None,
        gripper_opening=0.5,
        contact_frame_quaternion_xyzw=locked_frame,
    )

    front_camera_rotation = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    side_camera_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    agentview = _camera((180, 80, 80), rotation=front_camera_rotation)
    wrist = _camera((80, 180, 80), rotation=side_camera_rotation)
    identity_raster = render_near_field(agentview, wrist, _robot(), identity)
    rotated_raster = render_near_field(agentview, wrist, _robot(), rotated)

    assert identity_raster is not None and rotated_raster is not None
    # With no robot mesh in this fixture, target orientation is the only state
    # difference.  A gravity-stable Contact Camera must therefore produce the
    # exact same current RGB-D projection instead of counter-rotating it.
    assert np.array_equal(identity_raster, rotated_raster)


def test_gravity_stable_contact_frame_keeps_world_z_up() -> None:
    pose = Pose(
        (0.0, 0.0, 0.0),
        tuple(Rotation.from_euler("xyz", [70.0, -35.0, 48.0], degrees=True).as_quat()),
    )

    quaternion = near_field_module.gravity_stable_contact_frame_quaternion(pose)
    matrix = Rotation.from_quat(quaternion).as_matrix()

    assert np.allclose(matrix[:, 2], [0.0, 0.0, 1.0])
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(matrix), 1.0)


def test_contact_views_expose_complementary_tool_axes() -> None:
    front, side = near_field_module._contact_views("TEST", np.zeros(3))
    assert (front.horizontal_axis, front.vertical_axis, front.view_axis) == (
        "y",
        "z",
        "x",
    )
    assert (side.horizontal_axis, side.vertical_axis, side.view_axis) == (
        "x",
        "z",
        "y",
    )
    assert np.allclose(front.up, [0.0, 0.0, 1.0])
    assert np.allclose(side.up, [0.0, 0.0, 1.0])
    points = np.array(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
        dtype=np.float64,
    )
    front_u, front_v, _ = near_field_module._project(
        points, front, near_field_module.NEAR_FIELD_WIDTH, 268
    )
    side_u, side_v, _ = near_field_module._project(
        points, side, near_field_module.NEAR_FIELD_WIDTH, 268
    )

    # Tool X is collapsed by FRONT but visible in SIDE.
    assert np.allclose([front_u[0], front_v[0]], [front_u[1], front_v[1]])
    assert not np.allclose([side_u[0], side_v[0]], [side_u[1], side_v[1]])
    # Tool Y is visible in FRONT but collapsed by SIDE.
    assert not np.allclose([front_u[0], front_v[0]], [front_u[2], front_v[2]])
    assert np.allclose([side_u[0], side_v[0]], [side_u[2], side_v[2]])


def test_gripper_only_preview_needs_no_visual_edit(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    preview = NearFieldPreview(
        target_pose=Pose((0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=0.0,
    )

    raster = render_contact_focus(
        _camera((180, 180, 180)),
        None,
        _robot(),
        preview,
    )

    assert raster is not None
    assert raster.shape == (CONTACT_FOCUS_HEIGHT, CONTACT_FOCUS_WIDTH, 3)


def test_near_field_requires_current_tcp_proprioception() -> None:
    robot = RobotState(
        ee_pose=None,
        tcp_pose=None,
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=1.0,
        source_revision=1,
    )
    assert render_near_field(_camera((10, 20, 30)), None, robot) is None
