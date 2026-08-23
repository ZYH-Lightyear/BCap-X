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


def test_preview_gripper_overlays_on_the_current_contact_view(monkeypatch) -> None:
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

    raster = render_contact_focus(
        _camera((180, 180, 180)), None, _robot(), preview
    )

    assert raster is not None
    assert raster.shape == (CONTACT_FOCUS_HEIGHT, CONTACT_FOCUS_WIDTH, 3)
    violet = (raster[:, :, 0] > 90) & (raster[:, :, 2] > 170)
    assert np.count_nonzero(violet) > 100


def test_semantic_preview_uses_realized_tcp_and_has_no_palm_fill() -> None:
    preview = NearFieldPreview(
        target_pose=Pose((0.3, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=0.75,
        gripper_style="semantic-wireframe",
        realized_tcp_pose=Pose((0.02, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )

    triangles = near_field_module._preview_gripper_triangles(_robot(), preview)

    assert triangles is not None
    assert triangles.shape == (36, 3, 3)
    # Rendering follows the TCP realised by returned joints, not an unchecked
    # ideal target pose far away from it.
    assert abs(float(np.median(triangles[..., 0])) - 0.02) < 0.02
    assert near_field_module._preview_hand_triangles(_robot(), preview) is None


def test_semantic_preview_is_hidden_without_returned_joint_realisation() -> None:
    preview = NearFieldPreview(
        target_pose=Pose((0.02, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=None,
        gripper_opening=0.75,
        gripper_style="semantic-wireframe",
        realized_tcp_pose=None,
    )

    assert near_field_module._preview_gripper_triangles(_robot(), preview) is None


def test_preview_faintly_fills_only_the_rigid_hand_occupancy(monkeypatch) -> None:
    # Use a small 3-D palm proxy.  A single XY plane is edge-on in both
    # gravity-stable Contact views and therefore has no silhouette area.
    corners = np.array(
        [
            [-0.05, -0.04, -0.025],
            [0.05, -0.04, -0.025],
            [0.05, 0.04, -0.025],
            [-0.05, 0.04, -0.025],
            [-0.05, -0.04, 0.025],
            [0.05, -0.04, 0.025],
            [0.05, 0.04, 0.025],
            [-0.05, 0.04, 0.025],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    gripper = corners[faces]
    palm = gripper * np.array([0.55, 0.55, 0.70])[None, None, :]

    class LineOnlyFK:
        def triangles(self, joints, opening):
            del joints, opening
            return gripper

    class OccupancyFK(LineOnlyFK):
        def hand_triangles(self, joints, opening):
            del joints, opening
            return palm

    preview = NearFieldPreview(
        target_pose=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        joint_positions_rad=tuple(np.zeros(7)),
        gripper_opening=0.5,
    )
    monkeypatch.setattr(
        near_field_module,
        "load_panda_urdf_fk",
        lambda: LineOnlyFK(),
    )
    line_only = render_contact_focus(
        _camera((180, 180, 180)),
        None,
        _robot(),
        preview,
    )
    monkeypatch.setattr(
        near_field_module,
        "load_panda_urdf_fk",
        lambda: OccupancyFK(),
    )
    occupied = render_contact_focus(
        _camera((180, 180, 180)),
        None,
        _robot(),
        preview,
    )

    assert line_only is not None and occupied is not None
    changed = np.any(line_only != occupied, axis=-1)
    assert np.count_nonzero(changed) > 20
    # The majority of the grasp aperture remains unchanged: this is a palm
    # occupancy cue, not a return to an opaque full-gripper mask.
    assert np.count_nonzero(changed) < occupied.shape[0] * occupied.shape[1] * 0.1


def test_near_field_rotation_gizmo_does_not_draw_previous_preview(
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
    with_previous = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.5,
        previous_target_pose=previous,
        previous_gripper_opening=0.5,
        rotation_gizmo_frame="tool",
        rotation_gizmo_axis="y",
    )
    without_previous = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.5,
        rotation_gizmo_frame="tool",
        rotation_gizmo_axis="y",
    )
    without_gizmo = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=(0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gripper_opening=0.5,
    )

    first = render_contact_focus(
        _camera((180, 180, 180)), None, _robot(), with_previous
    )
    second = render_contact_focus(
        _camera((180, 180, 180)), None, _robot(), without_previous
    )
    baseline = render_contact_focus(
        _camera((180, 180, 180)), None, _robot(), without_gizmo
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


def test_direct_contact_camera_masks_current_action_source_surface(
    monkeypatch,
) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["intrinsics"] = np.array(
        [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera["pose_mat"] = np.eye(4, dtype=np.float64)
    source = np.array(
        [
            [-0.025, -0.025, 0.3],
            [0.025, -0.025, 0.3],
            [0.025, 0.025, 0.3],
            [-0.025, 0.025, 0.3],
            [0.0, 0.0, 0.3],
        ],
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
        source_points_base=source,
        contact_cameras=ContactCameraPair(front=camera, side=camera),
    )

    assert raster is not None
    edge = np.asarray(near_field_module._SOURCE_EDGE[:3], dtype=np.uint8)
    assert np.count_nonzero(np.all(raster == edge, axis=-1)) > 20
    assert not np.array_equal(
        raster[179, 416],
        np.array([116, 137, 158], dtype=np.uint8),
    )


def test_direct_contact_camera_uses_same_camera_finger_segmentation(monkeypatch) -> None:
    def reject_urdf_current_mask():
        raise AssertionError("current finger mask must not use separate URDF geometry")

    monkeypatch.setattr(
        near_field_module,
        "load_panda_urdf_fk",
        reject_urdf_current_mask,
    )
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["intrinsics"] = np.array(
        [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera["pose_mat"] = np.eye(4, dtype=np.float64)
    finger_mask = np.zeros(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH),
        dtype=bool,
    )
    finger_mask[120:210, 330:370] = True
    finger_mask[120:210, 462:502] = True
    camera["finger_mask"] = finger_mask
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
    edge = near_field_module._CURRENT_FINGER_EDGE
    assert np.count_nonzero(np.all(raster == edge, axis=-1)) > 20
    # Pixels outside MuJoCo's own segmentation remain untouched; no expanded
    # FK silhouette is invented around the real fingers.
    assert np.array_equal(
        raster[100, 350],
        np.array([116, 137, 158], dtype=np.uint8),
    )


def test_direct_contact_camera_paints_the_palm_floor_band(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["intrinsics"] = np.array(
        [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera["pose_mat"] = np.eye(4, dtype=np.float64)
    palm = np.zeros(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH),
        dtype=bool,
    )
    palm[200:206, 360:470] = True
    camera["palm_floor_mask"] = palm
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
    edge = near_field_module._PALM_FLOOR_EDGE
    assert np.count_nonzero(np.all(raster == edge, axis=-1)) > 10
    filled = raster[202, 400]
    assert filled[2] > filled[0]
    assert filled[1] > filled[0]


def test_direct_contact_camera_shows_move_axes_without_persistent_rotate_gimbal(
    monkeypatch,
) -> None:
    assert near_field_module._TRANSLATION_CORNER_WIDTH == 293
    assert near_field_module._TRANSLATION_CORNER_HEIGHT == 202
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
        assert not np.any(np.all(rotate_control == rgb, axis=-1))
    move_axis_counts = [
        np.count_nonzero(
            np.all(move_control == np.asarray(color[:3], dtype=np.uint8), axis=-1)
        )
        for color in near_field_module._AXIS_COLORS
    ]
    assert sum(count > 5 for count in move_axis_counts) == 2


def test_direct_translation_guide_omits_view_normal_axis(
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
        depth_color = np.asarray(
            near_field_module._AXIS_COLORS[depth_axis_index][:3],
            dtype=np.uint8,
        )
        assert not np.any(np.all(control == depth_color, axis=-1))
        for axis_index, color in enumerate(near_field_module._AXIS_COLORS):
            if axis_index == depth_axis_index:
                continue
            rgb = np.asarray(color[:3], dtype=np.uint8)
            assert np.count_nonzero(np.all(control == rgb, axis=-1)) > 5


def test_direct_contact_view_draws_single_five_centimetre_scale(monkeypatch) -> None:
    monkeypatch.setattr(near_field_module, "load_panda_urdf_fk", lambda: None)
    camera = _camera((116, 137, 158))
    camera["images"]["rgb"] = np.full(
        (near_field_module.CONTACT_PANEL_HEIGHT, CONTACT_FOCUS_WIDTH, 3),
        (116, 137, 158),
        dtype=np.uint8,
    )
    camera["intrinsics"] = np.array(
        [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = Pose((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
    preview = NearFieldPreview(
        target_pose=target,
        joint_positions_rad=None,
        gripper_opening=0.5,
        visual_edit=VisualEdit(
            kind="delta_move",
            frame="base",
            reference_pose=Pose((0.0, 0.0, 0.28), (0.0, 0.0, 0.0, 1.0)),
            delta_xyz_m=(0.0, 0.0, 0.02),
        ),
    )

    raster = render_contact_focus(
        camera,
        None,
        _robot(),
        preview,
        contact_cameras=ContactCameraPair(front=camera, side=camera),
    )

    assert raster is not None
    panel = raster[: near_field_module.CONTACT_PANEL_HEIGHT]
    scale_region = panel[-78:-22, -170:-8]
    accent = np.array([30, 64, 175], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    assert np.count_nonzero(np.all(scale_region == accent, axis=-1)) > 20
    assert np.count_nonzero(np.all(scale_region == white, axis=-1)) > 20


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
