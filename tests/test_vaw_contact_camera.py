from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vaw.context_runtime.contact_camera import (
    ContactCameraRequest,
    LiberoContactCameraProvider,
    LiberoOppositeSceneCameraProvider,
    OppositeSceneCameraRequest,
)
from vaw.context_runtime.geometry import project_world_to_pixel


class _FakeModel:
    def __init__(self) -> None:
        self.cam_pos = np.array(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64
        )
        self.cam_quat = np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.cam_fovy = np.array([45.0, 50.0], dtype=np.float64)
        self.stat = SimpleNamespace(extent=1.0)
        self.vis = SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10.0))

    @staticmethod
    def camera_name2id(name: str) -> int:
        return {"frontview": 0, "sideview": 1}[name]


class _FakeSim:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.model = _FakeModel()
        self.data = SimpleNamespace(
            xpos=np.array([[0.2, -0.1, 0.3]], dtype=np.float64),
            xquat=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        )
        self.fail_on = fail_on
        self.render_poses: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1

    def render(self, *, camera_name: str, width: int, height: int, depth: bool):
        if camera_name == self.fail_on:
            raise RuntimeError("synthetic render failure")
        camera_id = self.model.camera_name2id(camera_name)
        self.render_poses.append(
            (
                camera_name,
                self.model.cam_pos[camera_id].copy(),
                self.model.cam_quat[camera_id].copy(),
            )
        )
        color = 40 if camera_name == "frontview" else 180
        rgb = np.full((height, width, 3), color, dtype=np.uint8)
        if not depth:
            return rgb
        # A far plane makes every projected subject point visible.  The
        # provider must consume this privately and return RGB-only cameras.
        normalized_depth = np.full((height, width), 1.0, dtype=np.float64)
        return rgb, normalized_depth


class _SegmentationModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.geom_names = (
            "floor",
            "gripper0_finger1_visual",
            "gripper0_finger2_visual",
            "gripper0_finger1_collision",
            "gripper0_hand_visual",
        )

    def geom_name2id(self, name: str) -> int:
        return self.geom_names.index(name)


class _SegmentationSim(_FakeSim):
    def __init__(self) -> None:
        super().__init__()
        self.model = _SegmentationModel()

    def render(
        self,
        *,
        camera_name: str,
        width: int,
        height: int,
        depth: bool,
        segmentation: bool = False,
    ):
        if not segmentation:
            return super().render(
                camera_name=camera_name,
                width=width,
                height=height,
                depth=depth,
            )
        result = np.full((height, width, 2), -1, dtype=np.int32)
        # robosuite segmentation stores (MuJoCo object type, object id) and is
        # returned in OpenGL's bottom-up row order.
        result[4:12, 10:18, 0] = 5
        result[4:12, 10:18, 1] = 1
        result[4:12, 24:32, 0] = 5
        result[4:12, 24:32, 1] = 2
        # A collision geom must not enter the visual finger mask.
        result[15:20, 40:48, 0] = 5
        result[15:20, 40:48, 1] = 3
        # The palm sits directly above the fingers once the rows are flipped.
        result[12:21, 10:40, 0] = 5
        result[12:21, 10:40, 1] = 4
        return result


class _FakeEnv:
    def __init__(self, sim: _FakeSim) -> None:
        self.base_link_idx = 0
        self.handle = SimpleNamespace(env=SimpleNamespace(sim=sim))


class _OcclusionFakeSim(_FakeSim):
    """Hide the subject for cameras placed on the low-X side."""

    def render(self, *, camera_name: str, width: int, height: int, depth: bool):
        rendered = super().render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=depth,
        )
        if not depth:
            return rendered
        rgb, normalized_depth = rendered
        camera_id = self.model.camera_name2id(camera_name)
        if self.model.cam_pos[camera_id, 0] < 0.5:
            near_metric = 0.05
            near = self.model.vis.map.znear * self.model.stat.extent
            far = self.model.vis.map.zfar * self.model.stat.extent
            normalized = (1.0 - near / near_metric) / (1.0 - near / far)
            normalized_depth[:] = normalized
        return rgb, normalized_depth


def _request() -> ContactCameraRequest:
    return ContactCameraRequest(
        center_base_xyz=(0.4, -0.08, 0.06),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
    )


def test_libero_contact_camera_renders_dense_orthogonal_views_and_restores() -> None:
    sim = _FakeSim()
    initial_position = sim.model.cam_pos.copy()
    initial_quaternion = sim.model.cam_quat.copy()
    initial_fovy = sim.model.cam_fovy.copy()

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(_request())

    assert pair.front["images"]["rgb"].shape == (48, 80, 3)
    assert pair.side["images"]["rgb"].shape == (48, 80, 3)
    assert np.all(pair.front["images"]["rgb"] == 40)
    assert np.all(pair.side["images"]["rgb"] == 180)
    assert "depth" not in pair.front["images"]
    assert len(sim.render_poses) == 8
    front_position = sim.render_poses[0][1]
    side_position = sim.render_poses[1][1]
    assert not np.allclose(front_position, side_position)
    assert np.allclose(sim.model.cam_pos, initial_position)
    assert np.allclose(sim.model.cam_quat, initial_quaternion)
    assert np.allclose(sim.model.cam_fovy, initial_fovy)
    assert sim.forward_calls == 9
    assert pair.selection.front_sign == 1
    assert pair.selection.side_sign == 1


def test_contact_camera_attaches_pixel_exact_visual_finger_segmentation() -> None:
    sim = _SegmentationSim()

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(_request())

    for camera in (pair.front, pair.side):
        mask = camera["finger_mask"]
        assert mask.dtype == np.bool_
        assert mask.shape == (48, 80)
        assert np.all(mask[36:44, 10:18])
        assert np.all(mask[36:44, 24:32])
        assert not np.any(mask[28:33, 40:48])
        assert set(camera["images"]) == {"rgb"}
        assert not any(str(key).startswith("_mujoco") for key in camera)


def test_contact_camera_marks_only_a_thin_band_on_the_palm_underside() -> None:
    sim = _SegmentationSim()

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(_request())

    for camera in (pair.front, pair.side):
        band = camera["palm_floor_mask"]
        assert band.dtype == np.bool_
        assert band.shape == (48, 80)
        columns = np.flatnonzero(band.any(axis=0))
        assert columns.size > 0
        # The palm geom spans rows 27-35 after the vertical flip; only its
        # lowest few rows are marked, because the rest is not a contact face.
        rows = np.flatnonzero(band.any(axis=1))
        assert int(rows.max()) == 35
        assert int(rows.max() - rows.min()) < 8
        palm = np.zeros((48, 80), dtype=bool)
        palm[27:36, 10:40] = True
        assert not np.any(band & ~palm)
        # Marking the palm must not have eaten into the finger mask.
        assert not np.any(band & camera["finger_mask"])


def test_palm_band_follows_a_stepped_lower_contour() -> None:
    from vaw.context_runtime.contact_camera import _lower_edge_band

    mask = np.zeros((40, 6), dtype=bool)
    mask[10:20, 0:3] = True
    mask[10:31, 3:6] = True
    mask[:, 5] = False

    band = _lower_edge_band(mask, height=40)

    assert np.flatnonzero(band[:, 0]).max() == 19
    assert np.flatnonzero(band[:, 3]).max() == 30
    assert not band[:, 5].any()
    assert int(band[:, 0].sum()) == 5


def test_contact_framing_ceiling_scales_with_the_close_up_distance() -> None:
    from vaw.context_runtime.contact_camera import _framing_distance

    # Geometry far too tall to fit; the returned distance must saturate at the
    # ceiling, and that ceiling has to follow the configured close-up distance
    # so a narrower FOV does not silently clip the subject.
    tall = np.array([[0.0, 0.0, -4.0], [0.0, 0.0, 4.0]], dtype=np.float64)
    kwargs = {
        "horizontal_forward_base": np.array([1.0, 0.0, 0.0]),
        "up_base": np.array([0.0, 0.0, 1.0]),
        "width": 80,
        "height": 48,
        "fovy_deg": 34.0,
        "padding_m": 0.035,
    }
    near = _framing_distance(tall, None, np.zeros(3), minimum_m=0.26, **kwargs)
    far = _framing_distance(tall, None, np.zeros(3), minimum_m=0.38, **kwargs)
    assert far > near
    assert np.isclose(far / near, 0.38 / 0.26)


def test_contact_camera_reuses_requested_axis_ends_without_rescoring() -> None:
    sim = _FakeSim()
    request = ContactCameraRequest(
        center_base_xyz=(0.4, -0.08, 0.06),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
        preferred_signs=(-1, 1),
    )

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(request)

    assert pair.selection.front_sign == -1
    assert pair.selection.side_sign == 1
    assert len(sim.render_poses) == 2
    assert pair.selection.azimuth_offset_deg == 0.0


def test_side_elevation_tilts_only_the_side_camera_and_keeps_its_azimuth() -> None:
    sim = _FakeSim()
    locked_rotation = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
    request = ContactCameraRequest(
        center_base_xyz=(0.4, -0.08, 0.06),
        frame_quaternion_xyzw=tuple(Rotation.from_matrix(locked_rotation).as_quat()),
        width=80,
        panel_height=48,
        preferred_signs=(1, 1),
        side_elevation_deg=55.0,
    )

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(request)

    front_forward = np.asarray(pair.front["pose_mat"], dtype=np.float64)[:3, 2]
    side_forward = np.asarray(pair.side["pose_mat"], dtype=np.float64)[:3, 2]
    assert np.isclose(front_forward[2], 0.0, atol=1e-8)
    # The SIDE optical axis dips exactly 55 degrees below the horizon.
    assert np.isclose(
        np.degrees(np.arcsin(-side_forward[2])),
        55.0,
        atol=1e-6,
    )
    # Tilting must not swing the azimuth off the locked closing axis.
    horizontal = side_forward[:2] / np.linalg.norm(side_forward[:2])
    assert np.allclose(horizontal, locked_rotation[:2, 1], atol=1e-8)
    assert pair.selection.side_elevation_deg == 55.0
    assert pair.selection.summary()["side_elevation_deg"] == 55.0


def test_side_elevation_keeps_the_camera_on_the_framing_sphere() -> None:
    center = np.array([0.4, -0.08, 0.06])
    request_kwargs = {
        "center_base_xyz": tuple(center),
        "frame_quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
        "width": 80,
        "panel_height": 48,
        "preferred_signs": (1, 1),
    }
    level = LiberoContactCameraProvider(_FakeEnv(_FakeSim()))(
        ContactCameraRequest(**request_kwargs)
    )
    tilted = LiberoContactCameraProvider(_FakeEnv(_FakeSim()))(
        ContactCameraRequest(**request_kwargs, side_elevation_deg=55.0)
    )

    def radius(camera: dict) -> float:
        position = np.asarray(camera["pose_mat"], dtype=np.float64)[:3, 3]
        return float(np.linalg.norm(position - center))

    # Tilting orbits the subject rather than climbing away from it, so the
    # oblique panel keeps the zoom of the level one.
    assert np.isclose(radius(tilted.side), radius(level.side), rtol=1e-6)
    assert radius(level.side) > 0.0


def test_side_elevation_rejects_out_of_range_tilts() -> None:
    with pytest.raises(ValueError, match="side_elevation_deg"):
        LiberoContactCameraProvider(_FakeEnv(_FakeSim()))(
            ContactCameraRequest(
                center_base_xyz=(0.4, -0.08, 0.06),
                frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                width=80,
                panel_height=48,
                side_elevation_deg=95.0,
            )
        )


def test_contact_camera_axes_are_exactly_parallel_and_orthogonal_to_locked_frame() -> None:
    sim = _FakeSim()
    locked_rotation = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
    request = ContactCameraRequest(
        center_base_xyz=(0.4, -0.08, 0.06),
        frame_quaternion_xyzw=tuple(Rotation.from_matrix(locked_rotation).as_quat()),
        width=80,
        panel_height=48,
    )

    pair = LiberoContactCameraProvider(_FakeEnv(sim))(request)

    front_forward = np.asarray(pair.front["pose_mat"], dtype=np.float64)[:3, 2]
    side_forward = np.asarray(pair.side["pose_mat"], dtype=np.float64)[:3, 2]
    expected_front = pair.selection.front_sign * locked_rotation[:, 0]
    expected_side = pair.selection.side_sign * locked_rotation[:, 1]
    assert np.allclose(front_forward, expected_front, atol=1e-8)
    assert np.allclose(side_forward, expected_side, atol=1e-8)
    assert np.isclose(np.dot(front_forward, side_forward), 0.0, atol=1e-8)


def test_contact_camera_calibration_matches_flipped_rgb_pixel_axes() -> None:
    pair = LiberoContactCameraProvider(_FakeEnv(_FakeSim()))(_request())
    center = np.asarray(_request().center_base_xyz, dtype=np.float64)

    # Image coordinates are +u right and +v down.  Both gravity-stable views
    # must therefore project WORLD +Z upward (smaller v), while their own
    # screen-right base direction projects toward larger u.
    for camera, screen_right in (
        (pair.front, np.array([0.0, -1.0, 0.0])),
        (pair.side, np.array([1.0, 0.0, 0.0])),
    ):
        projected = project_world_to_pixel(
            np.stack(
                (
                    center,
                    center + 0.03 * screen_right,
                    center + np.array([0.0, 0.0, 0.03]),
                )
            ),
            camera["intrinsics"],
            camera["pose_mat"],
        )
        origin, right, up = projected
        assert right[0] > origin[0]
        assert up[1] < origin[1]
        assert np.all(projected[:, 2] > 0.0)


def test_libero_contact_camera_restores_after_render_failure() -> None:
    sim = _FakeSim(fail_on="sideview")
    initial_position = sim.model.cam_pos.copy()
    initial_quaternion = sim.model.cam_quat.copy()
    initial_fovy = sim.model.cam_fovy.copy()

    try:
        LiberoContactCameraProvider(_FakeEnv(sim))(_request())
    except RuntimeError as exc:
        assert "synthetic render failure" in str(exc)
    else:
        raise AssertionError("expected direct camera render to fail")

    assert np.allclose(sim.model.cam_pos, initial_position)
    assert np.allclose(sim.model.cam_quat, initial_quaternion)
    assert np.allclose(sim.model.cam_fovy, initial_fovy)


def test_contact_camera_uses_close_gravity_stable_horizontal_views() -> None:
    sim = _FakeSim()
    request = _request()
    LiberoContactCameraProvider(_FakeEnv(sim))(request)

    center_world = sim.data.xpos[0] + np.asarray(request.center_base_xyz)
    for _name, position, _quaternion in sim.render_poses:
        horizontal_distance = float(np.linalg.norm((position - center_world)[:2]))
        assert np.isclose(horizontal_distance, 0.16)
        assert position[2] == center_world[2]


def test_contact_camera_keeps_depth_private_when_scoring_subject_visibility() -> None:
    request = ContactCameraRequest(
        center_base_xyz=(0.4, -0.08, 0.06),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
        subject_points_base=np.array(
            [[0.4, -0.08, 0.06], [0.4, -0.07, 0.07]],
            dtype=np.float64,
        ),
    )
    pair = LiberoContactCameraProvider(_FakeEnv(_FakeSim()))(request)

    assert pair.selection.score > 0.0
    assert set(pair.front["images"]) == {"rgb"}
    assert set(pair.side["images"]) == {"rgb"}


def test_contact_camera_selects_pair_without_synthetic_foreground_occlusion() -> None:
    center = np.asarray(_request().center_base_xyz, dtype=np.float64)
    request = ContactCameraRequest(
        center_base_xyz=tuple(center),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
        subject_points_base=np.array(
            [
                center + [-0.01, -0.01, -0.01],
                center + [0.01, -0.01, 0.01],
                center + [-0.01, 0.01, 0.01],
                center + [0.01, 0.01, -0.01],
            ],
            dtype=np.float64,
        ),
    )

    pair = LiberoContactCameraProvider(_FakeEnv(_OcclusionFakeSim()))(request)

    assert pair.selection.front_sign == -1
    assert pair.selection.side_sign == 1
    assert pair.selection.front_visibility > 0.0
    assert pair.selection.side_visibility > 0.0


def test_contact_camera_zooms_out_to_keep_tall_destination_geometry_visible() -> None:
    sim = _FakeSim()
    center = np.asarray(_request().center_base_xyz, dtype=np.float64)
    request = ContactCameraRequest(
        center_base_xyz=tuple(center),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
        subject_points_base=np.array(
            [
                center + [-0.05, -0.04, -0.22],
                center + [0.05, 0.04, -0.20],
                center + [-0.04, 0.03, 0.20],
                center + [0.04, -0.03, 0.22],
            ],
            dtype=np.float64,
        ),
    )

    LiberoContactCameraProvider(_FakeEnv(sim))(request)

    center_world = sim.data.xpos[0] + center
    horizontal_distances = [
        float(np.linalg.norm((position - center_world)[:2]))
        for _name, position, _quaternion in sim.render_poses
    ]
    assert min(horizontal_distances) > 0.6


def test_contact_camera_required_preview_geometry_cannot_be_trimmed_by_subject_quantiles() -> None:
    sim = _FakeSim()
    center = np.asarray(_request().center_base_xyz, dtype=np.float64)
    dense_subject = np.repeat(center.reshape(1, 3), 10_000, axis=0)
    required_preview = np.array(
        [
            center + [0.0, 0.0, -0.28],
            center + [0.0, 0.0, 0.28],
        ],
        dtype=np.float64,
    )
    request = ContactCameraRequest(
        center_base_xyz=tuple(center),
        frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        width=80,
        panel_height=48,
        subject_points_base=dense_subject,
        required_points_base=required_preview,
    )

    LiberoContactCameraProvider(_FakeEnv(sim))(request)

    center_world = sim.data.xpos[0] + center
    horizontal_distances = [
        float(np.linalg.norm((position - center_world)[:2]))
        for _name, position, _quaternion in sim.render_poses
    ]
    assert min(horizontal_distances) > 0.7


def test_opposite_scene_camera_renders_one_complementary_view_and_restores() -> None:
    sim = _FakeSim()
    initial_position = sim.model.cam_pos.copy()
    initial_quaternion = sim.model.cam_quat.copy()
    initial_fovy = sim.model.cam_fovy.copy()
    request = OppositeSceneCameraRequest(
        center_base_xyz=(0.5, 0.0, 0.08),
        agentview_forward_base_xyz=(1.0, 0.0, 0.0),
        width=96,
        height=54,
    )

    camera = LiberoOppositeSceneCameraProvider(_FakeEnv(sim))(request)

    assert camera["images"]["rgb"].shape == (54, 96, 3)
    assert set(camera["images"]) == {"rgb"}
    assert camera["view_name"] == "opposite"
    assert len(sim.render_poses) == 1
    assert np.allclose(sim.model.cam_pos, initial_position)
    assert np.allclose(sim.model.cam_quat, initial_quaternion)
    assert np.allclose(sim.model.cam_fovy, initial_fovy)

    center_world = sim.data.xpos[0] + np.asarray(request.center_base_xyz)
    camera_position = sim.render_poses[0][1]
    viewing_direction = center_world - camera_position
    # 150 degrees around gravity is genuinely on the other side of the
    # agentview's +X viewing direction, while remaining oblique.
    assert float(np.dot(viewing_direction[:2], [1.0, 0.0])) < 0.0
    assert camera_position[2] > center_world[2]
