from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vaw.context_runtime.contact_camera import (
    ContactCameraRequest,
    LiberoContactCameraProvider,
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
        assert not depth
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
        return rgb


class _FakeEnv:
    def __init__(self, sim: _FakeSim) -> None:
        self.base_link_idx = 0
        self.handle = SimpleNamespace(env=SimpleNamespace(sim=sim))


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
    assert len(sim.render_poses) == 2
    front_position = sim.render_poses[0][1]
    side_position = sim.render_poses[1][1]
    assert not np.allclose(front_position, side_position)
    assert np.allclose(sim.model.cam_pos, initial_position)
    assert np.allclose(sim.model.cam_quat, initial_quaternion)
    assert np.allclose(sim.model.cam_fovy, initial_fovy)
    assert sim.forward_calls == 3


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
