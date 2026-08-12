"""Simulation-only, session-locked Contact Camera rendering for LIBERO-PRO.

The policy-visible Contact views are rendered directly by MuJoCo instead of
reprojecting surfaces observed by agentview / wrist RGB-D.  This deliberately
adds two simulation cameras and must not be described as sensor re-layout in
experiments.  Camera mutations are restored immediately after each render and
never enter the public Function or manifest contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ContactCameraRequest:
    """One immutable Contact Camera frame in robot-base coordinates."""

    center_base_xyz: tuple[float, float, float]
    frame_quaternion_xyzw: tuple[float, float, float, float]
    width: int
    panel_height: int


@dataclass(frozen=True)
class ContactCameraPair:
    """Direct MuJoCo RGB cameras plus private calibration for overlays."""

    front: dict[str, Any]
    side: dict[str, Any]


class LiberoContactCameraProvider:
    """Render gravity-stable orthogonal cameras without stepping LIBERO."""

    _CAMERAS: tuple[tuple[Literal["front", "side"], str, int], ...] = (
        ("front", "frontview", 0),
        ("side", "sideview", 1),
    )

    def __init__(
        self,
        env: Any,
        *,
        distance_m: float = 0.34,
        fovy_deg: float = 44.0,
    ) -> None:
        self._env = env
        self._distance_m = float(distance_m)
        self._fovy_deg = float(fovy_deg)
        if not np.isfinite(self._distance_m) or self._distance_m <= 0.0:
            raise ValueError("distance_m must be positive and finite")
        if not np.isfinite(self._fovy_deg) or not 5.0 <= self._fovy_deg <= 120.0:
            raise ValueError("fovy_deg must be in [5, 120]")

    def __call__(self, request: ContactCameraRequest) -> ContactCameraPair:
        sim = self._env.handle.env.sim
        center_base = _vector3(request.center_base_xyz, "center_base_xyz")
        frame_quaternion = _quaternion_xyzw(
            request.frame_quaternion_xyzw,
            "frame_quaternion_xyzw",
        )
        width = int(request.width)
        height = int(request.panel_height)
        if width <= 0 or height <= 0:
            raise ValueError("contact camera dimensions must be positive")

        base_position_world, base_rotation_world = _base_pose_world(self._env, sim)
        center_world = base_position_world + base_rotation_world @ center_base
        frame_rotation_world = (
            base_rotation_world @ Rotation.from_quat(frame_quaternion).as_matrix()
        )

        saved: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        rendered: dict[str, dict[str, Any]] = {}
        try:
            for view_name, camera_name, forward_index in self._CAMERAS:
                camera_id = int(sim.model.camera_name2id(camera_name))
                saved.append(
                    (
                        camera_id,
                        np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy(),
                        np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy(),
                        float(sim.model.cam_fovy[camera_id]),
                    )
                )
                forward_world = frame_rotation_world[:, forward_index]
                up_world = frame_rotation_world[:, 2]
                camera_position_world = center_world - self._distance_m * forward_world
                camera_rotation_world = _look_at_rotation(forward_world, up_world)
                camera_quaternion_xyzw = Rotation.from_matrix(camera_rotation_world).as_quat()

                sim.model.cam_pos[camera_id] = camera_position_world
                sim.model.cam_quat[camera_id] = np.roll(camera_quaternion_xyzw, 1)
                sim.model.cam_fovy[camera_id] = self._fovy_deg
                sim.forward()
                rgb = sim.render(
                    camera_name=camera_name,
                    width=width,
                    height=height,
                    depth=False,
                )
                rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1])
                intrinsics = _intrinsics(width, height, self._fovy_deg)
                base_from_camera = _base_from_image_camera(
                    base_position_world,
                    base_rotation_world,
                    camera_position_world,
                    camera_rotation_world,
                )
                rendered[view_name] = {
                    "images": {"rgb": rgb},
                    "intrinsics": intrinsics,
                    "pose_mat": base_from_camera,
                    "view_name": view_name,
                }
        finally:
            for camera_id, position, quaternion, fovy in saved:
                sim.model.cam_pos[camera_id] = position
                sim.model.cam_quat[camera_id] = quaternion
                sim.model.cam_fovy[camera_id] = fovy
            if saved:
                sim.forward()

        if "front" not in rendered or "side" not in rendered:
            raise RuntimeError("LIBERO contact cameras did not render both views")
        return ContactCameraPair(front=rendered["front"], side=rendered["side"])


def _base_pose_world(env: Any, sim: Any) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(env.base_link_idx)
    position = np.asarray(sim.data.xpos[body_id], dtype=np.float64).reshape(3)
    quaternion_wxyz = np.asarray(sim.data.xquat[body_id], dtype=np.float64).reshape(4)
    rotation = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    return position.copy(), rotation


def _look_at_rotation(forward_world: np.ndarray, up_hint_world: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera axes; cameras look along local -Z."""

    forward = _unit(forward_world, "camera forward")
    up_hint = _unit(up_hint_world, "camera up")
    right = _unit(np.cross(forward, up_hint), "camera right")
    up = _unit(np.cross(right, forward), "camera corrected up")
    rotation = np.column_stack((right, up, -forward))
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("contact camera rotation is not right-handed")
    return rotation


def _base_from_image_camera(
    base_position_world: np.ndarray,
    base_rotation_world: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
) -> np.ndarray:
    """Match the image-camera convention already used by FrankaLiberoTask."""

    base_from_mujoco_rotation = base_rotation_world.T @ camera_rotation_world
    # MuJoCo cameras look along local -Z with +Y up.  After vertically
    # flipping ``sim.render`` into normal image row order, our pinhole helpers
    # expect +X right, +Y down and +Z forward.  This is the same
    # ``Ry(pi) @ Rz(pi)`` correction used by FrankaLiberoTask, i.e.
    # diag(+1, -1, -1).  Using only ``Ry(pi)`` mirrors both overlay axes while
    # leaving the directly rendered RGB unchanged.
    base_from_image_rotation = base_from_mujoco_rotation @ np.diag((1.0, -1.0, -1.0))
    base_from_image_translation = base_rotation_world.T @ (
        camera_position_world - base_position_world
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = base_from_image_rotation
    transform[:3, 3] = base_from_image_translation
    return transform


def _intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    return np.array(
        [
            [focal, 0.0, 0.5 * width],
            [0.0, focal, 0.5 * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def _quaternion_xyzw(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not np.isfinite(result).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite xyzw quaternion")
    return result / norm


def _unit(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero and finite")
    return result / norm


__all__ = [
    "ContactCameraPair",
    "ContactCameraRequest",
    "LiberoContactCameraProvider",
]
