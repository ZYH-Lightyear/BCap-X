"""Small geometry helpers for the revision-local Context Runtime."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Contact-GraspNet's returned pose is 0.0166 m past the predicted fingertip
# contact along its local approach axis.
GRASP_POSE_TO_CONTACT_M = -0.0166

# Franka/LIBERO ``solve_ik`` accepts a fingertip/contact TCP target, then
# targets ``panda_hand`` at this local translation.  The backend's own value is
# preferred when available; this is the matching default for
# FrankaLiberoApiReduced.
DEFAULT_TCP_TO_HAND_LOCAL_XYZ = (0.0, 0.0, -0.1)

# Contact-GraspNet represents the parallel-jaw closing direction with local X,
# while Franka's ``panda_hand`` opens and closes along local Y.  CaP-X's full
# LIBERO API applies the same +90 degree local-Z correction before sending a
# sampled grasp to the robot.  Local Z (the grasp approach axis) is unchanged.
_GRASPNET_TO_PANDA_HAND = Rotation.from_euler("z", np.pi / 2.0).as_matrix()


def approach_axis(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return local +Z of a public wxyz grasp quaternion in robot-base."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat(np.roll(quaternion, -1)).as_matrix()[:, 2]


def shift_along_approach(
    position_xyz: np.ndarray,
    quaternion_wxyz: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    """Shift a position along the grasp frame's local +Z approach axis."""

    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    return position + float(distance_m) * approach_axis(quaternion_wxyz)


def project_world_to_pixel(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_mat: np.ndarray,
) -> np.ndarray:
    """Project robot-base points into calibrated camera pixels and depth."""

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    world_to_camera = np.linalg.inv(np.asarray(camera_pose_mat, dtype=np.float64))
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2:3]
    uvw = (np.asarray(intrinsics, dtype=np.float64) @ camera_points.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = uvw[:, :2] / np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    return np.concatenate((pixels, depth), axis=1)


def graspnet_pose_to_panda_hand(grasp_pose: np.ndarray) -> np.ndarray:
    """Convert a Contact-GraspNet pose to the Franka ``panda_hand`` frame."""

    pose = np.asarray(grasp_pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError("grasp pose must contain finite numbers")
    converted = pose.copy()
    converted[:3, :3] = pose[:3, :3] @ _GRASPNET_TO_PANDA_HAND
    return converted


def tcp_position_from_hand_pose(
    hand_position_xyz: tuple[float, float, float] | np.ndarray,
    hand_quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
    tcp_to_hand_local_xyz: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Recover the fingertip TCP position from an observed ``panda_hand`` pose.

    CaP-X solves ``p_hand = p_tcp + R(q) @ tcp_to_hand``.  Observation exposes
    the resulting hand-link pose, so execution receipts must invert that same
    transform before comparing it with the requested TCP target.
    """

    hand_position = np.asarray(hand_position_xyz, dtype=np.float64).reshape(3)
    quaternion = np.asarray(hand_quaternion_xyzw, dtype=np.float64).reshape(4)
    offset = np.asarray(tcp_to_hand_local_xyz, dtype=np.float64).reshape(3)
    if not (
        np.isfinite(hand_position).all()
        and np.isfinite(quaternion).all()
        and np.isfinite(offset).all()
    ):
        raise ValueError("hand pose and TCP offset must contain finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("hand quaternion must be non-zero")
    rotation = Rotation.from_quat(quaternion / norm)
    return hand_position - rotation.apply(offset)


def local_surface_depth(
    depth: np.ndarray,
    pixel_xy: tuple[float, float],
    *,
    radius_px: int = 2,
    absolute_gap_m: float = 0.01,
    relative_gap: float = 0.02,
) -> float:
    """Depth of the nearest unambiguous local surface around a VLM point.

    The small patch is split at metric depth discontinuities.  We select the
    surface containing the valid pixel nearest to the requested point, rather
    than taking a median across foreground and background.  Equally-near
    pixels from different surfaces are ambiguous and are rejected.
    """

    values = np.asarray(depth, dtype=np.float64)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"depth must be HxW, got {values.shape}")
    u = int(round(float(pixel_xy[0])))
    v = int(round(float(pixel_xy[1])))
    if not (0 <= u < values.shape[1] and 0 <= v < values.shape[0]):
        raise ValueError(f"point {pixel_xy} lies outside the depth image")
    y0, y1 = max(0, v - radius_px), min(values.shape[0], v + radius_px + 1)
    x0, x1 = max(0, u - radius_px), min(values.shape[1], u + radius_px + 1)
    patch = values[y0:y1, x0:x1]
    rows, cols = np.nonzero(np.isfinite(patch) & (patch > 0.0))
    if rows.size == 0:
        raise ValueError(f"no valid depth near pixel {[u, v]}")
    depths = patch[rows, cols]
    distances = np.square(rows + y0 - v) + np.square(cols + x0 - u)

    order = np.argsort(depths, kind="stable")
    sorted_depths = depths[order]
    cluster_for_sorted = np.zeros(len(order), dtype=np.int64)
    cluster = 0
    for index in range(1, len(order)):
        previous = float(sorted_depths[index - 1])
        current = float(sorted_depths[index])
        gap_limit = max(absolute_gap_m, relative_gap * min(previous, current))
        if current - previous > gap_limit:
            cluster += 1
        cluster_for_sorted[index] = cluster
    cluster_ids = np.empty(len(order), dtype=np.int64)
    cluster_ids[order] = cluster_for_sorted

    nearest = distances == distances.min()
    nearest_clusters = np.unique(cluster_ids[nearest])
    if nearest_clusters.size != 1:
        candidates = sorted(round(float(value), 6) for value in depths[nearest])
        raise ValueError(
            f"ambiguous depth discontinuity near pixel {[u, v]}: {candidates}"
        )
    selected = depths[cluster_ids == nearest_clusters[0]]
    return float(np.median(selected))


def pixel_to_base(
    pixel_xy: tuple[float, float],
    depth_m: float,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    """Lift one agentview pixel to robot-base XYZ."""

    u, v = (float(pixel_xy[0]), float(pixel_xy[1]))
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(base_from_camera, dtype=np.float64).reshape(4, 4)
    camera_ray = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    point_camera = camera_ray * float(depth_m)
    point_base = transform @ np.append(point_camera, 1.0)
    return point_base[:3]


__all__ = [
    "DEFAULT_TCP_TO_HAND_LOCAL_XYZ",
    "GRASP_POSE_TO_CONTACT_M",
    "approach_axis",
    "graspnet_pose_to_panda_hand",
    "local_surface_depth",
    "pixel_to_base",
    "project_world_to_pixel",
    "shift_along_approach",
    "tcp_position_from_hand_pose",
]
