"""Pure geometry helpers shared by ops, preview and render. No capx imports.

Also the single place the gripper's frame conventions are reconciled. Three
conventions meet at a grasp and none of them coincide; keeping the conversions
here means every op, preview and receipt speaks the same one.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Gripper frame conventions.
#
# A candidate's ``position`` in this workspace means *where the fingers close*.
# That is the only convention a model can reason about from a picture, and the
# only one whose number can be checked against the object's own points. The two
# constants below map it to and from the two foreign conventions:
#
# 1. ``plan_grasp`` returns Contact-GraspNet poses shifted +0.12 m along the
#    local approach axis, on top of a frame whose contact point already sits
#    0.1034 m out. Net: the returned position is 0.0166 m *past* the contact.
# 2. ``solve_ik`` treats its ``position`` as a "TCP" that is not the fingertips:
#    it applies ``_TCP_OFFSET = (0, 0, -0.1)`` to get the panda_hand link, while
#    the fingertips are a measured 0.0114 m in *front* of that link. Net: the
#    fingers close 0.0886 m behind whatever position you pass in.
#
# Passing (1) straight into (2) — which is what the first live runs did — closes
# the fingers 0.072 m short of the object every single time. Measured on five
# candidates of the alphabet-soup can: 0.0717 / 0.0674 / 0.0751 / 0.0764 /
# 0.0727 m, which is the derived 0.0886 - 0.0166 to within a millimetre.
# --------------------------------------------------------------------------- #

#: plan_grasp's returned position -> the contact point it predicted.
GRASP_POSE_TO_CONTACT_M = -0.0166
#: contact point -> the ``position`` solve_ik / goto_pose expect.
CONTACT_TO_IK_TARGET_M = 0.0886
#: fingertips -> the panda_hand link the observation reports.
CONTACT_TO_HAND_M = -0.0114


def approach_axis(quat_wxyz: np.ndarray) -> np.ndarray:
    """Unit vector the gripper advances along (local +z of the grasp frame)."""
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()[:, 2]


def shift_along_approach(
    position: np.ndarray, quat_wxyz: np.ndarray, distance: float
) -> np.ndarray:
    """Move ``position`` by ``distance`` along the grasp frame's approach axis."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    return pos + distance * approach_axis(quat_wxyz)


def project_world_to_pixel(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_mat: np.ndarray,
) -> np.ndarray:
    """Project world-frame points into pixel coordinates.

    Args:
        points_world: (N, 3) world-frame points.
        intrinsics: (3, 3) pinhole intrinsics.
        camera_pose_mat: (4, 4) camera-to-world extrinsics (pose of camera).

    Returns:
        (N, 3) array of [u, v, z_cam]; points behind the camera get z_cam <= 0.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    world_to_cam = np.linalg.inv(np.asarray(camera_pose_mat, dtype=np.float64))
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (world_to_cam @ homo.T).T[:, :3]
    z = cam[:, 2:3]
    uvw = (np.asarray(intrinsics, dtype=np.float64) @ cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.concatenate([uv, z], axis=1)


def mask_to_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_mat: np.ndarray,
    max_points: int = 5000,
) -> np.ndarray:
    """Deproject masked depth pixels to world-frame 3D points. Returns (N, 3)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    mask = np.asarray(mask, dtype=bool)
    vs, us = np.nonzero(mask)
    zs = depth[vs, us]
    valid = zs > 1e-6
    us, vs, zs = us[valid], vs[valid], zs[valid]
    if len(us) == 0:
        return np.zeros((0, 3))
    if len(us) > max_points:
        idx = np.random.default_rng(0).choice(len(us), max_points, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]
    K = np.asarray(intrinsics, dtype=np.float64)
    x = (us - K[0, 2]) / K[0, 0] * zs
    y = (vs - K[1, 2]) / K[1, 1] * zs
    cam_pts = np.stack([x, y, zs, np.ones_like(zs)], axis=1)
    world = (np.asarray(camera_pose_mat, dtype=np.float64) @ cam_pts.T).T[:, :3]
    return world


def depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_mat: np.ndarray,
    stride: int = 4,
    max_points: int = 20000,
) -> np.ndarray:
    """Deproject a full depth image (strided) to world points. Returns (N, 3)."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    full = np.zeros_like(depth, dtype=bool)
    full[::stride, ::stride] = True
    return mask_to_world_points(depth, full, intrinsics, camera_pose_mat, max_points=max_points)


def interpolate_path(start: np.ndarray, end: np.ndarray, num: int = 24) -> np.ndarray:
    """Linear position path (num, 3) from start to end, endpoints included."""
    t = np.linspace(0.0, 1.0, num)[:, None]
    return np.asarray(start).reshape(1, 3) * (1 - t) + np.asarray(end).reshape(1, 3) * t


def min_clearance(
    path: np.ndarray,
    obstacle_points: np.ndarray,
    chunk: int = 4096,
) -> float:
    """Minimum distance (m) between a polyline's vertices and an obstacle cloud."""
    if len(obstacle_points) == 0 or len(path) == 0:
        return float("inf")
    best = float("inf")
    obs = np.asarray(obstacle_points, dtype=np.float64)
    for i in range(0, len(obs), chunk):
        d = np.linalg.norm(path[:, None, :] - obs[None, i : i + chunk, :], axis=-1)
        best = min(best, float(d.min()))
    return best


def rotate_quat_wxyz(quat_wxyz: np.ndarray, axis: str, degrees: float) -> np.ndarray:
    """Rotate a wxyz quaternion about a world axis ('x'|'y'|'z') by degrees."""
    from scipy.spatial.transform import Rotation

    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be x/y/z, got '{axis}'")
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # xyzw
    delta = Rotation.from_euler(axis, degrees, degrees=True)
    out = (delta * r).as_quat()  # xyzw
    return np.array([out[3], out[0], out[1], out[2]], dtype=np.float64)
