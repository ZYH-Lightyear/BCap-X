"""Pure geometry helpers shared by ops, preview and render. No capx imports.

Candidate positions use the public CaP-X fingertip/contact TCP convention and
are passed to ``solve_ik`` unchanged.  The backend converts that target to its
internal ``panda_hand`` link.  ``robot_cartesian_pos`` reports the hand-link
pose, so code evaluating an executed target must invert the backend's TCP
offset before comparing positions.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Grasp-planner frame convention.
#
# A candidate's ``position`` in this workspace means *where the fingers close*.
# That is the only convention a model can reason about from a picture, and the
# only one whose number can be checked against the object's own points.
#
# ``plan_grasp`` returns Contact-GraspNet poses shifted +0.12 m along the local
# approach axis, on top of a frame whose contact point already sits 0.1034 m
# out. Net: the returned position is 0.0166 m past the contact. This is the only
# foreign-frame correction VAW owns. ``solve_ik`` applies its own TCP offset
# internally, so candidate positions are passed to it unchanged.
# --------------------------------------------------------------------------- #

#: plan_grasp's returned position -> the contact point it predicted.
GRASP_POSE_TO_CONTACT_M = -0.0166


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
