"""Private geometry for the rigid carried-volume preview.

The proxy is deliberately presentation-only.  It summarizes sensor-derived
object points as a gravity-stable box and can be bound to the observed TCP
after a close command.  None of the numeric geometry is part of the Agent
function contract or textual context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from vaw.context_runtime.model import Pose

_MIN_POINTS = 8
_ROBUST_QUANTILES = (0.01, 0.99)
_DEFAULT_PADDING_M = 0.003


@dataclass(frozen=True)
class ObjectVolumeProxy:
    """One gravity-stable OBB derived from revision-local sensor points."""

    query: str
    source_revision: int
    center_base_xyz: tuple[float, float, float]
    rotation_base_from_obb: tuple[tuple[float, float, float], ...]
    extent_xyz_m: tuple[float, float, float]

    def base_from_obb(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(self.rotation_base_from_obb, dtype=np.float64)
        transform[:3, 3] = np.asarray(self.center_base_xyz, dtype=np.float64)
        return transform


@dataclass(frozen=True)
class ObjectProxyCandidate:
    """A sensor proxy preserved across one successful object approach.

    The candidate is deliberately independent of which planner produced the
    approach pose.  A grasp seed and a point anchored inside a detected region
    are equivalent sources of revision-local object geometry.
    """

    action_id: str
    proxy: ObjectVolumeProxy


@dataclass(frozen=True)
class AttachmentHypothesis:
    """Rigid TCP-relative volume used only for counterfactual rendering."""

    query: str
    origin_action_id: str
    tcp_from_obb: tuple[tuple[float, float, float, float], ...]
    extent_xyz_m: tuple[float, float, float]

    def base_from_obb(self, tcp_pose: Pose) -> np.ndarray:
        return _pose_matrix(tcp_pose) @ np.asarray(self.tcp_from_obb, dtype=np.float64)


def fit_gravity_stable_proxy(
    query: str,
    source_revision: int,
    points_base: np.ndarray,
    *,
    padding_m: float = _DEFAULT_PADDING_M,
) -> ObjectVolumeProxy | None:
    """Fit a robust yaw-only OBB while keeping BASE +Z as gravity up."""

    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < _MIN_POINTS:
        return None

    lower, upper = np.quantile(points, _ROBUST_QUANTILES, axis=0)
    core = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if len(core) < _MIN_POINTS:
        core = points

    yaw = _minimum_area_yaw(core[:, :2])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local = core @ rotation
    local_lower, local_upper = np.quantile(local, _ROBUST_QUANTILES, axis=0)
    extent = local_upper - local_lower + 2.0 * float(padding_m)
    if not np.isfinite(extent).all() or np.any(extent <= 1e-4):
        return None
    center_local = 0.5 * (local_lower + local_upper)
    center_base = center_local @ rotation.T
    return ObjectVolumeProxy(
        query=str(query),
        source_revision=int(source_revision),
        center_base_xyz=tuple(float(value) for value in center_base),
        rotation_base_from_obb=tuple(
            tuple(float(value) for value in row) for row in rotation
        ),
        extent_xyz_m=tuple(float(value) for value in extent),
    )


def bind_proxy_to_tcp(
    candidate: ObjectProxyCandidate,
    tcp_pose: Pose,
) -> AttachmentHypothesis:
    """Freeze a grasp proxy relative to the actual post-approach TCP."""

    tcp_from_obb = np.linalg.inv(_pose_matrix(tcp_pose)) @ candidate.proxy.base_from_obb()
    return AttachmentHypothesis(
        query=candidate.proxy.query,
        origin_action_id=candidate.action_id,
        tcp_from_obb=tuple(
            tuple(float(value) for value in row) for row in tcp_from_obb
        ),
        extent_xyz_m=candidate.proxy.extent_xyz_m,
    )


def volume_triangles_base(
    attachment: AttachmentHypothesis,
    target_tcp_pose: Pose,
) -> np.ndarray:
    """Return twelve box triangles at a counterfactual target TCP pose."""

    half = 0.5 * np.asarray(attachment.extent_xyz_m, dtype=np.float64)
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners_local = signs * half
    base_from_obb = attachment.base_from_obb(target_tcp_pose)
    corners_base = (
        corners_local @ base_from_obb[:3, :3].T + base_from_obb[:3, 3]
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return np.ascontiguousarray(corners_base[faces])


def _minimum_area_yaw(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), 0.5 * np.pi)
    candidates = np.unique(np.round(angles, decimals=12))
    best: tuple[float, float] | None = None
    for angle in candidates:
        cosine, sine = np.cos(angle), np.sin(angle)
        # Row-vector projection into candidate axes.
        local = points @ np.array([[cosine, -sine], [sine, cosine]])
        span = np.ptp(local, axis=0)
        key = (float(span[0] * span[1]), float(angle))
        if best is None or key < best:
            best = key
    return 0.0 if best is None else best[1]


def _pose_matrix(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
    transform[:3, 3] = np.asarray(pose.position_xyz, dtype=np.float64)
    return transform


__all__ = [
    "AttachmentHypothesis",
    "ObjectProxyCandidate",
    "ObjectVolumeProxy",
    "bind_proxy_to_tcp",
    "fit_gravity_stable_proxy",
    "volume_triangles_base",
]
