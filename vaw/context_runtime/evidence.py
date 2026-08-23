"""Deterministic cross-action lifecycle for grounded visual evidence.

Grounded regions and points used to expire wholesale at every observation
revision, forcing the policy to re-ground objects that nobody touched.  This
module keeps that evidence alive as long as the current observation still
supports it, using only information the harness already owns:

- the agentview camera is fixed for one episode, so pixels and base-frame
  positions of an unmoved object stay exact;
- the robot's own silhouette comes from URDF forward kinematics, never from
  perception, so self-occlusion can be excluded from the comparison;
- aligned depth distinguishes *occluded* (a closer surface in front) from
  *gone* (the background behind the archived surface is now visible).

The resulting verdict per evidence entry is one of:

``verified``
    the visible, non-robot part of the archived window still matches the
    current observation; the entry is renewed for the new revision.
``occluded``
    the check could not run (robot body or another surface in front); the
    entry is kept, flagged, and re-checked against the *original* archive on
    later revisions.
``invalidated``
    positive evidence of change (archived surface gone or replaced); the
    entry is deleted and reported.

Failure to prove presence is never treated as proof of absence: deletion
requires positive evidence, occlusion only postpones the check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vaw.context_runtime.model import ContextState, RobotState

# Comparison support below this fraction of the archived object pixels means
# the check cannot say anything; the entry stays occluded rather than guessed.
MIN_VISIBLE_FRACTION = 0.35
# Depth agreement tolerance between the archive and the current observation.
DEPTH_TOLERANCE_M = 0.015
# Fraction of visible support that must agree in depth for a verified verdict.
DEPTH_MATCH_FRACTION = 0.75
# Fraction of visible support moving beyond tolerance that proves a change.
DEPTH_CHANGE_FRACTION = 0.25
# Mean absolute uint8 RGB difference allowed on depth-consistent support.
RGB_TOLERANCE = 14.0
# Half-size of the square archive window kept around a located point.
POINT_WINDOW_HALF_PX = 14
# The FK silhouette is dilated to swallow anti-aliased robot edges.
SILHOUETTE_DILATION_PX = 3


@dataclass(frozen=True)
class EvidenceArchive:
    """Grounding-time appearance of one evidence window, kept episode-long."""

    window_xyxy_px: tuple[int, int, int, int]
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray | None
    grounded_revision: int


def clip_window(
    bbox_xyxy_px: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(np.floor(float(bbox_xyxy_px[0]))))
    y0 = max(0, int(np.floor(float(bbox_xyxy_px[1]))))
    x1 = min(int(width), int(np.ceil(float(bbox_xyxy_px[2]))))
    y1 = min(int(height), int(np.ceil(float(bbox_xyxy_px[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def point_window(
    pixel_xy: tuple[float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    return clip_window(
        (
            pixel_xy[0] - POINT_WINDOW_HALF_PX,
            pixel_xy[1] - POINT_WINDOW_HALF_PX,
            pixel_xy[0] + POINT_WINDOW_HALF_PX,
            pixel_xy[1] + POINT_WINDOW_HALF_PX,
        ),
        width,
        height,
    )


def archive_window(
    camera: dict[str, Any],
    bbox_xyxy_px: tuple[float, float, float, float],
    *,
    mask: np.ndarray | None,
    revision: int,
) -> EvidenceArchive | None:
    """Snapshot one window of the current observation for later revalidation."""

    try:
        rgb = np.asarray(camera["images"]["rgb"])
        depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
    except (KeyError, TypeError):
        return None
    if rgb.ndim != 3 or depth.shape[:2] != rgb.shape[:2]:
        return None
    window = clip_window(bbox_xyxy_px, rgb.shape[1], rgb.shape[0])
    if window is None:
        return None
    x0, y0, x1, y1 = window
    return EvidenceArchive(
        window_xyxy_px=window,
        rgb=np.ascontiguousarray(rgb[y0:y1, x0:x1]),
        depth=np.ascontiguousarray(depth[y0:y1, x0:x1]),
        mask=(
            np.ascontiguousarray(mask[y0:y1, x0:x1])
            if mask is not None and mask.shape == rgb.shape[:2]
            else None
        ),
        grounded_revision=revision,
    )


def robot_silhouette(
    camera: dict[str, Any],
    robot: RobotState | None,
) -> np.ndarray | None:
    """Rasterize the full FK robot mesh into the camera as a boolean mask.

    The silhouette is self-knowledge, not perception: joints are observed,
    the mesh comes from the URDF, and the camera calibration is episode
    truth.  ``None`` means the silhouette is unavailable and callers must
    not exclude any pixels.
    """

    if (
        robot is None
        or robot.joint_positions_rad is None
        or robot.gripper_opening is None
    ):
        return None
    from vaw.context_runtime.gripper_mesh import (
        load_panda_urdf_fk,
        rasterize_silhouette,
    )

    fk = load_panda_urdf_fk()
    robot_triangles = getattr(fk, "robot_triangles", None) if fk is not None else None
    if not callable(robot_triangles):
        return None
    try:
        triangles = robot_triangles(
            np.asarray(robot.joint_positions_rad, dtype=np.float64),
            float(robot.gripper_opening),
        )
    except (RuntimeError, ValueError):
        return None
    try:
        rgb = np.asarray(camera["images"]["rgb"])
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return None
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    camera_from_base = np.linalg.inv(base_from_camera)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    points_camera = (homogeneous @ camera_from_base.T)[:, :3]
    depth = points_camera[:, 2]
    uvw = points_camera @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / depth[:, None]
    mask = rasterize_silhouette(
        uv.reshape(-1, 3, 2),
        depth.reshape(-1, 3),
        rgb.shape[1],
        rgb.shape[0],
    )
    if SILHOUETTE_DILATION_PX > 0 and np.any(mask):
        from scipy import ndimage

        mask = ndimage.binary_dilation(mask, iterations=SILHOUETTE_DILATION_PX)
    return mask


def _window_verdict(
    archive: EvidenceArchive,
    rgb: np.ndarray,
    depth: np.ndarray,
    silhouette: np.ndarray | None,
) -> str:
    if rgb.shape[:2] != depth.shape[:2]:
        return "invalidated"
    x0, y0, x1, y1 = archive.window_xyxy_px
    if x1 > rgb.shape[1] or y1 > rgb.shape[0]:
        # The camera geometry changed under us; nothing archived is comparable.
        return "invalidated"
    current_rgb = rgb[y0:y1, x0:x1].astype(np.float64)
    current_depth = depth[y0:y1, x0:x1].astype(np.float64)
    reference_rgb = archive.rgb.astype(np.float64)
    reference_depth = archive.depth

    support = np.isfinite(reference_depth) & (reference_depth > 0.0)
    support &= np.isfinite(current_depth) & (current_depth > 0.0)
    if archive.mask is not None:
        support &= archive.mask
    total_support = int(support.sum())
    if total_support == 0:
        return "occluded"
    if silhouette is not None:
        visible = support & ~silhouette[y0:y1, x0:x1]
    else:
        visible = support
    visible_count = int(visible.sum())
    if visible_count / total_support < MIN_VISIBLE_FRACTION:
        return "occluded"

    difference = current_depth[visible] - reference_depth[visible]
    same = float(np.mean(np.abs(difference) <= DEPTH_TOLERANCE_M))
    farther = float(np.mean(difference > DEPTH_TOLERANCE_M))
    nearer = float(np.mean(difference < -DEPTH_TOLERANCE_M))
    rgb_delta = float(np.mean(np.abs(current_rgb[visible] - reference_rgb[visible])))

    if same >= DEPTH_MATCH_FRACTION and rgb_delta <= RGB_TOLERANCE:
        return "verified"
    if farther >= DEPTH_CHANGE_FRACTION:
        # The background behind the archived surface is exposed: positive
        # evidence that the object left this window.
        return "invalidated"
    if nearer >= DEPTH_CHANGE_FRACTION:
        # A non-robot surface moved in front; presence is unprovable either
        # way, so postpone rather than delete.
        return "occluded"
    return "invalidated"


def revalidate_evidence(
    state: ContextState,
    private: Any,
    camera: dict[str, Any],
    *,
    invalidate_region_ids: tuple[str, ...] = (),
    invalidate_queries: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Renew, flag or delete carried evidence against the new observation.

    ``invalidate_region_ids``/``invalidate_queries`` name evidence whose
    change is implied by the action itself (the grasped object, the contact
    target); those entries are removed without a pixel vote.
    """

    if not state.regions and not state.points:
        private.evidence_archives.clear()
        return None
    try:
        rgb = np.asarray(camera["images"]["rgb"])
        depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
    except (KeyError, TypeError):
        rgb = None
        depth = None

    revision = state.observation_revision
    normalized_queries = {
        " ".join(str(query).casefold().split()) for query in invalidate_queries
    }
    removed: list[dict[str, str]] = []
    verified: list[str] = []
    occluded: list[str] = []
    silhouette = (
        robot_silhouette(camera, state.robot) if rgb is not None else None
    )

    def _forced(evidence_id: str, query: str) -> bool:
        return (
            evidence_id in invalidate_region_ids
            or " ".join(query.casefold().split()) in normalized_queries
        )

    removed_region_ids: set[str] = set()
    for region_id in list(state.regions):
        region = state.regions[region_id]
        if _forced(region_id, region.query):
            verdict = "invalidated"
            reason = "action_target"
        else:
            archive = private.evidence_archives.get(region_id)
            if archive is None or rgb is None:
                verdict = "invalidated"
                reason = "unverifiable"
            else:
                verdict = _window_verdict(archive, rgb, depth, silhouette)
                reason = "changed"
        if verdict == "invalidated":
            removed.append({"id": region_id, "query": region.query, "reason": reason})
            removed_region_ids.add(region_id)
            del state.regions[region_id]
            private.region_masks.pop(region_id, None)
            private.region_geometry.pop(region_id, None)
            private.evidence_archives.pop(region_id, None)
        elif verdict == "occluded":
            state.regions[region_id] = replace(
                region, source_revision=revision, status="occluded"
            )
            occluded.append(region_id)
        else:
            state.regions[region_id] = replace(
                region, source_revision=revision, status="verified"
            )
            verified.append(region_id)

    for point_id in list(state.points):
        point = state.points[point_id]
        if point.within_region_id in removed_region_ids:
            # The parent object demonstrably changed; a point anchored on it
            # has no independent claim to validity.
            verdict = "invalidated"
            reason = "parent_region_changed"
        elif _forced(point_id, point.query):
            verdict = "invalidated"
            reason = "action_target"
        else:
            archive = private.evidence_archives.get(point_id)
            if archive is None or rgb is None:
                verdict = "invalidated"
                reason = "unverifiable"
            else:
                verdict = _window_verdict(archive, rgb, depth, silhouette)
                reason = "changed"
        if verdict == "invalidated":
            removed.append({"id": point_id, "query": point.query, "reason": reason})
            del state.points[point_id]
            private.evidence_archives.pop(point_id, None)
        elif verdict == "occluded":
            state.points[point_id] = replace(
                point, source_revision=revision, status="occluded"
            )
            occluded.append(point_id)
        else:
            state.points[point_id] = replace(
                point, source_revision=revision, status="verified"
            )
            verified.append(point_id)

    report = {
        "verified": verified,
        "occluded": occluded,
        "removed": removed,
    }
    private.trace_diagnostics["evidence_revalidation"] = report
    return report


def change_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact policy-visible projection of one revalidation report."""

    if not report:
        return None
    result: dict[str, Any] = {}
    removed = [
        f"{item['id']}({item['query']})" for item in report.get("removed", ())
    ]
    if removed:
        result["removed"] = removed
    if report.get("occluded"):
        result["occluded"] = list(report["occluded"])
    if report.get("verified"):
        result["verified"] = list(report["verified"])
    return result or None


__all__ = [
    "EvidenceArchive",
    "archive_window",
    "change_summary",
    "clip_window",
    "point_window",
    "revalidate_evidence",
    "robot_silhouette",
]
