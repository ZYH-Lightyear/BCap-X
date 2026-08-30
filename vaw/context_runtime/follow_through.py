"""Perception-derived source follow-through after a close + lift.

This is not a gripper-opening readout.  It compares the source object's
observed location to the TCP displacement since the last successful close.
The status stays ``unknown`` until a lift is large enough to tell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from vaw.context_runtime.model import RegionEvidence
    from vaw.context_runtime.workspace import ContextWorkspace

FollowThroughStatus = Literal["unknown", "followed", "stationary"]

_TCP_LIFT_M = 0.015
_CENTROID_STILL_M = 0.008
_CENTROID_MOVED_M = 0.012
_BBOX_STILL_PX = 6.0
_BBOX_MOVED_PX = 10.0


@dataclass(frozen=True)
class GraspFollowThrough:
    query: str | None
    close_tcp_xyz: tuple[float, float, float]
    close_centroid_xyz: tuple[float, float, float] | None
    close_bbox_xyxy: tuple[float, float, float, float] | None
    status: FollowThroughStatus = "unknown"

    def summary(self) -> dict[str, Any] | None:
        if self.status == "unknown":
            return None
        result: dict[str, Any] = {"status": self.status}
        if self.query:
            result["query"] = self.query
        return result


def capture_close_snapshot(workspace: ContextWorkspace) -> None:
    """Snapshot the source object at the instant before close refreshes."""

    robot = workspace.state.robot
    if robot is None or robot.tcp_pose is None:
        workspace._private.grasp_follow_through = None
        return
    artifacts = workspace._private.last_physical_artifacts
    query = artifacts.subject_query if artifacts is not None else None
    region = _matching_region(workspace, query)
    if region is None and workspace.state.regions:
        region = next(reversed(tuple(workspace.state.regions.values())))
        if query is None:
            query = region.query
    workspace._private.grasp_follow_through = GraspFollowThrough(
        query=query,
        close_tcp_xyz=tuple(float(value) for value in robot.tcp_pose.position_xyz),
        close_centroid_xyz=_region_centroid(workspace, region),
        close_bbox_xyxy=region.bbox_xyxy_px if region is not None else None,
    )


def clear_follow_through(workspace: ContextWorkspace) -> None:
    workspace._private.grasp_follow_through = None


def refresh_follow_through(workspace: ContextWorkspace) -> None:
    """Promote unknown → followed/stationary after a decisive lift."""

    record = workspace._private.grasp_follow_through
    if record is None or record.status != "unknown":
        return
    robot = workspace.state.robot
    if robot is None or robot.tcp_pose is None:
        return
    tcp = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64)
    close_tcp = np.asarray(record.close_tcp_xyz, dtype=np.float64)
    tcp_delta = tcp - close_tcp
    tcp_move = float(np.linalg.norm(tcp_delta))
    if tcp_move < _TCP_LIFT_M:
        return
    region = _matching_region(workspace, record.query)
    if region is None:
        return
    centroid = _region_centroid(workspace, region)
    status: FollowThroughStatus | None = None
    if centroid is not None and record.close_centroid_xyz is not None:
        centroid_delta = np.asarray(centroid, dtype=np.float64) - np.asarray(
            record.close_centroid_xyz, dtype=np.float64
        )
        moved = float(np.linalg.norm(centroid_delta))
        if moved < _CENTROID_STILL_M:
            status = "stationary"
        elif moved >= _CENTROID_MOVED_M:
            alignment = float(
                np.dot(centroid_delta, tcp_delta) / (moved * tcp_move)
            )
            status = "followed" if alignment > 0.4 else "stationary"
    elif record.close_bbox_xyxy is not None:
        old = np.asarray(record.close_bbox_xyxy, dtype=np.float64)
        new = np.asarray(region.bbox_xyxy_px, dtype=np.float64)
        old_c = np.array([(old[0] + old[2]) / 2.0, (old[1] + old[3]) / 2.0])
        new_c = np.array([(new[0] + new[2]) / 2.0, (new[1] + new[3]) / 2.0])
        pixel = float(np.linalg.norm(new_c - old_c))
        if pixel < _BBOX_STILL_PX:
            status = "stationary"
        elif pixel >= _BBOX_MOVED_PX:
            status = "followed"
    if status is not None:
        workspace._private.grasp_follow_through = replace(record, status=status)


def follow_through_summary(workspace: ContextWorkspace) -> dict[str, Any] | None:
    record = workspace._private.grasp_follow_through
    return record.summary() if record is not None else None


def _matching_region(
    workspace: ContextWorkspace, query: str | None
) -> RegionEvidence | None:
    if not query:
        return None
    needle = " ".join(query.casefold().split())
    for region in workspace.state.regions.values():
        if " ".join(region.query.casefold().split()) == needle:
            return region
    return None


def _region_centroid(
    workspace: ContextWorkspace, region: RegionEvidence | None
) -> tuple[float, float, float] | None:
    if region is None:
        return None
    geometry = workspace._private.region_geometry.get(region.region_id)
    if geometry is None:
        return None
    points = np.asarray(geometry.filtered_object_points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    centroid = np.mean(points[:, :3], axis=0)
    return tuple(float(value) for value in centroid)
