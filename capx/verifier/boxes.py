"""Axis-aligned object boxes and transit-corridor clearance checks.

Objects are reduced to world-frame AABBs (best representation for cheap,
exact-enough collision arithmetic). The carried object is a box hanging below
the TCP; the transit corridor is the xy segment from start to goal, inflated
by the carried box's half extents plus a margin. For every obstacle box that
intrudes into the corridor, the carried box bottom must clear the obstacle
top by ``z_margin``.

All quantities are meters unless suffixed ``_cm``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectBox:
    """World-frame axis-aligned bounding box of a grounded object."""

    name: str
    center: np.ndarray  # (3,)
    extents: np.ndarray  # (3,) full sizes along x/y/z
    score: float = 1.0

    @property
    def min_z(self) -> float:
        return float(self.center[2] - self.extents[2] / 2.0)

    @property
    def max_z(self) -> float:
        return float(self.center[2] + self.extents[2] / 2.0)

    @classmethod
    def from_points(
        cls,
        name: str,
        points: np.ndarray,
        score: float = 1.0,
        lo_pct: float = 2.0,
        hi_pct: float = 98.0,
    ) -> "ObjectBox":
        """Robust AABB from a (possibly noisy) segment point cloud."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        lo = np.percentile(pts, lo_pct, axis=0)
        hi = np.percentile(pts, hi_pct, axis=0)
        return cls(name=name, center=(lo + hi) / 2.0, extents=np.maximum(hi - lo, 1e-4),
                   score=score)

    def as_dict_cm(self) -> dict:
        return {
            "name": self.name,
            "center_cm": [round(v * 100, 1) for v in self.center],
            "extents_cm": [round(v * 100, 1) for v in self.extents],
            "top_z_cm": round(self.max_z * 100, 1),
        }


def _point_segment_dist_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point ``p`` to segment ``a-b``, all in the xy plane."""
    p, a, b = p[:2], a[:2], b[:2]
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def corridor_clearance(
    held_box: ObjectBox,
    overhang: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    transit_tcp_z: float,
    obstacles: list[ObjectBox],
    xy_margin: float = 0.02,
    z_margin: float = 0.03,
) -> tuple[list[dict], float]:
    """Per-obstacle clearance rows for a straight-line transit.

    Args:
        held_box: box of the carried object (extents matter, pose ignored).
        overhang: TCP-to-carried-object-bottom distance (meters, measured).
        start_xy / goal_xy: (2,) transit segment endpoints in xy.
        transit_tcp_z: planned TCP height during the move.
        obstacles: scene boxes (the held object itself must not be included).
        xy_margin: lateral safety margin added to the corridor half-width.
        z_margin: vertical safety margin above obstacle tops.

    Returns:
        (rows, min_safe_tcp_z):
        rows: one dict per obstacle with xy intrusion and z clearance verdicts.
        min_safe_tcp_z: smallest TCP height that clears every intruding
            obstacle (meters); equals table-clearing height if none intrude.
    """
    held_half_xy = float(np.linalg.norm(held_box.extents[:2] / 2.0))
    corridor_halfwidth = held_half_xy + xy_margin
    carried_bottom_z = transit_tcp_z - overhang

    rows: list[dict] = []
    min_safe = 0.0
    for ob in obstacles:
        ob_half_xy = float(np.linalg.norm(ob.extents[:2] / 2.0))
        dist = _point_segment_dist_xy(ob.center, start_xy, goal_xy)
        intrudes = dist < corridor_halfwidth + ob_half_xy
        required_bottom = ob.max_z + z_margin
        z_clear = carried_bottom_z - ob.max_z
        ok = (not intrudes) or (carried_bottom_z >= required_bottom)
        if intrudes:
            min_safe = max(min_safe, required_bottom + overhang)
        rows.append({
            "obstacle": ob.name,
            "obstacle_top_z_cm": round(ob.max_z * 100, 1),
            "xy_dist_to_corridor_cm": round(max(dist - ob_half_xy, 0.0) * 100, 1),
            "intrudes_corridor": bool(intrudes),
            "z_clearance_cm": round(z_clear * 100, 1),
            "required_z_clearance_cm": round(z_margin * 100, 1),
            "pass": bool(ok),
        })
    return rows, float(min_safe)
