"""Release helpers for center-aware placement."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_tcp_release_pos(
    placement_affordance: dict[str, Any],
    held_object_frame: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the TCP target for release.

    ``placement_affordance["desired_object_center"]`` is where the object center
    should land. If the object was grasped off-center, subtract the stored
    ``object_center_offset_from_grasp`` so the object center, not the grasp point,
    aligns to the placement target.
    """

    if not placement_affordance or placement_affordance.get("desired_object_center") is None:
        raise ValueError("missing placement_affordance.desired_object_center")
    desired = np.asarray(placement_affordance["desired_object_center"], dtype=float)

    offset = None
    if held_object_frame:
        offset = held_object_frame.get("object_center_offset_from_grasp")
    if offset is None:
        return desired
    return desired - np.asarray(offset, dtype=float)


def place_quat_from_affordance(
    placement_affordance: dict[str, Any] | None,
    default: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0),
) -> np.ndarray:
    quat = (placement_affordance or {}).get("place_quat", default)
    return np.asarray(quat, dtype=float)
