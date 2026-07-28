"""Release helpers for center-aware placement and settle-before-retreat."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

DEFAULT_SETTLE_STEPS = 60


def compute_tcp_release_pos(
    placement_affordance: dict[str, Any],
    held_object_frame: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the TCP target for release.

    Prefer the typed affordance ``position`` (already TCP). Fall back to
    ``desired_object_center`` minus ``object_center_offset_from_grasp`` only when
    ``position`` is missing.
    """

    if placement_affordance and placement_affordance.get("position") is not None:
        return np.asarray(placement_affordance["position"], dtype=float)

    if not placement_affordance or placement_affordance.get("desired_object_center") is None:
        raise ValueError("missing placement_affordance.position / desired_object_center")
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
    quat = (placement_affordance or {}).get("quaternion_wxyz")
    if quat is None:
        quat = (placement_affordance or {}).get("place_quat", default)
    return np.asarray(quat, dtype=float)


def settle_after_open(
    open_fn: Callable[..., Any],
    *,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
) -> None:
    """Open the gripper and settle before any retreat motion.

    CapX gripper-open already steps the simulator. Prefer an explicit
    ``settle_steps`` / ``steps`` kwarg when the API accepts it; otherwise call
    once and treat that blocking open as the settle.
    """

    try:
        open_fn(settle_steps=int(settle_steps))
        return
    except TypeError:
        pass
    try:
        open_fn(steps=int(settle_steps))
        return
    except TypeError:
        pass
    open_fn()


def release_execution_checklist() -> list[str]:
    """Ordered executor checklist for a place trajectory."""

    return [
        "Execute transport_hover and record its motion status.",
        "Execute release_descend to affordance.position (TCP) and record its status.",
        "A failed primitive is evidence; it does not automatically stop later actions.",
        "Call settle_after_open on the gripper-open API; do not retreat yet.",
        "Only after settle, execute retreat_up.",
    ]
