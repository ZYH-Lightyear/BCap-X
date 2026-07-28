"""Canonical grasp-trajectory construction with structural safety checks."""

from __future__ import annotations

import math
from typing import Any

CANONICAL_TOP_DOWN_QUATERNION = [0.0, 1.0, 0.0, 0.0]


def build_grasp_trajectory(
    affordance: dict[str, Any],
    *,
    approach_height: float | None = None,
    lift_height: float | None = None,
) -> dict[str, Any]:
    """Build a bounded approach → pregrasp → grasp → lift trajectory.

    The approach and lift are always above the grasp TCP.  This structural
    invariant prevents an ambiguous approach-axis convention from producing a
    waypoint below the table.
    """

    if not isinstance(affordance, dict):
        raise ValueError("affordance must be a dict")

    grasp = _finite_vec3(
        affordance.get("position", affordance.get("pos")),
        "position",
    )
    quat = _finite_quat(
        affordance.get("quaternion_wxyz", affordance.get("quat")),
        "quaternion_wxyz",
    )
    approach = _optional_vec3(affordance.get("approach_position"), "approach_position")
    lift = _optional_vec3(
        affordance.get("lift_position", affordance.get("lift_pos")),
        "lift_position",
    )

    approach_delta = _positive_height(
        approach_height
        if approach_height is not None
        else affordance.get("approach_height", 0.15),
        "approach_height",
    )
    lift_delta = _positive_height(
        lift_height if lift_height is not None else affordance.get("lift_height", 0.20),
        "lift_height",
    )

    if approach is None:
        approach = [grasp[0], grasp[1], grasp[2] + approach_delta]
    if lift is None:
        lift = [grasp[0], grasp[1], grasp[2] + lift_delta]

    _require_above(approach, grasp, "approach_position")
    _require_above(lift, grasp, "lift_position")

    waypoints = [
        _waypoint("approach", "approach", approach, quat, "open"),
        _waypoint("pregrasp", "pregrasp", grasp, quat, "open"),
        _waypoint("grasp", "grasp", grasp, quat, "close"),
        _waypoint("lift", "lift", lift, quat, "hold"),
    ]
    return {
        "feasible": True,
        "waypoints": waypoints,
        "note": (
            "Canonical grasp template: approach above TCP → pregrasp → close → "
            "lift above TCP. IK and collision checks must validate the poses."
        ),
    }


def _waypoint(
    name: str,
    phase: str,
    position: list[float],
    quaternion: list[float],
    gripper: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "phase": phase,
        "position_xyz": list(position),
        "quaternion_wxyz": list(quaternion),
        "gripper": gripper,
    }


def _finite_vec3(raw: Any, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"affordance.{label} must be a length-3 finite vector")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"affordance.{label} must be finite")
    return values


def _optional_vec3(raw: Any, label: str) -> list[float] | None:
    return None if raw is None else _finite_vec3(raw, label)


def _finite_quat(raw: Any, label: str) -> list[float]:
    if raw is None:
        return list(CANONICAL_TOP_DOWN_QUATERNION)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"affordance.{label} must be a length-4 finite quaternion")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"affordance.{label} must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-8:
        raise ValueError(f"affordance.{label} must have non-zero norm")
    return [value / norm for value in values]


def _positive_height(raw: Any, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"affordance.{label} must be positive and finite")
    return value


def _require_above(candidate: list[float], grasp: list[float], label: str) -> None:
    if candidate[2] <= grasp[2]:
        raise ValueError(
            f"affordance.{label}.z must be above grasp position.z; "
            "refusing a downward or table-penetrating safety waypoint"
        )
