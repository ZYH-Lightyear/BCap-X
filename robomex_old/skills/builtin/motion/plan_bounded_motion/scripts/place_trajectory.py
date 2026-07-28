"""Fixed place-trajectory template from a GaP-style placement affordance."""

from __future__ import annotations

from typing import Any


def build_place_trajectory(
    affordance: dict[str, Any],
    *,
    approach_height: float | None = None,
) -> dict[str, Any]:
    """Build the canonical transport → release → open → retreat trajectory.

    ``affordance["position"]`` must already be the executable TCP drop pose.
    Never substitute ``desired_object_center`` for TCP.
    """

    if not isinstance(affordance, dict):
        raise ValueError("affordance must be a dict")
    tcp = _finite_vec3(affordance.get("position"), "position")
    quat = _finite_quat(
        affordance.get("quaternion_wxyz") or affordance.get("place_quat"),
        "quaternion_wxyz",
    )

    if affordance.get("approach_position") is not None:
        hover = _finite_vec3(affordance["approach_position"], "approach_position")
    else:
        height = float(
            approach_height
            if approach_height is not None
            else affordance.get("approach_height", 0.20)
        )
        hover = [tcp[0], tcp[1], tcp[2] + height]

    retreat_z = max(hover[2], tcp[2] + float(affordance.get("approach_height", 0.20)))
    retreat = [tcp[0], tcp[1], retreat_z]

    waypoints = [
        {
            "name": "transport_hover",
            "phase": "transport",
            "position_xyz": hover,
            "quaternion_wxyz": quat,
            "gripper": "hold",
            "note": "High transport above the drop TCP; do not invent a new XY/Z.",
        },
        {
            "name": "release_descend",
            "phase": "release",
            "position_xyz": tcp,
            "quaternion_wxyz": quat,
            "gripper": "hold",
            "note": "Descend to affordance.position (TCP). Never use desired_object_center.",
        },
        {
            "name": "open_settle",
            "phase": "open",
            "position_xyz": tcp,
            "quaternion_wxyz": quat,
            "gripper": "open",
            "note": "Open at the drop TCP and settle before retreat.",
        },
        {
            "name": "retreat_up",
            "phase": "retreat",
            "position_xyz": retreat,
            "quaternion_wxyz": quat,
            "gripper": "open",
            "note": "Straight-up retract after settle.",
        },
    ]
    return {
        "feasible": True,
        "waypoints": waypoints,
        "note": (
            "GaP-style place template: transport_hover → release_descend → "
            "open_settle → retreat_up. IK must validate these poses; do not "
            "rewrite release height on failure—finish infeasible instead."
        ),
    }


def _finite_vec3(raw: Any, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"affordance.{label} must be a length-3 finite vector")
    values = [float(v) for v in raw]
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        raise ValueError(f"affordance.{label} must be finite")
    return values


def _finite_quat(raw: Any, label: str) -> list[float]:
    if raw is None:
        return [0.0, 1.0, 0.0, 0.0]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"affordance.{label} must be a length-4 finite quaternion")
    values = [float(v) for v in raw]
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        raise ValueError(f"affordance.{label} must be finite")
    return values
