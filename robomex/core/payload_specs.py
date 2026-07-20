"""Canonical payload specs for cross-agent artifact schemas (M1-B3).

Single source of truth shared by three consumers:

- the artifact store rejects a bad payload at ``publish`` time,
- the specialist's typed-finish gate rejects it inside the producer's own
  budget so the agent can repair it immediately,
- the producer's output contract renders the same canonical shape into its
  system prompt.

Like ``robomex.core.edge_events``, this module is a dependency-free leaf so
``robomex.prompts`` and ``robomex.authoring`` can both import it without
cycles. Validation errors are written for LLM repair: they state what was
rejected and which lever fixes it, never just "invalid".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math


TRAJECTORY_PHASES: tuple[str, ...] = (
    "approach",
    "pregrasp",
    "grasp",
    "contact",
    "close",
    "open",
    "lift",
    "transport",
    "release",
    "retreat",
    "home",
)

GRIPPER_ACTIONS: tuple[str, ...] = ("open", "close", "hold")

PRIMITIVE_STATUSES: tuple[str, ...] = ("succeeded", "failed", "skipped")

MAX_TRAJECTORY_WAYPOINTS = 16


@dataclass(frozen=True)
class PayloadSpec:
    """One schema's canonical payload shape: validator + prompt-renderable doc."""

    schema: str
    doc: str
    validator: Callable[[dict[str, Any]], str | None]

    def validate(self, payload: dict[str, Any]) -> str | None:
        if not isinstance(payload, dict):
            return "payload must be a JSON object"
        return self.validator(payload)


def _finite_floats(raw: Any, length: int | None = None) -> list[float] | None:
    """Parse a finite float vector; None when the value does not qualify."""

    if not isinstance(raw, (list, tuple)):
        return None
    if length is not None and len(raw) != length:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in values):
        return None
    return values


def _validate_waypoint(index: int, raw: Any) -> str | None:
    label = f"waypoints[{index}]"
    if not isinstance(raw, dict):
        return f"{label} must be a JSON object"
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return f"{label} needs a non-empty string `name`"
    phase = raw.get("phase")
    if phase not in TRAJECTORY_PHASES:
        return (
            f"{label} `phase` is {phase!r}; use exactly one of: "
            f"{', '.join(TRAJECTORY_PHASES)}"
        )
    has_pose = "position_xyz" in raw
    has_joints = "joints" in raw
    if has_pose == has_joints:
        return (
            f"{label} must declare exactly one of `position_xyz` (Cartesian) "
            "or `joints` (joint-space), not both and not neither"
        )
    if has_pose:
        if _finite_floats(raw.get("position_xyz"), 3) is None:
            return f"{label} `position_xyz` must be 3 finite floats [x, y, z]"
        if _finite_floats(raw.get("quaternion_wxyz"), 4) is None:
            return (
                f"{label} Cartesian waypoints also need `quaternion_wxyz` "
                "as 4 finite floats [w, x, y, z]"
            )
    else:
        joints = raw.get("joints")
        if not isinstance(joints, (list, tuple)) or not 6 <= len(joints) <= 8:
            return f"{label} `joints` must be a list of 6-8 finite floats"
        if _finite_floats(joints) is None:
            return f"{label} `joints` contains non-finite or non-numeric values"
    gripper = raw.get("gripper")
    if gripper is not None and gripper not in GRIPPER_ACTIONS:
        return (
            f"{label} `gripper` is {gripper!r}; use one of: "
            f"{', '.join(GRIPPER_ACTIONS)} or omit it"
        )
    return None


def _validate_trajectory(payload: dict[str, Any]) -> str | None:
    feasible = payload.get("feasible")
    if not isinstance(feasible, bool):
        return "`feasible` must be a JSON boolean (true only after IK/bounds checks)"
    if feasible is not True:
        return (
            "do not publish an infeasible trajectory; finish with "
            '`"failure_kind": "infeasible"` instead so the graph can route '
            "back to the affordance stage"
        )
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, (list, tuple)) or not waypoints:
        return (
            "`waypoints` must be a non-empty ordered list of waypoint objects; "
            "flat fields like grasp_position/pregrasp_position are not consumable "
            "by the executor"
        )
    if len(waypoints) > MAX_TRAJECTORY_WAYPOINTS:
        return (
            f"{len(waypoints)} waypoints exceeds the bounded budget of "
            f"{MAX_TRAJECTORY_WAYPOINTS}; publish only the declared "
            "approach/contact/close/lift/retreat sequence"
        )
    for index, waypoint in enumerate(waypoints):
        error = _validate_waypoint(index, waypoint)
        if error:
            return error
    return None


def _validate_affordance(payload: dict[str, Any]) -> str | None:
    if _finite_floats(payload.get("position"), 3) is None:
        return "`position` must be 3 finite floats [x, y, z] in the declared frame"
    if _finite_floats(payload.get("quaternion_wxyz"), 4) is None:
        return "`quaternion_wxyz` must be 4 finite floats [w, x, y, z]"
    approach = payload.get("approach_dir_world")
    if approach is not None and _finite_floats(approach, 3) is None:
        return "`approach_dir_world`, when present, must be 3 finite floats"
    return None


def _validate_execution_evidence(payload: dict[str, Any]) -> str | None:
    primitives = payload.get("primitives")
    if not isinstance(primitives, (list, tuple)) or not primitives:
        return (
            "`primitives` must be a non-empty ordered list of "
            '{"name": "<api call>", "status": "succeeded|failed|skipped"} '
            "entries covering every executed primitive"
        )
    for index, item in enumerate(primitives):
        label = f"primitives[{index}]"
        if not isinstance(item, dict):
            return f"{label} must be a JSON object"
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            return f"{label} needs a non-empty string `name`"
        status = item.get("status")
        if status not in PRIMITIVE_STATUSES:
            return (
                f"{label} `status` is {status!r}; use exactly one of: "
                f"{', '.join(PRIMITIVE_STATUSES)}"
            )
    if not isinstance(payload.get("all_primitives_ok"), bool):
        return "`all_primitives_ok` must be a JSON boolean"
    failed = payload.get("failed_primitive")
    if failed is not None and not isinstance(failed, str):
        return "`failed_primitive`, when present, must be the failing primitive name"
    state = payload.get("terminal_robot_state")
    if state is not None and not isinstance(state, dict):
        return (
            "`terminal_robot_state` is injected by the runtime as a JSON object; "
            "do not overwrite it with another type"
        )
    return None


_TRAJECTORY_DOC = (
    "Canonical `robomex.trajectory.v1` payload:\n"
    "- `feasible`: bool — publish only when true; on infeasibility finish with "
    '`"failure_kind": "infeasible"` instead of publishing.\n'
    "- `waypoints`: ordered list (1-"
    f"{MAX_TRAJECTORY_WAYPOINTS}) of waypoint objects. Each waypoint:\n"
    "  - `name`: short unique string;\n"
    f"  - `phase`: one of {', '.join(TRAJECTORY_PHASES)};\n"
    "  - exactly one of `position_xyz` ([x,y,z], then `quaternion_wxyz` "
    "[w,x,y,z] is required) or `joints` (6-8 floats);\n"
    f"  - optional `gripper`: one of {', '.join(GRIPPER_ACTIONS)}.\n"
    "- All numbers must be finite. Optional `note` for rationale."
)

_AFFORDANCE_DOC = (
    "Canonical `robomex.affordance.v1` payload:\n"
    "- `position`: [x, y, z] finite floats in the declared frame;\n"
    "- `quaternion_wxyz`: [w, x, y, z] finite floats;\n"
    "- optional `approach_dir_world`: [x, y, z] unit-ish direction;\n"
    "- optional `grasp_family`/`reachable`/`note` metadata."
)

_EXECUTION_EVIDENCE_DOC = (
    "Canonical `robomex.execution_evidence.v1` payload:\n"
    '- `primitives`: ordered list of {"name": "<api call>", '
    f'"status": "{PRIMITIVE_STATUSES[0]}|{PRIMITIVE_STATUSES[1]}|'
    f'{PRIMITIVE_STATUSES[2]}"}} for every executed primitive;\n'
    "  motion primitives (goto_pose/move_to_joints) return a status dict — copy "
    "`converged`, `settled`, `stalled`, `timed_out`, `steps`, `step_cap`, and "
    "`final_error` into the entry and map `converged=False` to status `failed`; "
    "primitive failure is evidence and does not itself stop later declared "
    "waypoints or gripper actions—the executor records its continuation decision "
    "and the Verifier judges the completed attempt;\n"
    "- `all_primitives_ok`: bool;\n"
    "- optional `failed_primitive`: name of the first failing primitive;\n"
    "- `terminal_robot_state` is injected by the runtime after your motion "
    "blocks — do not fabricate it and do not call get_observation for it."
)


PAYLOAD_SPECS: dict[str, PayloadSpec] = {
    spec.schema: spec
    for spec in (
        PayloadSpec("robomex.trajectory.v1", _TRAJECTORY_DOC, _validate_trajectory),
        PayloadSpec("robomex.affordance.v1", _AFFORDANCE_DOC, _validate_affordance),
        PayloadSpec(
            "robomex.execution_evidence.v1",
            _EXECUTION_EVIDENCE_DOC,
            _validate_execution_evidence,
        ),
    )
}


def payload_spec_for(schema: str) -> PayloadSpec | None:
    return PAYLOAD_SPECS.get(str(schema))


def validate_payload(schema: str, payload: dict[str, Any]) -> str | None:
    """Validate one payload against its canonical spec; None means acceptable."""

    spec = payload_spec_for(schema)
    if spec is None:
        return None
    return spec.validate(payload)
