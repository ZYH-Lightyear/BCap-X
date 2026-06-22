"""Metric evidence report: textual metric evidence for VLM judgment (V1.4).

Design principle (proposal section 7.5): metric quantities (heights,
clearances, distances) are computed from point cloud geometry and handed to
the VLM as STRUCTURED TEXT. They are never rendered into images for the VLM
to eyeball -- a few centimeters is a few pixels under an oblique fixed
camera, while the same quantity is exact when computed.

A report has three layers:

1. measurements:   raw numbers with units (cm)
2. derived_checks: deterministic rule verdicts (name, rule, value, pass)
3. context:        action intent, measurement provenance, known caveats

Hard failures (e.g. IK unreachable) can reject without consulting the VLM.
The VLM receives the formatted report and judges overall consistency plus
borderline cases, returning approve / reject / uncertain.
"""

from __future__ import annotations

from typing import Any


def make_check(name: str, rule: str, value: float | bool, passed: bool, hard: bool = False) -> dict:
    """One deterministic rule verdict.

    Args:
        name: short check identifier.
        rule: human-readable rule expression (with units).
        value: the measured value the rule was evaluated on.
        passed: rule outcome.
        hard: if True, a failure rejects the action without VLM arbitration.
    """
    return {"name": name, "rule": rule, "value": value, "pass": bool(passed), "hard": bool(hard)}


def build_place_report(
    task: str,
    held_object: str,
    receptacle: str,
    surface_z_cm: float,
    overhang_cm: float,
    overhang_source: str,
    clearance_cm: float,
    planned_release_tcp_z_cm: float,
    place_point_dist_to_center_cm: float,
    region_radius_cm: float,
    ik_reachable: bool | None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metric evidence report for a PLACE action.

    Args:
        task: full task instruction.
        held_object / receptacle: object names as grounded.
        surface_z_cm: receptacle support surface height (world z, cm).
        overhang_cm: TCP-to-object-bottom distance (cm). Post-lift
            measurement preferred; pre-grasp OBB estimate acceptable if
            flagged in ``overhang_source``.
        overhang_source: e.g. "post-lift measured" or "pre-grasp OBB worst case".
        clearance_cm: planned drop clearance above the surface.
        planned_release_tcp_z_cm: planned TCP z at gripper open (cm).
        place_point_dist_to_center_cm: planned xy distance to receptacle center.
        region_radius_cm: receptacle region radius estimate.
        ik_reachable: IK verdict for the release pose (None = not checked).
        provenance: extra context (masks, frames, seeds) for the report.

    Returns:
        Report dict with measurements / derived_checks / context.
    """
    m = {
        "surface_z_cm": round(surface_z_cm, 1),
        "overhang_tcp_to_object_bottom_cm": round(overhang_cm, 1),
        "clearance_cm": round(clearance_cm, 1),
        "planned_release_tcp_z_cm": round(planned_release_tcp_z_cm, 1),
        "predicted_object_bottom_above_surface_cm": round(
            planned_release_tcp_z_cm - overhang_cm - surface_z_cm, 1
        ),
        "place_point_dist_to_receptacle_center_cm": round(place_point_dist_to_center_cm, 1),
        "receptacle_region_radius_cm": round(region_radius_cm, 1),
        "ik_reachable": ik_reachable,
    }

    expected_release = surface_z_cm + overhang_cm + clearance_cm
    checks = [
        make_check(
            "surface_z_sane", "-5cm < surface_z < 40cm", m["surface_z_cm"],
            -5.0 < surface_z_cm < 40.0,
        ),
        make_check(
            "overhang_sane", "1cm <= overhang <= 30cm", m["overhang_tcp_to_object_bottom_cm"],
            1.0 <= overhang_cm <= 30.0,
        ),
        make_check(
            "clearance_in_range", "0.5cm <= clearance <= 5cm", m["clearance_cm"],
            0.5 <= clearance_cm <= 5.0,
        ),
        make_check(
            "release_height_consistent",
            "planned_release_tcp_z == surface_z + overhang + clearance (tol 0.3cm)",
            m["planned_release_tcp_z_cm"],
            abs(planned_release_tcp_z_cm - expected_release) < 0.3,
        ),
        make_check(
            "place_point_in_region", "dist_to_center < 0.6 * region_radius",
            m["place_point_dist_to_receptacle_center_cm"],
            place_point_dist_to_center_cm < 0.6 * region_radius_cm,
        ),
    ]
    if ik_reachable is not None:
        checks.append(make_check("ik_reachable", "IK solves for release pose",
                                 ik_reachable, bool(ik_reachable), hard=True))

    return {
        "action": "place",
        "intent": f"place the held '{held_object}' onto '{receptacle}'",
        "task": task,
        "measurements": m,
        "derived_checks": checks,
        "context": {
            "overhang_source": overhang_source,
            "units": "centimeters, world frame z-up",
            **(provenance or {}),
        },
    }


def build_transit_report(
    task: str,
    held_object: str,
    goal: str,
    held_box_dict: dict,
    overhang_cm: float,
    overhang_source: str,
    transit_tcp_z_cm: float,
    corridor_rows: list[dict],
    min_safe_tcp_z_cm: float,
    obstacles: list[dict],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metric evidence report for a TRANSIT (carry) action.

    Instead of a single "lift X cm: OK" verdict, the report carries one row
    per obstacle along the corridor plus the computed minimal safe TCP
    height, so a rejection comes with its own repair value.

    Args:
        task: full task instruction.
        held_object / goal: grounded names of carried object and destination.
        held_box_dict: carried object box (``ObjectBox.as_dict_cm()``).
        overhang_cm: TCP-to-carried-object-bottom (cm) + its source note.
        transit_tcp_z_cm: planned TCP height during the move.
        corridor_rows: per-obstacle rows from ``corridor_clearance``.
        min_safe_tcp_z_cm: computed minimal safe TCP height (repair hint).
        obstacles: scene boxes as dicts (for context).
        provenance: extra context for the report.
    """
    carried_bottom = transit_tcp_z_cm - overhang_cm
    checks = [
        make_check(
            f"clear_{row['obstacle']}",
            "no corridor intrusion OR carried_bottom >= obstacle_top + margin",
            row["z_clearance_cm"],
            row["pass"],
        )
        for row in corridor_rows
    ]
    checks.append(make_check(
        "transit_height_safe", "transit_tcp_z >= min_safe_tcp_z",
        transit_tcp_z_cm, transit_tcp_z_cm >= min_safe_tcp_z_cm, hard=True,
    ))
    return {
        "action": "transit",
        "intent": f"carry the held '{held_object}' to '{goal}'",
        "task": task,
        "measurements": {
            "transit_tcp_z_cm": round(transit_tcp_z_cm, 1),
            "overhang_tcp_to_object_bottom_cm": round(overhang_cm, 1),
            "carried_object_bottom_z_cm": round(carried_bottom, 1),
            "min_safe_tcp_z_cm": round(min_safe_tcp_z_cm, 1),
            "held_object_box": held_box_dict,
        },
        "corridor_clearance_table": corridor_rows,
        "derived_checks": checks,
        "context": {
            "overhang_source": overhang_source,
            "scene_boxes": obstacles,
            "units": "centimeters, world frame z-up",
            "repair_hint": (
                f"raise transit TCP z to at least {min_safe_tcp_z_cm:.1f}cm"
                if transit_tcp_z_cm < min_safe_tcp_z_cm else "none"
            ),
            **(provenance or {}),
        },
    }


def hard_reject(report: dict[str, Any]) -> list[str]:
    """Names of failed hard checks (non-empty list means reject before VLM)."""
    return [c["name"] for c in report["derived_checks"] if c["hard"] and not c["pass"]]


def format_for_vlm(report: dict[str, Any]) -> str:
    """Render the report as the text block handed to the VLM judge."""
    lines = [
        "You are verifying a robot manipulation action BEFORE it is executed.",
        f"Action: {report['action']} | Intent: {report['intent']}",
        f"Task instruction: {report['task']}",
        "",
        "All metric quantities below were COMPUTED from point cloud geometry",
        "(units: cm, world frame, z-up). Do not estimate lengths from any image;",
        "judge only the numbers and their consistency.",
        "",
        "MEASUREMENTS:",
    ]
    for k, v in report["measurements"].items():
        lines.append(f"  - {k}: {v}")
    if "corridor_clearance_table" in report:
        lines += ["", "CORRIDOR CLEARANCE TABLE (one row per obstacle along the path):"]
        for row in report["corridor_clearance_table"]:
            lines.append(
                f"  - {row['obstacle']}: top_z={row['obstacle_top_z_cm']}cm, "
                f"xy_dist_to_corridor={row['xy_dist_to_corridor_cm']}cm, "
                f"intrudes={row['intrudes_corridor']}, "
                f"z_clearance={row['z_clearance_cm']}cm "
                f"(required {row['required_z_clearance_cm']}cm) -> "
                f"{'PASS' if row['pass'] else 'FAIL'}"
            )
    lines += ["", "RULE CHECKS (computed deterministically):"]
    for c in report["derived_checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        hard = " [hard]" if c["hard"] else ""
        lines.append(f"  - {c['name']}{hard}: {status} (rule: {c['rule']}, value: {c['value']})")
    lines += [
        "",
        f"CONTEXT: {report['context']}",
        "",
        "Your job:",
        "1. Check whether the numbers are mutually consistent and physically plausible",
        "   for this action (e.g. an overhang near zero is implausible when holding a bowl).",
        "2. Consider rule FAILs and borderline PASSes.",
        "3. Reply EXACTLY in this format:",
        "DECISION: approve | reject | uncertain",
        "REASONS: <one or two sentences>",
        "RISKS: <main risk if executed, or 'none'>",
    ]
    return "\n".join(lines)
