---
name: Plan Bounded Motion
category: motion
description: "Convert one selected affordance into a feasible, bounded motion candidate without moving the robot."
---

# Plan Bounded Motion

## Purpose

Check the selected pose or affordance against current robot geometry and produce a
bounded trajectory candidate for a separate Action Executor.

## When to use

Use after an affordance stage has selected a concrete grasp or placement pose and
before any world-changing action code is executed.

## When NOT to use

Do not use this skill to discover an affordance, move the robot, or repair an
infeasible pose by silently changing its contact geometry. Route those jobs to
perception, affordance, and action-execution specialists respectively.

## Workflow

1. Read only the supplied affordance and current robot state.
2. Check IK and basic approach/retreat feasibility.
3. Produce one preferred candidate plus compact alternatives when useful.
4. Save candidate geometry as typed artifacts and finish without commanding motion.

### Place affordances (required template)

When the affordance is a placement drop (has `desired_object_center` / `approach_position`
or comes from `find_placement`), call `build_place_trajectory(affordance)` and publish
that trajectory after IK checks. The template is fixed:

1. `transport_hover` — `approach_position` (or TCP + approach_height), gripper hold
2. `release_descend` — affordance `position` (TCP drop), gripper hold
3. `open_settle` — same TCP, gripper open
4. `retreat_up` — straight up, gripper open

Rules:

- `position` is the TCP. Never substitute `desired_object_center` as a motion target.
- Do not invent release z (no rim + 0.10 m rewrites).
- If IK fails on these poses, finish with `failure_kind: infeasible` and let the graph
  return to Affordance. Do not silently change the drop height.

### Grasp affordances (required template)

Call `build_grasp_trajectory(affordance)`. The canonical function accepts
`position`/`pos`, `quaternion_wxyz`/`quat`, optional `approach_position`, and
optional `lift_position`/`lift_pos`. It structurally enforces:

1. `approach` is above the grasp TCP, gripper open;
2. `pregrasp` and `grasp` use the exact selected TCP and quaternion;
3. `lift` is above the grasp TCP, gripper holding.

Never infer a signed displacement from `approach_dir_world`: axis conventions are
ambiguous and caused a live waypoint below the table. If an explicit approach or
lift point is not above the grasp TCP, finish `infeasible`.

### Optional cuRobo backend

Use cuRobo only when it is available and the canonical Cartesian path is repeatedly
infeasible or scene obstacles require collision-aware planning. Pass explicit
world-frame grasp candidates and the current target mask to
`plan_grasp_trajectory`; never ask it to discover an affordance. Convert a
successful `(T, 7)` result into `robomex.trajectory.v1` waypoints containing
`joints` only. A waypoint must never contain both `joints` and `position_xyz`.
If cuRobo is unavailable, return to the canonical Cartesian path; do not fail merely
because the optional backend is absent.

## Candidate Generation

Prefer the supplied candidate and derive only the approach, contact, retreat, and
optional transport waypoints needed to make it executable. For place, do not invent a
different waypoint set than `build_place_trajectory`.

## Local Checks

Check IK, finite values, workspace bounds, and approach direction. Keep alternatives
compact and ranked. For place, verify every waypoint uses the affordance TCP XY and
that release z equals `affordance.position[2]`.

## Failure Modes

If no candidate is feasible, report infeasibility and preserve the rejected candidate
reasons so the graph can return to affordance generation.

## Clean Reusable Rules

Motion planning computes a candidate; it never changes robot or environment state.

## Weak Priors

Short collision-free paths and modest clearances are usually preferable, but live
geometry and IK results dominate.

## Prohibited Shortcuts

- Do not call robot motion or gripper APIs.
- Do not repeat grounding or segmentation.
- Do not silently replace the supplied affordance with unrelated geometry.
- Do not treat `desired_object_center` as TCP.
- Report infeasibility explicitly so the graph can route back to Affordance.

## Artifacts to Save

- Selected trajectory and waypoint frames.
- Compact feasibility report and rejected-alternative reasons.

## Multimodal Evidence Contract

Consume only typed affordance geometry and current robot state. Preserve the
affordance observation epoch in the trajectory. Save the selected waypoint frames
and feasibility reasons; do not create new perceptual claims. If geometry is stale,
route `stale_observation`; if canonical poses fail structural/IK checks, route
`infeasible`.

## Optional Sidecars

`scripts/grasp_trajectory.py` and `scripts/place_trajectory.py` provide the two
canonical trajectory builders.

## Reference Code

`build_grasp_trajectory` and `build_place_trajectory` are contracted canonical
functions: they are already defined in your sandbox namespace when this skill
loads — call them directly, no import. Do
not probe it with `dir()` or `inspect`. Save artifacts under `ARTIFACTS_DIR`
(already defined in the sandbox), never under a CWD-relative path.

Ontology (this Franka / LIBERO env):

- Canonical top-down gripper quaternion is `[0, 1, 0, 0]` (π about x, wxyz).
- Identity `[1, 0, 0, 0]` is gripper skyward — never a grasp / place orientation.
- Prefer the quaternion already on the affordance (`INPUTS[...]["payload"]`); do not
  invent a new one.

```python
import json, os
import numpy as np

# Read affordance from INPUTS — do not retype position/quaternion literals.
affordance = INPUTS["affordance"]["payload"]
is_place = "desired_object_center" in affordance or "place_quat" in affordance
plan = (
    build_place_trajectory(affordance)
    if is_place
    else build_grasp_trajectory(affordance)
)
# plan keys: feasible, waypoints, note. Each waypoint dict:
# {"name", "phase", "position_xyz", "quaternion_wxyz", "gripper", "note"}
# with names: transport_hover -> release_descend -> open_settle -> retreat_up.

for wp in plan["waypoints"]:
    try:
        # return_info=True reports whether a silent orientation fallback was used.
        joints, info = solve_ik(
            np.asarray(wp["position_xyz"]),
            np.asarray(wp["quaternion_wxyz"]),
            return_info=True,
        )
        wp["ik_ok"] = info.get("orientation_used") == "requested"
        wp["orientation_used"] = info.get("orientation_used")
        if not wp["ik_ok"]:
            wp["ik_error"] = f"orientation fallback to {wp['orientation_used']}"
    except Exception as exc:
        wp["ik_ok"], wp["ik_error"] = False, str(exc)
plan["feasible"] = all(wp["ik_ok"] for wp in plan["waypoints"])

trajectory_path = os.path.join(ARTIFACTS_DIR, "trajectory.json")
with open(trajectory_path, "w") as f:
    json.dump(plan, f, indent=2)

NODE_RESULT = {
    "outputs": {
        "trajectory": {
            "payload": {
                "feasible": plan["feasible"],
                "waypoints": plan["waypoints"],
                "note": plan.get("note", ""),
            },
            "confidence": 0.9 if plan["feasible"] else 0.2,
            "frame": "world",
            "artifacts": {"trajectory_file": "trajectory.json"},
        }
    },
    "recommended_next": "execute" if plan["feasible"] else "replan_affordance",
    "failure_kind": "" if plan["feasible"] else "infeasible",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```

For a grasp affordance, never hand-author waypoint arithmetic. Use
`build_grasp_trajectory`, IK-check the returned Cartesian waypoints with
`return_info=True`, and stop at lift — release/home waypoints belong to a place
subgoal, not to a pick plan.
