---
name: Find Placement
category: affordance
description: "Compute a GaP-style drop affordance: object center in the target zone and an executable TCP pose."
---

# Find Placement

## Purpose

Produce a placement affordance for a held object. The typed affordance `position` is the
executable TCP drop target. `desired_object_center` is evidence for where the object
center should land. This skill does not execute motion.

## When to use

Use when the robot is already holding an object and the task asks to put it in, on, or
relative to a target. Read `EVIDENCE["held_object_frame"]` when present so TCP
compensation can use grasp offset / `tcp_to_bottom`.

## When NOT to use

Do not use unless the object is believed held, substitute a visual target center
for an executable TCP, or execute motion from this read-only affordance skill.

## Workflow

1. Ground the placement target using perception skills and obtain target points.
2. Decide the target mode:
   - `open_container`: basket, bin, cup, bowl, or any target with usable interior space.
   - `support_surface`: plate, tray, tabletop, cabinet top, stacking target, or object
     support surface. A bowl on a plate is a support-surface placement case.
3. Call `compute_drop_affordance(...)` (or `estimate_placement_affordance`, which wraps
   it). Pass `held_object_frame` and, when known, `grasp_ee_z`.
4. Publish the returned affordance: `position` / `quaternion_wxyz` are the TCP drop pose
   for `robomex.affordance.v1`. Keep `desired_object_center`, `approach_position`, and
   zone fields in `EVIDENCE["placement_affordance"]`.
5. Save a 2D overlay and, when useful, a 3D visualization showing both the desired
   object center and the TCP drop pose.

## Candidate Generation

- For open containers, put the object center *inside* the cavity: zone floor is near
  the basket bottom (+ lip), zone ceiling is the rim. Never default to rim + 0.10 m.
- For support surfaces, keep a small clearance above the live top surface.
- Prefer free-space / opening center in XY; rim/wall points bias the raw mask centroid.
- If the held object was grasped off-center, compute TCP from
  `desired_object_center - object_center_offset_from_grasp` (and `ee_to_obj_z` when
  `grasp_ee_z` is available). Do not leave TCP equal to the object center by default.

## Local Checks

- Confirm the selected point projects over the intended target, not a neighboring
  object or container wall.
- For open containers, `desired_object_center.z` must be `<= rim_top`.
- `position` (TCP) must be finite and distinct from `desired_object_center` when an
  offset is available.
- If target points are sparse or biased, return uncertainty instead of inventing a
  precise center.

## Failure Modes

- Wrong target: re-ground with a more specific target description.
- Object lands off-center after an edge/rim grasp: check whether TCP used the
  held-object center offset.
- Object bounces or topples: lower the release clearance and retry only after a state
  check.
- Target interior not visible: use a conservative support/opening estimate and mark
  uncertainty.

## Clean Reusable Rules

- Typed affordance `position` is TCP. Object-center state lives in
  `desired_object_center`.
- Open-container placement and support-surface placement are different zone models.
- Post-release checks should inspect the object-target relation, not just gripper
  opening.

## Weak Priors

- Prior release heights may seed a candidate, but live depth/mask geometry and held
  frame dominate.
- Known grasp offsets guide TCP compensation when the current held-object frame is
  partial.

## Prohibited Shortcuts

- Do not hardcode target coordinates, pixels, seed layouts, or object-name answer
  tables.
- Do not publish rim + 0.10 m as the default open-container release height.
- Do not treat `desired_object_center` as the motion target.
- Do not use the target mask centroid blindly when the visible surface is biased.

## Artifacts to Save

- Placement point overlay on the target image.
- Optional 3D visualization showing target points, desired object center, and TCP
  drop pose.

## Multimodal Evidence Contract

Consume fresh target points plus the typed held-object frame. Publish
`desired_object_center`, executable `position`, approach pose, target relation,
and an overlay showing the target zone and compensated TCP. Preserve the grounding
epoch; route `wrong_grounding` when target identity is ambiguous and `infeasible`
when finite compensation cannot be constructed.

## Optional Sidecars

`scripts/placement_affordance.py` provides `estimate_container_zone`,
`compute_drop_affordance`, overlay helpers, and 3D visualization. Always prefer
`compute_drop_affordance` over hand-rolled height heuristics.

## Reference Code

The skill's `scripts/` directory is already on `sys.path`; call the helper exactly as
below — do not probe it with `dir()` or `inspect`.

```python
import numpy as np
from placement_affordance import (
    compute_drop_affordance,
    save_placement_overlay,
    save_placement_3d_visualization,
)

# Target (container/surface) points arrive as a file ref:
points = np.load(INPUTS["target_points"]["refs"][0]["path"])
affordance = compute_drop_affordance(
    points,
    target_name=target_name,
    evidence=EVIDENCE,                      # also stores EVIDENCE["placement_affordance"]
    held_object_frame=EVIDENCE.get("held_object_frame"),
    grasp_ee_z=grasp_z,                     # grasp TCP z at grasp time, or None
    drop_clearance=0.05,
    approach_height=0.20,
)

# Returned keys: position (executable TCP drop target), quaternion_wxyz,
# approach_position, approach_height, desired_object_center (evidence only,
# never a motion target), mode, zone_floor, zone_ceiling, rim_top,
# object_half_height, tcp_compensated, strategy, note.
# Prefer affordance["quaternion_wxyz"] — never invent [1,0,0,0].
# All values are plain JSON-safe lists/floats — no .tolist() needed.

save_placement_3d_visualization(points, affordance, ARTIFACTS_DIR)
NODE_RESULT = {
    "outputs": {
        "placement_affordance": {
            "payload": affordance,
            "confidence": 0.85,
            "frame": "world",
            "artifacts": {},
        }
    },
    "recommended_next": "plan_motion",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
