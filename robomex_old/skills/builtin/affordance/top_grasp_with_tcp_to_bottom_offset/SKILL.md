---
name: Top Grasp With TCP To Bottom Offset
category: affordance
description: "Compute a top grasp and the TCP-to-object-bottom offset needed for later center-aware placement."
---

# Top Grasp With TCP To Bottom Offset

## Purpose

Compute a top grasp for upright or compact objects and record how far the grasp TCP is
from the object's bottom. This offset lets later placement reason about where the
object base or center will be when the gripper releases.

## When to use

Use for upright boxes, cans, bottles, cartons, and other objects where the highest
visible point and OBB center give a stable top grasp. Do not use for open bowls whose
center is empty; use a rim/container affordance instead.

## When NOT to use

Do not use for open rims, lying/elongated objects, multi-object masks, or any case
where a top-down pinch would contact empty space.

## Workflow

1. Segment the target and obtain world-frame points.
2. Estimate object top, bottom, OBB center, and horizontal center.
3. Choose a top-down grasp slightly below the highest point.
4. Compute `tcp_to_bottom = grasp_z - bottom_z`.
5. Publish grasp pose and `EVIDENCE["held_object_frame"]` fields for placement.

## Candidate Generation

- Use OBB center for x/y when the mask is clean.
- Use robust percentiles for top and bottom z when there is noisy depth.
- Include `tcp_to_bottom`, `object_center_at_grasp`, and
  `object_center_offset_from_grasp`.
- Return uncertainty if the point cloud is too sparse or height is implausible.

## Local Checks

- Confirm the object is upright or compact enough for top grasp.
- Avoid using a top grasp on hollow/open objects.
- Check that the target top is above the table and not from a background point.
- Check reachability before motion.

## Failure Modes

- Top point belongs to a distractor: re-segment or filter workspace.
- Placement later lands too high or too low: inspect `tcp_to_bottom` and release height.
- Object slips: switch to side or sampled grasp and recompute held-object frame.

## Clean Reusable Rules

- Grasp evidence should include placement-relevant geometry, not only a pose.
- TCP, object center, and object bottom are distinct physical references.
- Held-object frame evidence should be refreshed after a different grasp family.

## Weak Priors

- A few centimeters below the visible top is often a safe initial top-grasp depth.
- The OBB center is a useful x/y reference when segmentation is clean.

## Prohibited Shortcuts

- Do not reuse a fixed tcp-to-bottom offset across objects.
- Do not infer bottom height from object category alone.
- Do not use this skill for concave/open containers without checking rim geometry.

## Artifacts to Save

- Optional 3D plot with top, bottom, center, and selected grasp point.
- Compact held-object frame record for later release.

## Multimodal Evidence Contract

Consume fresh target points and optional geometry from one epoch. Publish the exact
grasp TCP/quaternion, explicit `approach_position` and `lift_position`, held-object
offsets, and an inspectable overlay. Never leave approach-axis sign for a downstream
LLM to infer.

## Optional Sidecars

`scripts/top_grasp_offset.py` contains a pure helper for computing the top grasp and
offset fields from segmented points and an optional OBB.

## Reference Code

The skill's `scripts/` directory is already on `sys.path`; import the module directly
and call the helper exactly as below — do not probe it with `dir()` or `inspect`.

```python
import numpy as np
from top_grasp_offset import compute_top_grasp_with_tcp_to_bottom_offset

points = np.load(INPUTS["object_points"]["refs"][0]["path"])  # arrays arrive as file refs
geometry = INPUTS.get("object_geometry", {}).get("payload", {})  # optional port
grasp = compute_top_grasp_with_tcp_to_bottom_offset(
    points,
    obb=geometry.get("obb"),      # or None
    z_grasp_offset=0.04,          # metres below the visible top
    lift_height=0.20,
)
if not grasp["ok"]:
    raise RuntimeError(grasp["reason"])  # e.g. "empty_points"

# Returned keys: ok, strategy, pos, quat (wxyz, top-down), lift_pos,
# top_z, bottom_z, tcp_to_bottom, object_center_at_grasp,
# object_center_offset_from_grasp, num_points.
# ALWAYS use grasp["quat"] — never invent [1,0,0,0] (identity = gripper skyward).
# All values are plain JSON-safe lists/floats — no .tolist() needed.
EVIDENCE["grasp_affordance"] = grasp
EVIDENCE["held_object_frame"] = {
    "object_center_at_grasp": grasp["object_center_at_grasp"],
    "object_center_offset_from_grasp": grasp["object_center_offset_from_grasp"],
    "tcp_to_bottom": grasp["tcp_to_bottom"],
    "top_z": grasp["top_z"],
    "bottom_z": grasp["bottom_z"],
}
NODE_RESULT = {
    "outputs": {
        "grasp_affordance": {
            "payload": {
                "position": grasp["pos"],
                "quaternion_wxyz": grasp["quat"],  # reference, do not retype
                "approach_dir_world": [0.0, 0.0, 1.0],
                "grasp_family": "top_down",
            },
            "confidence": 0.85,
            "frame": "world",
            "artifacts": {},
        },
        "held_object_frame": {
            "payload": EVIDENCE["held_object_frame"],
            "confidence": 0.85,
            "frame": "world",
            "artifacts": {},
        },
    },
    "recommended_next": "plan_motion",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
