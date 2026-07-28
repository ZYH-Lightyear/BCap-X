---
name: Compute OBB Short Axis Grasp
category: affordance
description: "Compute a top-down grasp pose aligned to the short horizontal axis of an object's oriented bounding box."
---

# Compute OBB Short Axis Grasp

## Purpose

Produce a grasp affordance for thin, flat, or elongated objects by fitting an oriented
bounding box and aligning the gripper across the object's short horizontal axis. This
skill does not move the robot.

## When to use

Use after segmenting an object whose footprint orientation matters: cards, packages,
books, flat blocks, low containers, and elongated tabletop items. Avoid it when the
object is concave, heavily occluded, or better handled by an open-container rim grasp.

## When NOT to use

Do not use for degenerate/noisy OBBs, open rims, severe occlusion, or a target whose
short axis is not a meaningful finger-opening direction.

## Workflow

1. Read segmented world-frame points from local grounding evidence.
2. Fit an oriented bounding box and identify the vertical OBB axis.
3. Among the two horizontal axes, select the shorter extent as the finger-opening axis.
4. Build a top-down end-effector orientation whose x-axis follows that short axis.
5. Choose a grasp depth below the top surface and publish `EVIDENCE["grasp_affordance"]`.

## Candidate Generation

- Generate at least one top-down candidate at the OBB center.
- For thin objects, try shallow depth below the top surface.
- For thicker objects, use a configurable fraction down from the top.
- Return the OBB center, extent, selected axis, grasp pose, lift pose, and confidence.

## Local Checks

- OBB axes are hints; inspect the image if the selected axis contradicts visible shape.
- Reject candidates if the mask includes table or neighboring objects.
- Check IK before execution when possible.
- If the object is open or bowl-like, switch to a rim/container affordance.

## Failure Modes

- Unstable OBB due to partial mask: re-segment with a tighter prompt or crop.
- Grasp line crosses the long axis: swap the selected horizontal axis.
- Candidate too high or too low: adjust depth from top and verify with a visual overlay.
- Object slips after lift: use the bounded-offset `failed_grasp` recovery in
  `pick_object`, change one contact hypothesis, and verify.

## Clean Reusable Rules

- For flat/thin objects, alignment matters as much as center.
- OBB-derived grasp is an affordance; motion skill still owns execution and lift check.
- Save compact geometry evidence rather than reusing raw point clouds across turns.

## Weak Priors

- Top-down quaternion is usually safe for tabletop flat objects.
- A short-axis pinch is usually better than pinching along the long object axis.

## Prohibited Shortcuts

- Do not hardcode object dimensions or table coordinates.
- Do not trust OBB when the mask visibly includes background.
- Do not execute before checking reachability for unusual poses.

## Artifacts to Save

- Optional overlay or 3D plot showing OBB axes, chosen short axis, and selected grasp.
- Compact record with point count, OBB extent, chosen axis, grasp position, and
  confidence.

## Multimodal Evidence Contract

Consume same-epoch points and OBB evidence. Publish the selected axis, exact grasp
TCP/quaternion, explicit approach/lift positions, and an overlay showing jaw/axis
alignment. Degenerate axes emit `infeasible` rather than a default scene-specific
yaw.

## Optional Sidecars

`scripts/obb_short_axis.py` contains a pure helper for converting points and an OBB dict
into a grasp affordance. Act may use environment OBB utilities, then call this helper to
standardize output keys.

## Reference Code

The skill's `scripts/` directory is already on `sys.path`; call the helper exactly as
below — do not probe it with `dir()` or `inspect`.

```python
import numpy as np
from obb_short_axis import compute_obb_short_axis_grasp

points = np.load(INPUTS["object_points"]["refs"][0]["path"])  # arrays arrive as file refs
g = INPUTS["object_geometry"]["payload"]["obb"]  # small values stay inline in payload
obb = {"center": g["center"], "extent": g["extent"], "R": g["rotation_matrix"]}
grasp = compute_obb_short_axis_grasp(
    points,
    obb,
    grasp_depth_fraction_from_top=0.5,
    lift_height=0.15,
)

# Returned keys: ok, strategy, pos, quat (wxyz), lift_pos, obb_center,
# obb_extent, vertical_axis_index, short_axis_index, short_axis, num_points.
# Use grasp["quat"] — never invent [1,0,0,0] (identity = gripper skyward).
# All values are plain JSON-safe lists/floats — no .tolist() needed.
EVIDENCE["grasp_affordance"] = grasp
NODE_RESULT = {
    "outputs": {
        "grasp_affordance": {
            "payload": {
                "position": grasp["pos"],
                "quaternion_wxyz": grasp["quat"],
                "grasp_family": "obb_short_axis",
            },
            "confidence": 0.8,
            "frame": "world",
            "artifacts": {},
        }
    },
    "recommended_next": "plan_motion",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
