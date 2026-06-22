---
name: Grasp Object
category: action
description: Derive a simple top-down grasp from an object's segmented 3D points, execute it, and confirm the hold by lifting slightly.
---

# Grasp Object

Grasp a named object that has already been segmented into filtered 3D world `points`
(see the segment-object skill). This skill is self-contained: it derives a simple
top-down grasp from those points — no separate grasp-candidate generator is required.

## When to use

The gripper is empty and you have the object's filtered 3D `points` (from segmentation).

## Procedure

```python
import numpy as np

# Derive a top-down grasp from the segmented points: center over the object's
# (x, y) and grasp near its top surface, gripper pointing straight down.
center = points.mean(axis=0)
top_z = points[:, 2].max()
grasp_pos = np.array([center[0], center[1], top_z])
grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])   # top-down; adapt to the env convention

open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.075)   # approach from above, then descend
close_gripper()

# Confirm the hold with a small lift before doing anything else:
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.10
goto_pose(lift_pos, grasp_quat)

obs = get_observation()
gripper_width = obs["robot_cartesian_pos"][-1]        # 0 = fully closed, 1 = fully open
assert gripper_width > 0.05, "gripper fully closed -> nothing was grasped"
```

## Rules

- Keep `z_approach` modest (~0.075 m): `move_to_joints` interpolation is rudimentary,
  large offsets drift.
- Always `open_gripper` before approaching; a half-closed gripper hits the object.
- The post-close width check is mandatory — it is the cheapest grasp-failure detector.
- The top-down pose above is a sensible default; if a grasp-pose API is available in
  the namespace, prefer it, and adapt `grasp_quat` to the env's orientation convention.

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: gripper width after closing is above zero and the object moved up with the lift.

## Failure modes

- Gripper fully closed (missed): `open_gripper`, re-segment the object for fresh
  points, retry once (e.g. grasp a little lower than `top_z`).
- Object slipped during lift: grasp closer to the object center, or lower toward its
  mid-height instead of the very top.
