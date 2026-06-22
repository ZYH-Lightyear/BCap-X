---
name: Release At
category: action
description: Release a held object at a given 3D place position by moving above it, lowering with clearance, opening the gripper, and retreating.
---

# Release At

Release the object currently held in the gripper at a known 3D `place_pos` (see the
find-placement skill). This skill is self-contained: it moves over the release point,
opens the gripper, and retreats so the object settles.

## When to use

The gripper is holding an object and you have a 3D `place_pos` above the placement target
(from find_placement).

## Procedure

```python
import numpy as np

place_quat = np.array([0.0, 1.0, 0.0, 0.0])   # top-down; adapt to the env convention

# Approach the release point from above (modest approach offset), then release.
goto_pose(place_pos, place_quat, z_approach=0.075)
open_gripper()

obs = get_observation()
gripper_width = obs["robot_cartesian_pos"][-1]   # 0 = closed, 1 = open
assert gripper_width > 0.5, "gripper did not open -> object not released"

# Retreat straight up so the object stays where it was dropped.
retreat = place_pos.copy()
retreat[2] += 0.10
goto_pose(retreat, place_quat)
```

## Rules

- Release ABOVE the target with clearance; releasing at the exact contact height can
  topple a container or wedge the object. `find_placement` already adds clearance to
  `place_pos` — keep it.
- Keep `z_approach` modest (~0.075 m): `move_to_joints` interpolation is rudimentary,
  large offsets drift.
- The post-open width check is mandatory — it is the cheapest release-failure detector
  (a still-low width means the object is still held).
- Retreat upward before doing anything else, so the placed object is not knocked.

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: gripper width is high (open) after the release and the object stayed at the target.

## Failure modes

- Gripper width stayed low (object still held): re-issue `open_gripper`, confirm width,
  then retreat.
- Object bounced out / target toppled: lower `place_pos[2]` toward the target surface
  (less clearance) via find_placement and retry once.
- Object placed but off-target: re-run find_placement for a fresher release point.
