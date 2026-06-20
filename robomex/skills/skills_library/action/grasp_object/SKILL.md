---
name: Grasp Object
category: action
description: Approach a vetted grasp pose from above, close the gripper, and confirm the object is actually held by lifting slightly.
---

# Grasp Object

Execute a grasp at a vetted, IK-feasible grasp pose (see the grasp-candidates skill)
and confirm the hold before doing anything else.

## When to use

Once a feasible `grasp_pos` / `grasp_quat` is in hand and the gripper is empty.

## Procedure

```python
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
- Do not move toward the place target until the lift check passed.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: gripper width after closing is above zero and the object moved up with the lift.

## Failure modes

- Gripper fully closed (missed): `open_gripper`, re-segment the object, get fresh
  grasp candidates, retry once.
- Object slipped during lift: retry with a candidate whose approach is more
  vertical, or grasp closer to the object center.
