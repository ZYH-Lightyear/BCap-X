---
name: motion-control
description: Use when moving an arm, solving IK, sending joints, using goto_pose, resetting home, opening or closing the gripper, or recovering from IK failure.
---

# Motion Control

Use these APIs after perception and geometry have produced a target pose.

## APIs

```python
joints = solve_ik(position, quaternion_wxyz)
move_to_joints(joints)
goto_pose(position, quaternion_wxyz, approach_z=0.0)
home()
open_gripper()
close_gripper()
```

Pass `arm="left"`, `arm="right"`, `arm="arm0"`, or `arm="arm1"` only when the current environment exposes multiple arms and the task explicitly needs one. Single-arm tasks should omit `arm`.

## Workflow

```python
quat = np.array([0.0, 1.0, 0.0, 0.0])
goto_pose(target_pos, quat, approach_z=0.06)
```

For release:

```python
goto_pose(release_pos, quat, approach_z=0.05)
open_gripper()
```

For contact actions, move in short segments:

```python
for p in waypoints:
    goto_pose(p, quat)
```

## Pitfalls

- All quaternions are WXYZ.
- `approach_z` is a world +Z approach offset before descending to target.
- IK failures often improve by raising the target, using top-down orientation, calling `home()`, or splitting motion into shorter waypoints.
- Re-observe after gripper commands and long motions.

## Related Skills

Use `$geometry-and-frames` to produce valid target poses, `$grasp-object` for grasp execution, and `$debug-and-recovery` after IK or scene-stagnation failures.
