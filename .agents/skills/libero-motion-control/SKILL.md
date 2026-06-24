---
name: libero-motion-control
description: Use for CaP-X LIBERO robot execution when moving the Franka arm, solving IK, sending joints, using goto_pose, resetting home, opening or closing the gripper, recovering from IK failure, or timing approach and release motions.
---

# LIBERO Motion Control

Use these APIs to move the robot reliably after perception and geometry have produced a target pose.

## APIs

```python
joints = solve_ik(position, quaternion_wxyz)
move_to_joints(joints)
goto_pose(position, quaternion_wxyz, z_approach=0.0)
goto_home_joint_position()
open_gripper()
close_gripper()
```

## Choosing A Motion API

- Prefer `goto_pose(position, quat, z_approach=...)` for normal pick and place motions.
- Use `solve_ik + move_to_joints` when you need to inspect or adjust joint targets explicitly.
- Use `goto_home_joint_position()` before retrying from a tangled pose or after severe occlusion.

## Workflow

```python
quat = np.array([0.0, 1.0, 0.0, 0.0])  # WXYZ top-down default
goto_pose(target_pos, quat, z_approach=0.06)
```

For release:

```python
goto_pose(release_pos, quat, z_approach=0.05)
open_gripper()
```

For contact actions, move in short segments:

```python
for p in waypoints:
    goto_pose(p, quat)
```

## Pitfalls

- All quaternions are WXYZ.
- `z_approach` already performs the approach and descent; do not repeat the exact same `goto_pose` unless needed.
- IK failures often improve by raising the target, using top-down orientation, going home, or splitting motion into shorter waypoints.
- Always re-observe after gripper commands and after long motions.

## Related Skills

Use `$libero-geometry-and-frames` to produce valid target poses, `$libero-grasp-object` for grasp execution, and `$libero-debug-and-recovery` after IK or scene-stagnation failures.
