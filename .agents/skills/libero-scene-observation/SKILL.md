---
name: libero-scene-observation
description: Use when solving CaP-X LIBERO or LIBERO-Pro tasks that require reading the current scene, camera images, depth, robot pose, gripper state, or checking whether a pick, place, drawer, stove, microwave, basket, plate, cabinet, or caddy action changed the environment.
---

# LIBERO Scene Observation

Use `get_observation()` before planning, after every meaningful action, and when deciding whether to regenerate code or finish.

## API

```python
obs = get_observation()
```

`obs` contains:

- `obs["agentview"]["images"]["rgb"]`: RGB image, `(H, W, 3)`, `uint8`.
- `obs["agentview"]["images"]["depth"]`: depth image, `(H, W)`, meters.
- `obs["agentview"]["intrinsics"]`: camera intrinsics, `(3, 3)`.
- `obs["agentview"]["pose_mat"]`: camera-to-world transform, `(4, 4)`.
- `obs["robot0_eye_in_hand"]`: wrist camera with the same `images`, `intrinsics`, and `pose_mat` structure.
- `obs["robot_cartesian_pos"]`: end-effector pose `[x, y, z, qw, qx, qy, qz, gripper]`.
- `obs["robot_joint_pos"]`: 7 arm joints plus gripper state.

## Workflow

1. Read `agentview` first for tabletop layout and receptacles.
2. Use the wrist camera when the gripper or held object blocks the main camera.
3. After `goto_pose`, `move_to_joints`, `open_gripper`, or `close_gripper`, call `get_observation()` again before making geometric decisions.
4. For grasp checks, inspect the final gripper value in `robot_cartesian_pos`; a fully closed gripper after a grasp often means the object was missed.

## Pitfalls

- Do not reuse old RGB/depth/masks after the robot or objects move.
- Depth is already squeezed to `(H, W)` in the LIBERO reduced API.
- Quaternions in robot state and control APIs are WXYZ, not XYZW.

## Related Skills

Use `$libero-language-grounding` to identify objects in images, `$libero-segmentation-to-points` to convert observations into 3D points, and `$libero-debug-and-recovery` when observations do not change after an attempted action.
