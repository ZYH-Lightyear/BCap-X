---
name: scene-observation
description: Use when a CaP-X SkillBench task needs current camera images, depth, robot pose, gripper state, or a before/after check after an action.
---

# Scene Observation

Use `get_observation()` before planning, after every meaningful action, and when deciding whether to retry or finish.

## API

```python
obs = get_observation()
cam = get_camera("main", obs)
wrist = get_camera("wrist", obs)  # when available
caps = capabilities()
```

`obs["cameras"]` maps semantic camera names to normalized camera dicts:

- `cam["rgb"]`: RGB image, `(H, W, 3)`, `uint8`.
- `cam["depth"]`: depth image, `(H, W)`, meters.
- `cam["intrinsics"]`: camera intrinsics, `(3, 3)`, when available.
- `cam["pose_mat"]`: camera-to-world transform, `(4, 4)`, when available.
- `obs["robot"]["arms"]["default"]`: normalized end-effector state when available.
- `obs["raw"]`: backend-specific observation for debugging only.

## Workflow

1. Use `get_camera("main")` for the global scene layout.
2. Use `get_camera("wrist")` when the gripper or held object blocks the main view.
3. Re-observe after `goto_pose`, `move_to_joints`, `open_gripper`, or `close_gripper`.
4. Print only compact debug facts: available cameras, target point count, chosen pose, and gripper value.

## Pitfalls

- Do not rely on backend camera names such as `agentview` or `robot0_robotview`.
- Do not reuse old RGB/depth/masks after the robot or objects move.
- Quaternions in SkillBench motion APIs are WXYZ.

## Related Skills

Use `$language-grounding` to identify targets, `$segmentation-to-points` to convert observations into geometry, and `$debug-and-recovery` when observations do not change after an action.
