---
name: libero-debug-and-recovery
description: Use when a CaP-X LIBERO run has stderr, traceback, timeout, IK failure, SAM3 empty mask, wrong object grounding, tiny or noisy point cloud, grasp miss, fully closed gripper after grasp, no visible scene change, or repeated failed pick, place, drawer, stove, microwave, or push action.
---

# LIBERO Debug And Recovery

Diagnose failures from the top of the perception-control stack downward. Change one layer at a time.

## Recovery Order

1. **Observation**: call `get_observation()` and verify the scene changed as expected.
2. **Grounding**: if the wrong object or region was chosen, make the language query more specific.
3. **Segmentation**: if SAM3 returns nothing or the mask is wrong, switch text prompt to point prompt or re-ground.
4. **Points**: if point count is tiny/noisy, check depth, rescale pixels, and run `filter_noise`.
5. **Geometry**: if pose is bad, inspect center, top z, OBB extent, and quaternion WXYZ order.
6. **Grasp**: if the gripper fully closes or object does not lift, retry with fresh points and a more central or lower grasp.
7. **Motion**: if IK fails, raise target, use top-down orientation, go home, or split into waypoints.
8. **Completion**: after any successful action, re-observe before finishing.

## Common Fixes

- **Malformed VLM output**: re-ask for "ONLY JSON" with a smaller schema.
- **0-1000 normalized box/point bug**: common with Qwen-family VLMs. Convert normalized coordinates to real pixels before SAM3 or `pixel_to_world_point`:

```python
h, w = rgb.shape[:2]
x_px = int(round(x_norm / 1000 * (w - 1)))
y_px = int(round(y_norm / 1000 * (h - 1)))
```

  GPT/OpenAI-style models may obey explicit pixel-coordinate prompts, so do not rescale blindly; compare returned values with `w` and `h`.
- **SAM3 empty result**: nudge the point toward the box center or try text prompt segmentation.
- **Wrong similar object**: include color, label, relative position, and nearby reference objects.
- **IK failure**: use `np.array([0.0, 1.0, 0.0, 0.0])`, increase z, or call `goto_home_joint_position()`.
- **Scene unchanged**: print intermediate targets, lower speed by using shorter waypoints, and re-run perception.
- **Put both tasks**: do not assume the first placement succeeded; verify before moving to the second object.

## Minimal Debug Prints

```python
print("target point count", len(points))
print("target center", center)
print("release", release)
print("robot", get_observation()["robot_cartesian_pos"])
```

Keep debug output short and actionable; the next code turn will receive stdout and stderr.

## Related Skills

Return to `$libero-scene-observation`, `$libero-language-grounding`, `$libero-segmentation-to-points`, `$libero-grasp-object`, `$libero-motion-control`, or `$libero-place-and-release` depending on the failing layer.
