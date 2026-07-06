---
name: debug-and-recovery
description: Use when a run has stderr, traceback, timeout, IK failure, empty mask, wrong grounding, noisy point cloud, grasp miss, no visible scene change, or repeated failed action.
---

# Debug And Recovery

Diagnose failures from the top of the perception-control stack downward. Change one layer at a time.

## Recovery Order

1. Observation: call `get_observation()` and verify the scene changed.
2. Grounding: if the wrong object or region was chosen, make the query more specific.
3. Segmentation: if masks are empty or wrong, switch text prompt to point prompt or re-ground.
4. Points: if point count is tiny/noisy, check depth, rescale pixels, and run `filter_noise`.
5. Geometry: if pose is bad, inspect center, top z, OBB extent, and WXYZ order.
6. Grasp: if the gripper fully closes or object does not lift, retry with fresh points and a more central or lower grasp.
7. Motion: if IK fails, raise target, use top-down orientation, call `home()`, or split into waypoints.
8. Completion: after any action, re-observe before finishing.

## Common Fixes

- Malformed VLM output: re-ask for "ONLY JSON" with a smaller schema.
- Normalized point bug: convert `0..1000` coordinates to real pixels before segmentation or `pixel_to_world_point`.
- Empty segmentation: nudge the point toward the box center or try text segmentation.
- Wrong similar object: include color, label, relative position, and nearby reference objects.
- IK failure: use `np.array([0.0, 1.0, 0.0, 0.0])`, increase z, call `home()`, or use shorter waypoints.
- Scene unchanged: print intermediate targets, use shorter waypoints, and re-run perception.

## Minimal Debug Prints

```python
print("cameras", sorted(get_observation()["cameras"]))
print("target point count", len(points))
print("target center", center)
print("release", release)
print("capabilities", capabilities())
```

Keep debug output short and actionable.

## Related Skills

Return to `$scene-observation`, `$language-grounding`, `$segmentation-to-points`, `$grasp-object`, `$motion-control`, or `$place-and-release` depending on the failing layer.
