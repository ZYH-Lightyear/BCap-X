---
name: place-and-release
description: Use when placing a held object into or onto a target region, or next to/left/right/front/back of another object.
---

# Place And Release

Place a held object into or onto a target region after grasping.

## APIs

This skill composes `get_camera`, `ground_point`, `segment_object`, `mask_to_world_points`, `get_oriented_bounding_box`, `goto_pose`, and `open_gripper`.

## Workflow

1. Confirm the object is held with a fresh observation.
2. Ground the target receptacle or surface.
3. Segment the target if visible and convert the mask to world points.
4. Estimate placement center:

```python
obb = get_oriented_bounding_box(target_points)
center = obb["center"]
release = center.copy()
release[2] = target_points[:, 2].max() + 0.08
quat = np.array([0.0, 1.0, 0.0, 0.0])
```

5. Apply relation offsets for left, right, front, back, or compartments.
6. Move above the target, release, and lift away:

```python
goto_pose(release, quat, approach_z=0.06)
open_gripper()
retreat = release.copy()
retreat[2] += 0.08
goto_pose(retreat, quat)
```

7. Re-observe before deciding completion or manipulating a second object.

## Pitfalls

- Do not release too low.
- Do not use the source object's old points as the placement target.
- Recompute the target after opening drawers or moving objects.

## Related Skills

Use `$grasp-object` before placing, `$language-grounding` for target regions, `$segmentation-to-points` for target geometry, and `$motion-control` for execution.
