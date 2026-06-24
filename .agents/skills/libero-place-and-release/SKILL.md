---
name: libero-place-and-release
description: Use in CaP-X LIBERO tasks that say place, put, put in, put on, put to the left or right of, put both objects, basket, plate, stove, cabinet, drawer, rack, caddy compartment, or release a held object at a target region.
---

# LIBERO Place And Release

Place a held object into or onto a target region after grasping.

## APIs

This skill composes `get_observation`, `point_prompt_molmo`, `query_vlm`, `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, `mask_to_world_points`, `get_oriented_bounding_box_from_3d_points`, `goto_pose`, and `open_gripper`.

## Workflow

1. Confirm the object is held with a fresh observation.
2. Ground the target receptacle or surface: basket, plate, stove, cabinet top, drawer interior, rack, or caddy compartment.
3. Segment the target if it is visible and convert its mask to world points.
4. Estimate placement center:

```python
obb = get_oriented_bounding_box_from_3d_points(target_points)
center = obb["center"]
release = center.copy()
release[2] = target_points[:, 2].max() + 0.08
quat = np.array([0.0, 1.0, 0.0, 0.0])
```

5. Apply relation offsets when the language asks for `left`, `right`, `front`, `back`, or a compartment.
6. Move above the target, release, and lift away:

```python
goto_pose(release, quat, z_approach=0.06)
open_gripper()
retreat = release.copy()
retreat[2] += 0.08
goto_pose(retreat, quat)
```

7. Re-observe to decide if the task is complete or if a second object must be placed.

## Placement Heuristics

- `in basket` or `in tray`: release near the interior center, above the rim.
- `on plate` or `on stove`: release over the top surface center.
- `to the left/right of plate`: use a lateral offset from the plate center, not the object center.
- `put both`: complete one object fully, re-observe, then repeat for the second.

## Pitfalls

- Do not release too low; collisions can knock the target away.
- Do not use the source object's old points as the placement target.
- Recompute the target after opening drawers or moving objects.

## Related Skills

Use `$libero-grasp-object` before placing, `$libero-language-grounding` for target regions, `$libero-segmentation-to-points` for target geometry, and `$libero-motion-control` for execution.
