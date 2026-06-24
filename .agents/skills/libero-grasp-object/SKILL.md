---
name: libero-grasp-object
description: Use in CaP-X LIBERO tasks that say pick, pick up, grasp, move, put, place, put both, put in basket, put on plate, or require holding an object before placing it elsewhere.
---

# LIBERO Grasp Object

Plan and execute a grasp for a segmented or localized object.

## APIs

```python
grasp_poses, grasp_scores = plan_grasp(points)
grasp_poses, grasp_scores = plan_grasp_from_point_clouds(point_clouds)
pose = select_top_down_grasp(grasp_poses, grasp_scores)
open_gripper()
goto_pose(position, quaternion_wxyz, z_approach=0.06)
close_gripper()
obs = get_observation()
```

## Workflow

1. Use `$libero-segmentation-to-points` to get clean world-frame object points.
2. Try learned grasp planning first:

```python
grasp_poses, grasp_scores = plan_grasp(points)
grasp_pose = select_top_down_grasp(grasp_poses, grasp_scores)
position, quat_wxyz = decompose_transform(grasp_pose)
```

3. If grasp planning fails, use a top-down fallback from the point cloud:

```python
center = points.mean(axis=0)
top_z = points[:, 2].max()
position = np.array([center[0], center[1], top_z])
quat_wxyz = np.array([0.0, 1.0, 0.0, 0.0])
```

4. Execute approach, close, and lift:

```python
open_gripper()
goto_pose(position, quat_wxyz, z_approach=0.06)
close_gripper()
lift = position.copy()
lift[2] += 0.08
goto_pose(lift, quat_wxyz)
```

5. Re-observe. If the gripper is fully closed and the object did not move, re-segment and retry once with a slightly lower or more central grasp.

## Pitfalls

- Always open the gripper before approach.
- Do not use stale points after a failed contact.
- Avoid large `z_approach`; modest offsets reduce IK drift.
- For thin or flat objects, learned grasp candidates may beat a simple top-down center grasp.

## Related Skills

Use `$libero-language-grounding` and `$libero-segmentation-to-points` before grasping, `$libero-motion-control` for execution, and `$libero-place-and-release` after the object is held.
