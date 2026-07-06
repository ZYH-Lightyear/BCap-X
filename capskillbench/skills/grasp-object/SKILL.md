---
name: grasp-object
description: Use for pick, pick up, grasp, move, put, place, or any task requiring holding an object before placing it elsewhere.
---

# Grasp Object

Plan and execute a grasp for a segmented or localized object.

## APIs

```python
grasps_cam, scores = plan_grasps(mask, camera="main")
pose_world, score = select_top_down_grasp(grasps_cam, scores, cam["pose_mat"])
open_gripper()
goto_pose(position, quaternion_wxyz, approach_z=0.06)
close_gripper()
```

## Workflow

1. Use `$segmentation-to-points` to get clean world-frame object points.
2. Try learned grasp planning first:

```python
cam = get_camera("main")
grasps_cam, scores = plan_grasps(mask, camera="main")
grasp_world, score = select_top_down_grasp(grasps_cam, scores, cam["pose_mat"])
if grasp_world is not None:
    position, quat_wxyz = decompose_transform(grasp_world)
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
goto_pose(position, quat_wxyz, approach_z=0.06)
close_gripper()
lift = position.copy()
lift[2] += 0.08
goto_pose(lift, quat_wxyz)
```

5. Re-observe. If the gripper fully closes and the object did not move, re-segment and retry once.

## Pitfalls

- Always open the gripper before approach.
- `plan_grasps` returns camera-frame poses.
- `goto_pose` expects world-frame position and WXYZ quaternion.
- Do not use stale points after failed contact.

## Related Skills

Use `$language-grounding` and `$segmentation-to-points` before grasping, `$motion-control` for execution, and `$place-and-release` after the object is held.
