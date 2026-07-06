---
name: geometry-and-frames
description: Use when code needs object centers, extents, oriented bounding boxes, transforms, WXYZ quaternions, target poses, offsets, directions, or waypoints.
---

# Geometry And Frames

Use this skill after segmentation or grounding when converting geometry into robot poses or contact paths.

## APIs

```python
obb = get_oriented_bounding_box(points)
quat = rotation_matrix_to_quaternion(R)
pos, quat = decompose_transform(T)
points2 = transform_points(points, transform_matrix)
waypoints = interpolate_segment(p1, p2, step=0.03)
direction = normalize_vector(v)
```

## Workflow

1. Use `get_oriented_bounding_box(points)` for centers and extents.
2. Use WXYZ quaternions for all SkillBench motion APIs.
3. For simple tabletop pick/place, a top-down quaternion is often enough:

```python
top_down = np.array([0.0, 1.0, 0.0, 0.0])
```

4. For pushing or pulling, normalize the direction and generate short waypoints.

## Frame Rules

- `mask_to_world_points` returns world-frame points.
- `depth_to_point_cloud` returns camera-frame points.
- `plan_grasps` returns camera-frame grasp poses.
- `select_top_down_grasp` expects camera-frame grasps plus `cam["pose_mat"]`.

## Pitfalls

- Do not use an OBB orientation blindly for gripper orientation.
- For placement, use the target surface center and a conservative release height.
- Very noisy or sparse points make OBBs unreliable; return to `$segmentation-to-points`.

## Related Skills

Use `$segmentation-to-points` before geometry, `$motion-control` to execute poses, and `$articulated-and-contact-actions` for push or pull paths.
