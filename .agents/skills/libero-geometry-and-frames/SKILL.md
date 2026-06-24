---
name: libero-geometry-and-frames
description: Use for CaP-X LIBERO manipulation when code needs object centers, extents, oriented bounding boxes, world-camera transforms, WXYZ quaternions, target poses, relative offsets, push or pull directions, interpolated waypoints, or point cloud transforms.
---

# LIBERO Geometry And Frames

Use this skill after segmentation or grounding when converting geometry into robot poses or contact paths.

## APIs

```python
obb = get_oriented_bounding_box_from_3d_points(points)
quat = rotation_matrix_to_quaternion(R)
pos, quat = decompose_transform(T)
points2 = transform_points(points, transform_matrix)
waypoints = interpolate_segment(p1, p2, step=0.03)
direction = normalize_vector(v)
```

## Workflow

1. Use `get_oriented_bounding_box_from_3d_points(points)` for object or receptacle center:

```python
obb = get_oriented_bounding_box_from_3d_points(points)
center = obb["center"]
extent = obb["extent"]
R = obb["R"]
```

2. Use WXYZ quaternions for all LIBERO motion APIs:

```python
quat_wxyz = rotation_matrix_to_quaternion(R)
```

3. For simple tabletop pick/place, a top-down quaternion is often enough:

```python
top_down = np.array([0.0, 1.0, 0.0, 0.0])
```

4. For pushing or pulling, normalize the direction and generate short waypoints:

```python
direction = normalize_vector(target - start)
waypoints = interpolate_segment(start, start + 0.08 * direction, step=0.02)
```

## Frame Rules

- `mask_to_world_points` returns world-frame points.
- `depth_to_point_cloud` returns camera-frame points.
- `transform_points` can move points between frames if you provide the correct 4x4 transform.
- Robot target quaternions are WXYZ; SciPy uses XYZW unless explicitly using scalar-first helpers.

## Pitfalls

- Do not use an OBB orientation blindly for gripper orientation; use it only when object alignment matters.
- For placement, use the target surface center and a conservative release height above the surface.
- Very noisy or sparse points make OBB unreliable; return to `$libero-segmentation-to-points`.

## Related Skills

Use `$libero-segmentation-to-points` before geometry, `$libero-motion-control` to execute poses, and `$libero-articulated-and-contact-actions` for push or pull directions.
