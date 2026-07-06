---
name: segmentation-to-points
description: Use when an image object or region must become a mask, world-frame 3D points, a point cloud, or one world coordinate.
---

# Segmentation To Points

Convert image-level targets into 3D geometry for grasping, placement, and contact actions.

## APIs

```python
results = segment_object("black bowl", camera="main")
results = segment_object((x, y), camera="main")
points = mask_to_world_points(mask, camera="main")
world_point = pixel_to_world_point(x, y, camera="main")
points, colors = filter_noise(points)
points = subsample_point_cloud(points, max_points=10000)
```

## Workflow

1. Get a fresh observation and `cam = get_camera("main")`.
2. Prefer text segmentation when the object is visually distinctive.
3. Prefer point segmentation after `$language-grounding` when there are similar objects.
4. Choose the highest-score non-empty mask:

```python
results = segment_object((x, y), camera="main")
assert results, "segmentation returned no masks"
mask = max(results, key=lambda r: r.get("score", 0.0))["mask"]
```

5. Convert and clean points:

```python
points = mask_to_world_points(mask, camera="main")
points, _ = filter_noise(points)
if len(points) > 10000:
    points = subsample_point_cloud(points, 10000)
assert len(points) > 50, "not enough target points"
```

## Choosing Outputs

- Use world-frame `points` for OBBs, placement centers, fallback grasp centers, and object size.
- Use `plan_grasps(mask, camera="main")` when learned grasp planning is available.
- Use `pixel_to_world_point` for handles, buttons, and small contact points.

## Pitfalls

- Pixel coordinates are `(x, y)` = `(col, row)`.
- Masks and depth images are indexed as `[row, col]`.
- Recompute masks after object or camera motion.
- Empty masks or tiny point sets usually mean wrong grounding, occlusion, or invalid depth.

## Related Skills

Use `$language-grounding` before point-prompt segmentation, `$geometry-and-frames` to reason over points, and `$grasp-object` to grasp segmented objects.
