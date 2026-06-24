---
name: libero-segmentation-to-points
description: Use in CaP-X LIBERO pick, place, put-in, put-on, drawer, stove, basket, plate, cabinet, or grasp tasks when an image object or receptacle must become a SAM3 mask, world-frame 3D points, a point cloud, or a single world coordinate.
---

# LIBERO Segmentation To Points

Convert image-level targets into 3D geometry that grasping, placement, and contact actions can use.

## APIs

```python
results = segment_sam3_text_prompt(rgb, "black bowl")
results = segment_sam3_point_prompt(rgb, (x, y))
points = mask_to_world_points(mask, depth, intrinsics, pose_mat)
points, colors = filter_noise(points)
points = subsample_point_cloud(points, max_points=10000)
cloud = depth_to_point_cloud(depth, intrinsics)
world_point = pixel_to_world_point(u, v, z, intrinsics, pose_mat)
```

## Workflow

1. Get a fresh observation.
2. Prefer text prompt segmentation when the object is visually distinctive.
3. Prefer point prompt segmentation after `$libero-language-grounding` when there are multiple similar objects.
   If the point came from a Qwen-family VLM, convert possible `0..1000` normalized coordinates to pixels first:

```python
h, w = rgb.shape[:2]
if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
    x = int(round(x / 1000 * (w - 1)))
    y = int(round(y / 1000 * (h - 1)))
else:
    x, y = int(round(x)), int(round(y))
```

4. Pick the highest-score SAM3 result with a non-empty mask:

```python
results = segment_sam3_point_prompt(rgb, (x, y))
assert results, "SAM3 returned no masks"
mask = max(results, key=lambda r: r.get("score", 0.0))["mask"]
```

5. Convert the mask to world points and clean them:

```python
cam = obs["agentview"]
points = mask_to_world_points(mask, cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"])
points, _ = filter_noise(points)
if len(points) > 10000:
    points = subsample_point_cloud(points, 10000)
assert len(points) > 50, "not enough object points"
```

## Choosing Outputs

- Use dense `points` for `plan_grasp`, OBB, placement center, and object size.
- Use `pixel_to_world_point` for small handles, buttons, contact points, or push targets.
- Use `depth_to_point_cloud` when a full camera-frame cloud is needed before selecting a region.

## Pitfalls

- Recompute masks after any object or camera-changing motion.
- SAM3 returns a list; never assume result 0 is best.
- Empty masks, zero depth, and tiny point sets usually mean wrong grounding or occlusion.
- Qwen-style VLM point/box outputs may be normalized to `0..1000`; rescale them to real image pixels before point-prompt SAM3 or `pixel_to_world_point`.
- Filter points before OBB, grasp planning, or center estimation.

## Related Skills

Use `$libero-language-grounding` before point-prompt segmentation, `$libero-geometry-and-frames` to reason over points, and `$libero-grasp-object` to grasp segmented objects.
