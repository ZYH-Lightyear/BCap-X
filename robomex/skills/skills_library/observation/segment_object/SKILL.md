---
name: Segment Object by Language
category: observation
description: Ground a named object into a 2D mask and noise-filtered 3D world points (SAM3 text prompt), the prerequisite for any geometric reasoning.
---

# Segment Object by Language

Ground an object mentioned in the task into pixels and 3D world points before any
geometric reasoning. This produces the `mask` and filtered `points` later steps rely on.

## When to use

Whenever you need to locate a named object (to grasp it, measure it, or place into
it). Run it fresh after the scene changes — never reuse a stale mask.

## Procedure

```python
obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]

results = segment_sam3_text_prompt(rgb, text_prompt=object_name)
assert results, f"no mask for '{object_name}'"
mask = max(results, key=lambda r: r["score"])["mask"]          # (H, W) bool

points = mask_to_world_points(mask, depth, cam["intrinsics"], cam["pose_mat"])
points, _ = filter_noise(points)                                # always filter before geometry
```

## Rules

- Use the most specific object name available; add color/position qualifiers when
  several similar objects are visible.
- `segment_sam3_text_prompt` returns a LIST of candidate dicts; pick by `score`,
  never assume index 0.
- Always `filter_noise` before estimating geometry (OBB, height, grasp) from points.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: the mask covers exactly the named object and `len(points)` is at least a few hundred.

## Failure modes

- Empty result list from SAM3: re-segment with a more specific name (add color or
  position, e.g. "the red mug on the left").
- Text prompt keeps failing: get a pixel with `point_prompt_molmo`, then segment
  with `segment_sam3_point_prompt`.
