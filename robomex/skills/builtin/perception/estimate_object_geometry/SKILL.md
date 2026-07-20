---
name: Estimate Object Geometry
category: perception
description: "Summarize segmented 3D points into coarse shape, pose, dimensions, and grasp-relevant geometry hints."
---

# Estimate Object Geometry

## Purpose

Turn segmented object points into a compact physical description. The result helps
choose a grasp or placement strategy without making the geometry estimate a hard
truth.

## When to use

Use after `segment_object` when the object is not an obvious upright compact item, when
pose matters, or after a failed action changed the object. Typical cases include
bowls, cups, lying cylinders, elongated objects, thin packages, and partially occluded
targets.

## When NOT to use

Do not use raw scene points, mix multiple object masks, command robot motion, or
promote a noisy PCA axis to unquestioned physical truth.

## Workflow

1. Read filtered object points from local evidence or the current code block.
2. Compute robust summaries: center, top/bottom z, height, XY extent, oriented bounding
   box when available, and principal axes.
3. Produce a coarse `pose_hint`, such as `upright`, `lying_or_elongated`,
   `flat_or_low`, `open_or_container_like`, or `compact_or_uncertain`.
4. Publish only the compact summary needed by downstream reasoning.

## Candidate Generation

- For elongated objects, compare principal extents and identify the long body axis.
- For low/flat objects, check height against the visible footprint before deciding
  top-down depth.
- For bowls/cups/open objects, treat empty interior and rim/side geometry separately.
- For uncertain masks, generate geometry from each credible candidate rather than
  averaging unrelated points.

## Local Checks

- Reject geometry if points include the table, robot, or a target container wall.
- Check that the height and footprint are plausible for the named object family.
- Treat OBB axes as hints; camera viewpoint, partial masks, and occlusion can rotate
  them.
- If the pose hint contradicts the image, prefer a fresh segmentation or crop review.

## Failure Modes

- Sparse points: re-ground or use a tighter crop before estimating.
- Background leakage: re-segment before selecting a grasp family.
- Unstable OBB axis: rely on multiple simple summaries rather than one axis.
- Object moved during a prior action: refresh observation and recompute geometry.

## Clean Reusable Rules

- Geometry is for strategy selection, not proof of task success.
- Shape hints should reduce risk: top-down for compact upright objects, rim/GraspNet for
  open shapes, PCA/side for elongated or fallen objects.
- Store big point clouds locally; share only compact facts and artifact references.

## Weak Priors

- Object families suggest initial strategies, but live geometry dominates.
- Previous failure notes can suggest which dimension to inspect first.

## Prohibited Shortcuts

- Do not use object name alone to force a strategy when live geometry disagrees.
- Do not promote a fixed size, pose, or coordinate from one benchmark seed into a
  default rule.

## Artifacts to Save

- Optional geometry overlay or 3D plot when the strategy choice is uncertain.
- Short text summary in the trace: pose hint, dimensions, and reason for the chosen
  grasp family.

## Multimodal Evidence Contract

Consume only the typed world-frame points and observation epoch produced by
grounding. Publish compact robust dimensions and pose hints with the same epoch.
Save an OBB/PCA visualization when axis choice changes grasp strategy. Sparse or
non-finite inputs emit `infeasible`; they are not repaired with fabricated extents.

## Optional Sidecars

`estimate_object_geometry` is a contracted canonical function pre-bound from
`scripts/object_geometry.py`. Call it directly on the supplied point artifact;
use environment geometry utilities only for optional visual diagnostics.

## Reference Code

```python
import numpy as np

points = np.load(INPUTS["object_points"]["refs"][0]["path"])
geometry = estimate_object_geometry(points)
```
