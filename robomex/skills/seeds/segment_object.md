---
kind: observation
skill_id: segment_object
name: Segment Object by Language
description: Ground a named object into a 2D mask and filtered 3D points using language-conditioned segmentation (SAM3-style), the prerequisite for any geometric reasoning.
requires: []
produces: [object_mask, object_points]
keywords: [segment, mask, sam, locate, points, point_cloud, find, perception]
verify:
  - "Overlay the mask on the RGB frame; the highlighted region must cover exactly the named object and nothing else."
  - "Most mask pixels must carry valid depth, otherwise the 3D points are unreliable."
recovery:
  - "Empty or wrong mask: re-segment with a more specific name (add color or position, e.g. 'the red mug on the left')."
  - "Mask keeps failing: fall back to point-prompt segmentation from a VLM-selected pixel."
version: "0.1"
---

Ground an object mentioned in the task into pixels and 3D points before any geometric reasoning:

```python
mask_pc = get_object_3d_points_and_masks_from_language(object_name)
points, _ = filter_noise(mask_pc["points_3d"])
mask = mask_pc["agentview_mask"]
```

Rules:

- Use the most specific object name available; add color/position qualifiers when several similar objects are visible.
- Always `filter_noise` the raw points before estimating geometry (OBB, height, grasp) from them.
- Never proceed on an empty or tiny mask: re-segment with a refined name instead.
- The filtered `points` back claims like object geometry and grasp poses; segment again after the scene changed (e.g. after a grasp), do not reuse stale masks.
