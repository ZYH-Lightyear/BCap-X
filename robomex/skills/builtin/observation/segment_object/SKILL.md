---
name: Segment Object by Language
category: observation
description: Ground a named object into a 2D mask and noise-filtered 3D world points (VLM localization then SAM3), the prerequisite for locating or grasping it.
---

# Segment Object by Language

Ground an object mentioned in the task into pixels and 3D world points. This produces
the `mask` and filtered `points` that downstream steps (e.g. grasping) rely on.

## When to use

Whenever you need to locate a named object (to grasp it, measure it, or place into
it). Run it fresh after the scene changes — never reuse a stale mask.

## Procedure

Localize the target with a VLM first (it disambiguates *which* object the task means),
then let SAM3 produce the precise mask from that point. Fall back to SAM3 text grounding
only if the VLM localization fails.

```python
import json
import re
import numpy as np

obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
H, W = rgb.shape[:2]

# 1) VLM localization. Ask for a box in 0-1000 NORMALIZED coords (Qwen's native
#    convention) so the answer is resolution-independent and easy to rescale.
q = (
    "You are given a tabletop image. "
    f"Find the single object that best matches: '{object_name}'. "
    "Reply with ONLY a JSON object {\"box\": [x1, y1, x2, y2]}, where coordinates are "
    "NORMALIZED to the range 0-1000 (x to the right, y downward). No prose."
)
reply = query_vlm(q, images=rgb)

box = None
m = re.search(r"\{.*\}", reply, re.S)
if m:
    try:
        box = json.loads(m.group(0)).get("box")
    except json.JSONDecodeError:
        box = None

mask = None
if box and len(box) == 4:
    # Rescale 0-1000 normalized coords -> real pixels by the actual image size.
    x1, y1, x2, y2 = box
    cx = float(np.clip((x1 + x2) / 2.0 / 1000.0 * W, 0, W - 1))
    cy = float(np.clip((y1 + y2) / 2.0 / 1000.0 * H, 0, H - 1))
    # 2) SAM3 point prompt at the VLM-located point -> precise mask.
    pr = segment_sam3_point_prompt(rgb, (cx, cy))
    if pr:
        mask = max(pr, key=lambda r: r["score"])["mask"]

if mask is None:
    # 3) Fallback: SAM3 text grounding (less reliable when similar objects coexist).
    results = segment_sam3_text_prompt(rgb, text_prompt=object_name)
    assert results, f"no mask for '{object_name}'"
    mask = max(results, key=lambda r: r["score"])["mask"]      # (H, W) bool

points = mask_to_world_points(mask, depth, cam["intrinsics"], cam["pose_mat"])
points, _ = filter_noise(points)                                # always filter before geometry
```

## Rules

- VLM localization comes first: it picks the *right* object semantically. SAM3 only
  turns that point into a precise mask — do not skip the VLM step when several similar
  objects are on the table (that is exactly when text grounding picks the wrong one).
- VLM coordinates are NORMALIZED to 0-1000 (Qwen convention). ALWAYS rescale by the real
  `W`/`H` (`x_px = x / 1000 * W`); never feed raw 0-1000 values to SAM3 as pixels.
- Use the most specific object name available; add color/position qualifiers when
  several similar objects are visible.
- `segment_sam3_*` return a LIST of candidate dicts; pick by `score`, never assume index 0.
- Always `filter_noise` before using the points downstream (e.g. to derive a grasp).

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: the mask covers exactly the named object and `len(points)` is at least a few hundred.

## Failure modes

- VLM returns malformed / non-JSON output: the code falls back to SAM3 text grounding
  automatically; you can also re-ask `query_vlm` with a stricter "ONLY JSON" instruction.
- Wrong object grasped (e.g. milk instead of soup): the VLM localized poorly — re-ask
  with a more specific description (color, position, "the can labelled ..."), or crop
  the image to the relevant region before querying.
- SAM3 point prompt returns nothing at the VLM point: nudge the point toward the box
  center, or try a couple of points sampled inside the rescaled box.
- Empty result list from SAM3 text fallback: get a pixel with `point_prompt_molmo`, then `segment_sam3_point_prompt`.
