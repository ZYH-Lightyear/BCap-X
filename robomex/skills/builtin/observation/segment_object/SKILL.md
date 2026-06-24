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
then let SAM3 produce the precise mask from that point.

```python
import json, os, re
import numpy as np
from PIL import Image as _I, ImageDraw as _D

obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
H, W = rgb.shape[:2]

# 1) VLM localization — box in 0-1000 normalized coords (Qwen convention).
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

assert box and len(box) == 4, f"VLM localization failed for '{object_name}': {reply!r}"

# Rescale 0-1000 normalized coords -> real pixels.
x1, y1, x2, y2 = box
px = [x1 / 1000.0 * W, y1 / 1000.0 * H, x2 / 1000.0 * W, y2 / 1000.0 * H]
EVIDENCE["target_box"] = [float(v) for v in px]

# Save VLM box overlay to disk (debug / verifier evidence).
_ann = _I.fromarray(rgb.copy())
_D.Draw(_ann).rectangle(px, outline=(255, 0, 0), width=3)
_box_path = os.path.join(ARTIFACTS_DIR, "segment_vlm_box.png")
_ann.save(_box_path)
EVIDENCE["vlm_box_image"] = _box_path

# 2) SAM3 point prompt at the VLM-located center -> precise mask.
cx = float(np.clip((px[0] + px[2]) / 2.0, 0, W - 1))
cy = float(np.clip((px[1] + px[3]) / 2.0, 0, H - 1))
pr = segment_sam3_point_prompt(rgb, (cx, cy))
assert pr, f"SAM3 returned no mask at ({cx:.0f}, {cy:.0f}) for '{object_name}'"
mask = max(pr, key=lambda r: r["score"])["mask"]  # (H, W) bool

# Save SAM3 mask contour overlay to disk (debug / verifier evidence).
import cv2 as _cv
_vis = rgb.copy()
_contours, _ = _cv.findContours(mask.astype(np.uint8), _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
_cv.drawContours(_vis, _contours, -1, (0, 255, 0), 2)
_D.Draw(_I.fromarray(_vis)).rectangle(px, outline=(255, 0, 0), width=2)
_mask_path = os.path.join(ARTIFACTS_DIR, "segment_sam3_mask.png")
_I.fromarray(_vis).save(_mask_path)
EVIDENCE["sam3_mask_image"] = _mask_path

points = mask_to_world_points(mask, depth, cam["intrinsics"], cam["pose_mat"])
points, _ = filter_noise(points)
```

`ARTIFACTS_DIR` is automatically set to the current sub-goal's output directory.
`EVIDENCE` is a sandbox-persistent dict (seeded at sub-goal start). The saved images
let both the Verifier Agent and human reviewers inspect exactly what VLM grounded and
what SAM3 segmented.

## Rules

- VLM localization comes first: it picks the *right* object semantically. SAM3 only
  turns that point into a precise mask — do not skip the VLM step when several similar
  objects are on the table.
- VLM coordinates are NORMALIZED to 0-1000 (Qwen convention). ALWAYS rescale by the real
  `W`/`H` (`x_px = x / 1000 * W`); never feed raw 0-1000 values to SAM3 as pixels.
- Use the most specific object name available; add color/position qualifiers when
  several similar objects are visible.
- `segment_sam3_*` return a LIST of candidate dicts; pick by `score`, never assume index 0.
- Always `filter_noise` before using the points downstream (e.g. to derive a grasp).

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: the mask covers exactly the named object and `len(points)` is at least a few hundred.

## Failure modes

- VLM returns malformed / non-JSON output: re-ask `query_vlm` with a stricter "ONLY JSON"
  instruction, or try a more specific object description.
- Wrong object grounded (e.g. milk instead of soup): the VLM localized poorly — re-ask
  with a more specific description (color, position, "the can labelled ..."), or crop
  the image to the relevant region before querying.
- SAM3 point prompt returns nothing at the VLM point: nudge the point toward the box
  center, or try a couple of points sampled inside the rescaled box.
