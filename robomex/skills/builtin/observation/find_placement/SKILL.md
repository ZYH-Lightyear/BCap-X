---
name: Find Placement
category: observation
description: Ground a named placement target (container / region / surface) and compute a safe 3D release point above it, the prerequisite for placing a held object.
---

# Find Placement

Ground the named placement target into pixels + 3D world points, then derive a single
safe **release point** above it. This produces the `place_pos` that the release action
relies on, and publishes the target's box so the Verifier can check *which* target.

## When to use

When the gripper is holding an object and you need to know *where* to drop it (into a
container, onto a region/plate, etc.). Run it fresh against the live scene — never reuse
a stale target location.

## Procedure

Localize the target with a VLM first (it disambiguates *which* target the task means),
let SAM3 produce the precise mask, lift it to 3D, then place the release point above the
target's top-center with clearance.

```python
import json
import re
import numpy as np

obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
H, W = rgb.shape[:2]

# 1) VLM localization in 0-1000 NORMALIZED coords (Qwen convention), then rescale.
q = (
    "You are given a tabletop image. "
    f"Find the single placement target that best matches: '{target_name}'. "
    "Reply with ONLY a JSON object {\"box\": [x1, y1, x2, y2]}, coordinates NORMALIZED "
    "to 0-1000 (x right, y down). No prose."
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
EVIDENCE["place_box"] = None  # published in REAL pixels for the Verifier
if box and len(box) == 4:
    px = [box[0] / 1000.0 * W, box[1] / 1000.0 * H, box[2] / 1000.0 * W, box[3] / 1000.0 * H]
    EVIDENCE["place_box"] = [float(v) for v in px]
    cx = float(np.clip((px[0] + px[2]) / 2.0, 0, W - 1))
    cy = float(np.clip((px[1] + px[3]) / 2.0, 0, H - 1))
    pr = segment_sam3_point_prompt(rgb, (cx, cy))
    if pr:
        mask = max(pr, key=lambda r: r["score"])["mask"]

if mask is None:
    # Fallback: SAM3 text grounding (less reliable when similar targets coexist).
    results = segment_sam3_text_prompt(rgb, text_prompt=target_name)
    assert results, f"no mask for placement target '{target_name}'"
    mask = max(results, key=lambda r: r["score"])["mask"]

points = mask_to_world_points(mask, depth, cam["intrinsics"], cam["pose_mat"])
points, _ = filter_noise(points)
assert len(points) > 50, "too few target points to place reliably"

# 2) Release point: above the target's (x, y) center, a clearance above its top surface.
center = points.mean(axis=0)
top_z = points[:, 2].max()
clearance = 0.10  # release above the opening so the object drops in / settles on top
place_pos = np.array([center[0], center[1], top_z + clearance])
EVIDENCE["place_point"] = [float(v) for v in place_pos]
```

## Rules

- VLM localization comes first: it picks the *right* target semantically; SAM3 turns that
  point into a precise mask. Do not skip the VLM step when several similar targets exist.
- VLM coordinates are NORMALIZED to 0-1000 (Qwen convention). ALWAYS rescale by the real
  `W`/`H`; never feed raw 0-1000 values to SAM3 as pixels.
- Release ABOVE the target (clearance over `top_z`), never at/below the rim — dropping at
  contact height can topple a container or jam the object.
- Always `filter_noise` before deriving geometry; reject near-empty point sets.
- Publish `EVIDENCE["place_box"]` / `EVIDENCE["place_point"]` so the Verifier can audit
  which target you chose without re-localizing.

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: the box lands on the named target and `place_pos` sits above its top.

## Failure modes

- VLM returns malformed / non-JSON output: the code falls back to SAM3 text grounding;
  you can also re-ask `query_vlm` with a stricter "ONLY JSON" instruction.
- Wrong target chosen (e.g. the wrong bowl): re-ask the VLM with a more specific
  description (color, position, "the basket on the left").
- Sparse/!noisy target points (thin rim, transparent container): nudge the point toward
  the visible body, or raise the clearance so the release tolerates depth error.
