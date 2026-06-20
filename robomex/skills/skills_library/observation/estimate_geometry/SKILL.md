---
name: Estimate Object Geometry
category: observation
description: Turn filtered 3D points into a verified size/position estimate (center, extents, top/bottom z) so heights and clearances are computed, never guessed.
---

# Estimate Object Geometry

Turn an object's (or container's) filtered world points into a size/position estimate
so heights and clearances are measured, never guessed.

## When to use

After segmenting an object, before any motion that depends on its size — picking
(needs extents/center) or placing (needs the container's `top_z` and `center`).

## Procedure

```python
obb = get_oriented_bounding_box_from_3d_points(points)   # keys: center, extent, R
center = obb["center"]                                    # (3,) world frame
# Heights are most robust straight from the world-frame z values:
top_z = points[:, 2].max()
bottom_z = points[:, 2].min()
height = top_z - bottom_z
```

## Report (hand off your result for verification)

When you finish, leave a manifest in `RESULT` so the Verifier checks *your* actual
artifacts (not a re-run). Save the heavy arrays to disk under `EVIDENCE_DIR` (a
str already in scope) and put only scalars + paths in `RESULT`:

```python
import os, numpy as np
os.makedirs(EVIDENCE_DIR, exist_ok=True)
np.save(os.path.join(EVIDENCE_DIR, "points.npy"), points)   # filtered points you measured
np.save(os.path.join(EVIDENCE_DIR, "mask.npy"), mask)       # the SAM mask you used
RESULT = {
    "skill": "estimate_geometry", "object": object_name,
    "height": float(height), "top_z": float(top_z), "bottom_z": float(bottom_z),
    "n_points": int(len(points)),
    "obb": {"center": obb["center"].tolist(),
            "extent": obb["extent"].tolist(),
            "R": obb["R"].tolist()},
    "points_path": os.path.join(EVIDENCE_DIR, "points.npy"),
    "mask_path": os.path.join(EVIDENCE_DIR, "mask.npy"),
}
```

## Rules

- `extent` is in the OBB's own frame (axes follow `R`); for vertical heights and
  surface z, use the world-frame z statistics of the points instead.
- Use this for BOTH the manipulated object (its height feeds the place clearance)
  and the container (its `top_z` and `center` define the drop pose).

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: extents are plausible (~0.01–0.5 m) and `top_z > bottom_z` within the workspace.

### Verifier reference (`scripts/verify.py`)

These functions are preloaded for the Verifier as `VERIFY_PRIMITIVES["estimate_geometry"]`
(already wired to the sandbox APIs + the executor's `RESULT` manifest). Call them, adapt
them, or copy them.

- **One-shot (recommended):**
  `verify(object_name, out_dir, rubric_path=None, model=..., server_url=..., api_key=None, manifest_path=None, use_vlm=True) -> dict`
  — loads the executor's artifacts from its `RESULT` manifest (falls back to an independent
  re-measure only if absent), runs numeric guards, renders the provenance overlay to
  `<out_dir>/geometry_overlay.png`, VLM-judges it against `ref/verify.md`, prints
  `VERIFY_RESULT <json>`, and returns `{verdict, confidence, reason, overlay, evidence_source, height, top_z, bottom_z, extent, ...}`.
- **Building blocks (to compose your own check):**
  - `load_claim(camera="agentview", manifest_path=None) -> (cam, rgb, mask, points, obb, stats) | None`
    — reads the executor's real artifacts (`RESULT` / on-disk `.npy`); returns `None` if no manifest.
  - `render_evidence(cam, rgb, mask, points, obb, stats, out_path) -> path`
    — draws the SAM mask (yellow), reprojected points (red), measured OBB (green), and top/bottom
    z markers onto the agentview RGB.
  - `numeric_guards(stats) -> (ok, flags)` — cheap deterministic sanity checks.
  - `judge(image_path, rubric, stats, model, server_url, api_key=None) -> {verdict, confidence, reason}`
    — VLM-judge an overlay against a rubric.

## Failure modes

- Absurd extents: re-filter the points (`filter_noise`) or re-segment the object,
  then re-estimate.
