---
name: Segment Object by Language
category: perception
description: "Ground a language-described object into compact bbox, mask, point-cloud, and artifact evidence."
---

# Segment Object by Language

## Purpose

Locate one task-relevant object from live RGB-D observations. The output is evidence
for later geometry, grasp, placement, or state recovery. This skill is workflow
memory: it tells you how to ground robustly, not a fixed detector script.

## When to use

Use this when a later action needs object identity, a mask, 3D points, or a compact
world-frame center. Re-run it after the scene changes, after a failed grasp, or when a
previous object fact may be stale. If the object may already be held or placed, first
ask a categorical state question; do not force a new table-top localization.

For ambiguous targets, Act should enumerate a small candidate set, save crops/overlays,
and use categorical VLM or Verifier checks as needed. Do not delegate execution-grade
grounding to a SubAgent.

## When NOT to use

Do not re-segment a target that is already verified as held or placed, use this
skill to judge action success, or ask a general VLM to invent metric coordinates.

## Workflow

1. Preserve the full referring expression from the task. Do not collapse "the bowl to
   the right of the ramekin" into "bowl".
2. Use dedicated grounding APIs for spatial outputs. Use `vlm_bbox_detection`,
   `vlm_point_detection`, and SAM3; use `query_vlm` only for categorical image or crop
   judgments, never for coordinates, boxes, or points.
3. Convert accepted masks to world points with depth and camera pose, then denoise.
4. Save bbox, mask, crop, or candidate overlays into `ARTIFACTS_DIR`.
5. Keep bbox, mask, points, and center summaries local in `EVIDENCE`; save artifact
   paths for review.

Conventional local keys are `EVIDENCE["object_grounding"]`,
`EVIDENCE["grounding.mask"]`, and `EVIDENCE["grounding.points"]`. Keep large arrays
local; do not promote this geometry as cross-agent state.

## Candidate Generation

- Start specific and relational: color, material, text, shape, container/support, and
  left/right/front/back relations should stay in the prompt.
- When multiple similar objects exist, enumerate a small candidate set instead of
  taking the highest detector score.
- Filter candidates by generic geometry implied by the task: compact cylinder, flat
  package, elongated bottle, open bowl, support surface, or container.
- Convert only top-ranked masks to 3D points. Do not project every SAM3 mask.
- If identity is uncertain, save candidate crops and ask `query_vlm` to choose between
  crops categorically.

## Local Checks

- The mask should cover the described object, not the robot, table, target container,
  or a neighbor.
- The 3D points should be finite, non-empty, and physically plausible for the object
  category and scene surface.
- Relation words must be checked against other visible candidates, not ignored.
- If the object is expected to be held, elevated points near the gripper are more
  relevant than table points.
- Do not spend turns printing `obs.keys()`, image shapes, or function signatures unless
  an API truly failed; those checks rarely advance the physical task.

## Failure Modes

- Wrong similar object: sharpen the referring expression, enumerate candidates, and
  rank with relation plus geometry.
- Box on the robot or support surface: reject and re-ground from a fresh observation.
- Empty or noisy point cloud: retry with a better mask/crop or return uncertainty.
- Object not visible or already placed: return a state-oriented uncertain result rather
  than hallucinating a box.

## Clean Reusable Rules

- Detector score is a signal, not truth. Geometry, relation, and live state can
  override it.
- Similar-object scenes require candidate enumeration and local checks.
- `query_vlm` can judge identity or state from images; it must not be used as a
  coordinate detector.
- Large masks and point clouds stay in local `EVIDENCE` or artifacts. Cross-agent
  communication should use verifier verdicts, not executable grounding facts.

## Weak Priors

- Object aliases and common confusion pairs may guide prompt choice, but they are only
  starting points.
- Past failures can suggest which geometry to inspect first; they must not override
  the live image and depth.

## Prohibited Shortcuts

- Do not encode fixed pixels, fixed world coordinates, seed-specific locations, or
  benchmark answer tables as default rules.
- Do not choose a candidate only because it matched a previous task layout.
- Do not ask `query_vlm` to output bbox/point coordinates.

## Artifacts to Save

- `segment_vlm_box.png` or equivalent bbox overlay.
- `segment_sam3_mask.png` or equivalent mask overlay.
- Candidate crop/contact sheet when the target is ambiguous.

## Multimodal Evidence Contract

Consume one fresh RGB-D observation epoch. Publish a target-specific box/mask
overlay and finite world-frame points carrying that epoch. If the referring
expression remains ambiguous, emit `wrong_grounding` with candidate overlays
instead of publishing confident geometry for the wrong object.

## Optional Sidecars

`scripts/segment_object.py` contains `ground_object_from_observation`, a runnable helper
that uses the live sandbox APIs passed through `apis=globals()`. Use it only when it
matches the current problem; otherwise implement the same workflow directly. The
helper writes the conventional local keys `object_grounding`, `grounding.mask`, and
`grounding.points`.

## Reference Code

The skill's `scripts/` directory is already on `sys.path`; call the helper exactly as
below — do not probe it with `dir()` or `inspect`.

```python
from segment_object import ground_object_from_observation

obs = get_observation()
grounding = ground_object_from_observation(
    obs,
    target_name=target_name,          # referring expression, e.g. "alphabet soup can"
    artifacts_dir=ARTIFACTS_DIR,      # overlays are saved here
    evidence=EVIDENCE,                # fills object_grounding / grounding.mask / grounding.points
    apis=globals(),                   # passes vlm_bbox_detection, segment_sam3_*, ...
)

# Returned keys: target, bbox, mask_key, points_key, center_xyz, num_points,
# and artifacts = {"bbox_overlay": ..., "mask_overlay": ...} (absolute paths
# under ARTIFACTS_DIR). Persist mask/points arrays as .npy under ARTIFACTS_DIR.
import os
import numpy as np
mask_path = os.path.join(ARTIFACTS_DIR, "mask.npy")
points_path = os.path.join(ARTIFACTS_DIR, "object_points.npy")
np.save(mask_path, EVIDENCE["grounding.mask"])
np.save(points_path, EVIDENCE["grounding.points"])
NODE_RESULT = {
    "outputs": {
        "object_grounding": {
            "payload": {
                "target": grounding["target"],
                "bbox_xyxy": grounding["bbox"],
                "center_xyz": grounding["center_xyz"],
                "num_points": grounding["num_points"],
            },
            "confidence": 0.9,
            "frame": "world",
            "artifacts": {
                "mask_npy": "mask.npy",
                "points_npy": "object_points.npy",
                "mask_overlay": "segment_sam3_mask.png",
            },
        }
    },
    "recommended_next": "geometry_or_affordance",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
