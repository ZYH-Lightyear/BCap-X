# Act-Owned Grounding Notes

This reference replaces the old grounding-delegation pattern.

Act owns execution-grade grounding. When a target must be localized for motion, Act
should:

- inspect the current observation;
- call `vlm_bbox_detection` / `vlm_point_detection` for spatial grounding, never
  `query_vlm` for boxes or points;
- call SAM3 from the accepted bbox or point;
- convert the mask to world points and denoise them;
- save bbox/mask/crop overlays under `ARTIFACTS_DIR`;
- keep large arrays in local `EVIDENCE` under keys such as `grounding.mask` and
  `grounding.points`.

Use the Verifier SubAgent only for categorical checks such as "is this crop the target
object?" or "does the current image show the object held by the gripper?". The Verifier
should not return executable grounding geometry.
