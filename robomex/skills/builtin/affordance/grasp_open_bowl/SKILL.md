---
name: Open Bowl Rim Grasp
category: affordance
description: "Propose top-down rim or side-wall grasp candidates for an upward-facing open bowl."
---

# Open Bowl Rim Grasp

## Purpose

Generate grasp affordance evidence for an open bowl or similar upward-facing container.
This skill proposes contact points, orientations, object-center offsets, and artifacts.
It does not move the robot.

## When to use

Use after grounding a bowl/open container with visible rim or side wall. Prefer this
over center top-down grasping when the center is empty space or when a center pinch
would push the object. Avoid it for cans, boxes, closed bottles, and objects without a
contactable rim/side feature.

For difficult bowl grasps, Act should run the geometry/candidate analysis itself after
loading this skill. Use the Verifier SubAgent only to check a concrete visual claim,
for example whether the current fingertips visibly straddle the rim.

## Workflow

1. Read the target mask/points and estimate bowl center, rim height, visible radius, and
   side-wall support.
2. Sample several rim directions and several depths slightly below the visible rim.
3. Use a top-down approach. The jaw axis should be tangent to the rim and roughly
   orthogonal to the bowl radius at the contact point.
4. Check IK feasibility and visual contact with the segmented rim/side-wall region.
5. Save a candidate overlay and return the selected candidate plus the held-object
   center offset.

Conventional local keys are `EVIDENCE["object_grounding"]` for input,
`EVIDENCE["grasp_affordance"]` for selected candidate evidence, and
`EVIDENCE["held_object_frame"]` for `object_center_at_grasp` and
`object_center_offset_from_grasp`.

## Candidate Generation

- Prefer observed rim/support points over a synthetic perfect circle.
- Try multiple vertical margins below the rim; the correct depth depends on rim
  thickness, segmentation noise, and bowl shape.
- The grasp point may be slightly below the rim, but it must remain on the bowl side
  wall or rim support region.
- Record `object_center_offset_from_grasp = object_center - grasp_pos`. This is required
  for center-aware placement later.

## Local Checks

- The candidate point should project onto the bowl side/rim, not the empty interior.
- The jaw axis should be tangent to the rim; the approach should be from above.
- Reject candidates detached from the segmented bowl or outside IK reach.
- If all candidates look poor, return uncertainty or ask the Verifier to diagnose the
  current alignment rather than forcing a motion.

## Failure Modes

- Candidate appears off the bowl: lower radius scale, choose observed rim points, or
  re-segment.
- Top-down rim grasp closes on empty air: move slightly down the side wall or switch to
  GraspNet.
- IK infeasible: try another rim sector, reduce standoff, or return the failure reason.
- Bowl is placed off-center later: verify that release used the stored object-center
  offset.

## Clean Reusable Rules

- Open-container grasps target contactable rim or side features, not the empty center.
- The grasp point and object center are usually different for bowls.
- Later placement should align the object center to the target center, not the rim
  grasp point.
- Candidate quality is visual contact plus geometry plus IK, not only a numeric score.

## Weak Priors

- Rim depth can start from a few millimeters below the visible rim, then adapt to live
  geometry.
- Past successful offsets can be fallback hints, but current point cloud and review
  artifact dominate.

## Prohibited Shortcuts

- Do not assume a fixed bowl coordinate, fixed image pixel, fixed plate relation, or
  benchmark layout.
- Do not use gripper-width closure alone to reject a thin-rim grasp.

## Artifacts to Save

- `affordance_candidates.png` or equivalent overlay with grasp point, approach axis,
  jaw axis, score, and IK status.
- `affordance_candidates_3d.png` or equivalent 3D review plot when the point looks
  detached, the object is concave, or candidate ranking is uncertain.
- Return actual image paths in `artifact_refs`; a directory path alone is not enough
  for Act to review the affordance.

## Optional Sidecars

`scripts/bowl_grasp.py` contains `propose_open_bowl_grasps`, a pure geometry helper.
Use it when the segmented points are reliable. The helper returns candidate poses and
offsets; it never executes gripper or motion APIs.

Call pattern:

- Import `scripts/bowl_grasp.py` from the loaded skill base directory.
- Call `propose_open_bowl_grasps(points, solve_ik_fn=solve_ik)` with segmented
  world-frame target points.
- Save review artifacts with `robomex.perception.save_grasp_affordance_overlay` and,
  when 3D review is useful, `robomex.perception.save_grasp_affordance_3d`.
- Read `out["selected_candidate"]` first, then inspect `out["candidates"]` only if a
  fallback is needed.

Returned candidate keys:

- `pos`: grasp TCP position in world frame.
- `quat`: grasp TCP quaternion in world-frame `wxyz` order.
- `object_center`: estimated object center at grasp height.
- `object_center_offset_from_grasp`: `object_center - pos`; pass this forward for
  center-aware placement.
- `pregrasp_pos`: suggested approach position above the contact pose.
- `approach_axis`, `jaw_axis`, `radial_axis`: compact geometry axes for review.
- `ik_ok`, `ik_error`, `score`: feasibility and ranking signals.

Do not assume keys named `grasp_pos`, `position`, or `quaternion_wxyz` unless the
returned dict actually contains them.
