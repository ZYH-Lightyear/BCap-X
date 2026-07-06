---
name: GraspNet Candidate Grasp
category: affordance
description: "Use Contact-GraspNet candidates as read-only 6-DoF grasp affordance evidence."
---

# GraspNet Candidate Grasp

## Purpose

Generate, rank, and visualize learned 6-DoF grasp candidates from depth and a target
mask. This skill is for affordance analysis; Act still owns final robot execution.

## When to use

Use after `segment_object` when simple top-down or PCA/side grasping is unreliable:
open containers, cups, bowls, irregular objects, clutter, tilted objects, or repeated
misses. It is also a good fallback when local geometry heuristics disagree.

Act runs this analysis directly. If the resulting candidate or execution state is
visually ambiguous, ask the Verifier SubAgent to review that concrete claim.

## Workflow

1. Use the current RGB-D camera, target mask, intrinsics, and camera pose.
2. Run Contact-GraspNet or the available `plan_grasp` API on the masked depth.
3. Transform each candidate from camera frame to world frame.
4. Filter top candidates by IK feasibility and local visual plausibility.
5. Save an affordance overlay and return one selected candidate plus uncertainty.

## Candidate Generation

- Inspect more than the top score when the first pose is IK-infeasible or visually
  implausible.
- Keep candidate lists local or in artifacts; return only selected/top compact entries.
- If segmentation is weak, fix the mask before trusting GraspNet scores.
- For bowl/cup cases, compare GraspNet output with open-rim affordance when both are
  available.

## Local Checks

- Candidate pose must be transformed with the camera pose before IK checks.
- The grasp point should lie near the target mask/points, not a neighbor or support.
- Reject poses that collide obviously with the table, container wall, or robot.
- Do not use gripper width as an affordance-quality filter before execution.

## Failure Modes

- No candidates: re-check depth, mask, and target crop.
- Top candidate wrong object: tighten segmentation and rerun.
- IK infeasible: try lower-ranked candidates or a simpler top-down/side strategy.
- Geometrically implausible candidate: return uncertainty and save the overlay.

## Clean Reusable Rules

- Learned grasp score is not enough; use IK and visual contact checks.
- Execution success is checked after motion and lift, not inside this affordance skill.
- GraspNet is a fallback for uncertain physical geometry, not a replacement for target
  grounding.

## Weak Priors

- Bowls/cups/containers often benefit from GraspNet, but open-rim geometry may be
  cheaper and more interpretable when the rim is clear.
- Previous failed grasp pose can guide which candidate family to avoid next.

## Prohibited Shortcuts

- Do not execute the highest-scoring candidate blindly.
- Do not store all raw candidates or large arrays in cross-agent evidence.
- Do not hardcode object-specific candidate indices from past seeds.

## Artifacts to Save

- Candidate overlay with target mask, selected point, approach axis, jaw axis, score,
  and IK status.
- Optional short text trace describing why the chosen candidate beat alternatives.

## Optional Sidecars

No sidecar is required. Use live environment APIs such as `plan_grasp`, transform
utilities, IK, and `save_grasp_affordance_overlay` when available.
