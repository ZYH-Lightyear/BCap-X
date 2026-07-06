---
name: Find Placement
category: affordance
description: "Estimate where the held object's center should land on or inside a target."
---

# Find Placement

## Purpose

Produce a placement affordance for a held object. The affordance describes the desired
object center, target mode, top-down orientation, uncertainty, and review artifacts. It
does not execute motion.

## When to use

Use when the robot is already holding an object and the task asks to put it in, on, or
relative to a target. For nontrivial targets, Act should compute the placement
affordance directly and save review artifacts before release.

## Workflow

1. Ground the placement target using perception skills.
2. Decide the target mode:
   - `open_container`: basket, bin, cup, bowl, or any target with usable interior space.
   - `support_surface`: plate, tray, tabletop, cabinet top, stacking target, or object
     support surface. A bowl on a plate is a support-surface placement case.
3. Estimate where the held object's center should end up, not where the TCP should go.
4. Save a 2D overlay and, when useful, a 3D visualization of target points, desired
   center, and possible TCP compensation.
5. Return compact placement evidence for `release_at`.

The conventional local key is `EVIDENCE["placement_affordance"]`, with
`desired_object_center`, `place_quat`, `mode`, strategy notes, and artifact paths.

## Candidate Generation

- For open containers, prefer the opening or free-space center. Visible walls and rims
  often bias the raw mask centroid.
- For support surfaces, prefer the usable surface footprint center and a low,
  controlled release height.
- For relational targets, keep the relation in the target description and candidate
  ranking.
- If the held object was grasped off-center, keep the placement candidate in object
  coordinates; let release logic convert it into a TCP target.

## Local Checks

- Confirm the selected point projects over the intended target, not a neighboring
  object or container wall.
- Check clearance: high enough to avoid collision, low enough to avoid bounce/topple.
- If the target is small, save a review artifact before release.
- If target points are sparse or biased, return uncertainty instead of inventing a
  precise center.

## Failure Modes

- Wrong target: re-ground with a more specific target description.
- Object lands off-center after an edge/rim grasp: check whether `release_at` used the
  held-object center offset.
- Object bounces or topples: lower the release clearance and retry only after a state
  check.
- Target interior not visible: use a conservative support/opening estimate and mark
  uncertainty.

## Clean Reusable Rules

- Placement is about desired object state. It should usually target the held object's
  center, not the gripper/TCP.
- Open-container placement and support-surface placement are different affordances.
- Post-release checks should inspect the object-target relation, not just gripper
  opening.

## Weak Priors

- Prior release heights or common target modes can initialize a candidate, but live
  depth and mask geometry should set the final value.
- Known object grasp offsets can guide TCP compensation if the current held-object
  frame is missing.

## Prohibited Shortcuts

- Do not hardcode target coordinates, pixels, seed layouts, or object-name answer
  tables.
- Do not use the target mask centroid blindly when the visible surface is biased.

## Artifacts to Save

- Placement point overlay on the target image.
- Optional 3D visualization showing target points, desired object center, and TCP
  release point when an offset is applied.

## Optional Sidecars

`scripts/placement_affordance.py` contains helper functions for estimating placement,
saving overlays, and saving 3D visualizations. Use those helpers when they match the
current target; otherwise implement the same object-center placement workflow directly.
