---
name: Release At
category: motion
description: "Release a held object at a placement affordance, compensating off-center grasps when available."
---

# Release At

## Purpose

Execute a bounded release attempt from placement evidence. The goal is to make the
held object's center land at the desired target point, then retreat without disturbing
the placed object.

## When to use

Use after `find_placement` or equivalent evidence says where the held object's center
should land. The object should already be held. If holding state is uncertain, check it
before moving to the target.

## Workflow

1. Read `placement_affordance.desired_object_center` and `place_quat`.
2. Read optional `held_object_frame.object_center_offset_from_grasp`.
3. Convert object-center target to TCP release pose:
   `tcp_release_pos = desired_object_center - object_center_offset_from_grasp`.
   If no offset exists, use the desired object center as the TCP target and mark lower
   confidence for edge/rim grasps.
4. Move top-down to a safe pre-release pose, descend to release height, open gripper,
   and retreat upward.
5. Use a state-first post-release check before attempting recovery.

The conventional local inputs are `EVIDENCE["placement_affordance"]` and optional
`EVIDENCE["held_object_frame"]`. In short form, when
`offset = object_center_offset_from_grasp`, use
`tcp_release_pos = desired_object_center - offset`. Do not align the grasp point to the
target center when this offset is known.

## Candidate Generation

- Release height comes from placement affordance and live geometry, not a fixed global
  value.
- For rim/edge/handle grasps, generate the TCP pose from object-center compensation.
- For uncertain offset, prefer a cautious low release and post-release check rather
  than a high drop.

## Local Checks

- The TCP target should not align the grasp point to the target center when an object
  center offset is known.
- Opening state should be checked, but task success is the object-target relation.
- Retreatment should move away without dragging or knocking the object.
- If the object is still held/stuck, address that state before re-segmenting on the
  table.

## Failure Modes

- Object placed off-center: verify offset compensation and placement target center.
- Object bounced/toppled: lower the release height and retry only after a state check.
- Object still held: open again or retreat before re-localizing.
- Wrong target: re-run placement grounding with a more specific description.

## Clean Reusable Rules

- Release is center-aware: object center and TCP are different after off-center grasps.
- Post-release verification is categorical state reasoning, not coordinate detection.
- Recovery begins with state classification: at target, outside target, still held, not
  visible, or uncertain.

## Weak Priors

- Top-down release is the default v1 orientation for most LIBERO placement tasks.
- Known object-center offsets can be reused as a fallback only when current evidence is
  missing.

## Prohibited Shortcuts

- Do not hardcode release coordinates or target-specific offsets from one seed.
- Do not assume a failed release before checking the live image.
- Do not ask `query_vlm` for coordinates during post-release checking.

## Artifacts to Save

- Placement overlay and optional 3D placement visualization from the affordance step.
- Attempt record with desired object center, applied offset summary, release pose, and
  post-release state.

## Optional Sidecars

`scripts/release.py` contains small helpers for `compute_tcp_release_pos` and
`place_quat_from_affordance`. Use them when helpful, or implement the same formula
directly.
