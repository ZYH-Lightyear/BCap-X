---
name: Pick Object
category: high_level
description: High-level grasp of a named object off a surface, orchestrating perception (segment, geometry, grasp candidates) and the grasp action end-to-end.
---

# Pick Object

Pick a named object off a surface. This is a compound skill: it does not call raw
APIs itself, it orchestrates the leaf skills below. Consult each leaf's guidance
when you write the code for that step; adapt it, do not copy it blindly. The live
observation is authoritative.

## When to use

The task needs the robot to pick up / grasp a named object that is resting on a
surface and is not yet held.

## Decomposition

1. **segment_object** — ground `object_name` into a mask + filtered 3D points.
2. **estimate_geometry** — turn the points into an OBB / size estimate so the grasp
   reasons about real extents.
3. **grasp_candidates** — get GraspNet candidates and keep a top-down, IK-feasible
   one. Never trust the raw top-1.
4. **grasp_object** — execute the vetted grasp and confirm the hold with a small lift.

This is the default reading order, not a fixed script — you may deviate when the
scene calls for it (e.g. skip geometry for a trivially small object).

## Postcondition

The named object is held in the closed gripper: gripper width is above zero and the
object lifts with the arm.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: gripper width is above zero and the named object lifts with the arm.

## Failure modes

- Grasp missed (gripper fully closed): open the gripper, re-run segment + grasp
  candidates for a fresh IK-feasible pose, retry once.
- Wrong object grasped: re-segment with a more specific name (add color/position)
  before regenerating grasp candidates.
