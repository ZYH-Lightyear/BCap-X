---
name: Pick Object
category: high_level
description: High-level grasp of a named object off a surface, orchestrating segmentation and the grasp action end-to-end.
---

# Pick Object

Pick a named object off a surface. This is a compound skill: it does not call raw
APIs itself, it points you at the leaf skills you may need. Consult each leaf's
guidance when you write the code for that step; adapt it, do not copy it blindly.
The live observation is authoritative.

## When to use

The task needs the robot to pick up / grasp a named object that is resting on a
surface and is not yet held.

## Building blocks (compose these yourself)

You — the Coding Agent — decide how to combine these, in what order, and whether to
loop or skip, based on each leaf skill's own "When to use" / experience and the live
scene. There is NO fixed observe-then-act pipeline and NO mandatory hand-off contract
between them; just write the code that gets this object grasped.

- **segment_object** (observation) — ground `object_name` into a mask + filtered 3D
  world points; the usual starting point since a grasp needs to know where the object is.
- **grasp_object** (action) — derive a grasp pose from those points and execute it,
  confirming the hold with a small lift.

A typical composition is segment → grasp, but you are free to re-segment, retry the
grasp, or interleave observations as the scene calls for it.

## Postcondition

The named object is held in the closed gripper: gripper width is above zero and the
object lifts with the arm.

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: gripper width is above zero and the named object lifts with the arm.

## Failure modes

- Grasp missed (gripper fully closed): open the gripper, re-run segment + grasp
  candidates for a fresh IK-feasible pose, retry once.
- Wrong object grasped: re-segment with a more specific name (add color/position)
  before regenerating grasp candidates.
