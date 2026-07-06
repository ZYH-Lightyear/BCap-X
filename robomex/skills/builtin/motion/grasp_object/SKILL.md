---
name: Grasp Object
category: motion
description: "Execute a bounded grasp attempt from grounding and affordance evidence, with lift-state checking."
---

# Grasp Object

## Purpose

Execute one grasp attempt for a named object. This motion skill turns grounding and
affordance evidence into robot motion, then checks whether the object is actually held.

## When to use

Use when the gripper is empty, the target object is grounded, and Act has chosen a
grasp from its current local perception/affordance analysis. If grasp strategy is
uncertain, run more perception/affordance analysis before executing motion.

## Workflow

1. Choose the grasp family from live geometry and evidence:
   - simple top-down for compact upright objects;
   - open-bowl rim for upward-facing bowls/open containers;
   - PCA/side for elongated or lying objects;
   - GraspNet for complex, cluttered, or uncertain shapes.
2. Save or reference a candidate overlay before moving when the grasp is nontrivial.
3. Open the gripper, approach with modest clearance, descend to the grasp pose, and
   close.
4. Lift a short distance before doing any transport.
5. Use fresh observations to judge whether the named object is held and moving with the
   gripper.

## Candidate Generation

- For obvious compact objects, Act may create a top-down candidate from the object
  center and top surface.
- For nontrivial geometry, use a dedicated affordance skill and save an artifact before
  motion. Use the Verifier SubAgent only to check a concrete visual alignment or
  post-lift state claim.
- Do not keep trying arbitrary quaternions in the motion block; if candidate generation
  is unclear, return to affordance analysis.

## Local Checks

- IK should be checked before executing the pose when possible.
- Approach clearance should be enough to avoid collisions but not so large that the
  simple interpolator drifts.
- After lift, judge visual coupling between object and gripper. Gripper width is a weak
  supporting signal only.
- For thin rims, bowls, cups, plates, and edge grasps, a nearly closed gripper can still
  be a valid hold.

## Failure Modes

- Object remains on the surface: re-open, re-observe, and change depth or strategy.
- Wrong object was grasped: stop transport, re-ground the target, and avoid reusing the
  same ambiguous mask.
- Object rolled or changed pose: recompute geometry before retrying.
- Object slips during lift: choose a deeper/body/rim contact or switch to GraspNet.

## Clean Reusable Rules

- Execute one bounded grasp attempt, then inspect state. Do not write a long blind
  retry script without observations.
- A post-lift visual state check is stronger than gripper-width thresholding.
- Act owns motion and candidate selection. The Verifier SubAgent may only check state
  or diagnose an attempt.

## Weak Priors

- Compact upright cans/boxes often tolerate simple top-down.
- Open/concave objects often need rim-specific or GraspNet candidates.
- A previous failure can suggest the next strategy but should not override fresh
  observations.

## Prohibited Shortcuts

- Do not hardcode grasp coordinates, pixels, object poses, or seed-specific locations.
- Do not repeat a failed pose without a changed observation or changed strategy.
- Do not call `goto_home_joint_position` in a way that opens/releases the gripper while
  holding an object.

## Artifacts to Save

- Affordance overlay for nontrivial grasps.
- Short attempt record: strategy, selected pose summary, whether lift check passed, and
  any failure reason.

## Optional Sidecars

No sidecar is required. Use the motion primitives available in the sandbox and the
selected candidate evidence from perception/affordance skills.
