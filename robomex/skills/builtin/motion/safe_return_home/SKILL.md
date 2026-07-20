---
name: Safe Return Home
category: motion
description: "Retreat vertically before returning home to avoid sweeping through tabletop objects."
---

# Safe Return Home

## Purpose

Move the arm back to the home joint configuration through a safe vertical retreat,
reducing the chance of knocking objects while exiting a manipulation attempt.

## When to use

Use after a failed attempt, after release, or before re-observation when the arm is low
near the table. If the robot is holding an object, first decide whether returning home
would preserve or endanger the held state.

## When NOT to use

Do not return home from a low pose without a vertical clearance move, open the
gripper as part of recovery, or use home motion as evidence that the task succeeded.

## Workflow

1. Capture current end-effector pose from fresh observation.
2. Build a safe pose by keeping x/y/orientation and raising z to a configured retreat
   height.
3. Move to the safe pose.
4. Return to home joint configuration.
5. Re-observe before planning the next manipulation step.

## Candidate Generation

This skill generates a safe retreat waypoint, not a manipulation target. The waypoint
should be high enough to clear nearby objects but not so high that IK becomes unstable.

## Local Checks

- Do not open the gripper as part of this skill.
- If holding an object, check clearance before sweeping to home.
- Use current pose and current orientation rather than a task-specific coordinate.
- If vertical retreat is infeasible, stop and re-plan rather than moving laterally low.

## Failure Modes

- IK fails for high retreat: lower the retreat height slightly or move in smaller
  increments.
- Held object collides during home motion: retreat higher or choose a safer staging
  pose.
- Camera view remains occluded: re-run observation after home and choose another view.

## Clean Reusable Rules

- Before long lateral motion near a cluttered table, move upward first.
- Safe reset is part of closed-loop recovery, not task success.
- Returning home should preserve evidence that a failure occurred.

## Weak Priors

- A z height around the upper workspace is usually safer than the current low pose.
- Home view often improves perception after cluttered manipulation.

## Prohibited Shortcuts

- Do not call home from a low table pose when objects are close.
- Do not release a held object unless the current plan explicitly says to.
- Do not treat returning home as verification of success.

## Artifacts to Save

- Log current pose, retreat pose, whether the gripper was believed to be holding
  something, and whether the move succeeded.

## Multimodal Evidence Contract

Consume a fresh robot pose and holding-state belief. Publish retreat/home primitive
status plus runtime-captured terminal state. On either primitive failure, stop and
emit `execution_fault`; do not alter gripper state.

## Optional Sidecars

No sidecar is required. Implement with fresh observation, one vertical retreat pose,
and the environment home primitive.
