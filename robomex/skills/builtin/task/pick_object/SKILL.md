---
name: Pick Object
category: task
description: "Compose perception, affordance, motion, and state checks to pick up a named object."
---

# Pick Object

## Purpose

Pick a named object from a surface. This task skill guides Act's composition of leaf
skills. It is not a fixed pipeline and not a SubAgent profile.

## When to use

Use when the task requires the robot to grasp an object that is not currently held. If
the object may already be held, first perform a state check instead of re-localizing it
on the table.

## Workflow

1. Establish current state: target not held, visible or recoverable, and gripper ready.
2. Ground the target object with `segment_object`. If identity is ambiguous, Act should
   enumerate candidates and use crops/VLM state checks inside its own code.
3. Estimate geometry when the grasp family is not obvious.
4. Obtain grasp affordance evidence:
   - simple top-down from `grasp_object` for compact upright objects;
   - `grasp_open_bowl` for open upward-facing bowls;
   - `grasp_pca_side` for elongated/lying objects;
   - `grasp_graspnet` for complex or uncertain cases.
5. Execute one bounded grasp attempt with `grasp_object`.
6. After lift, check whether the named object is held. Finish only when the pick
   postcondition is satisfied or when returning a clear failure/uncertainty.

## Candidate Generation

- Act generates grounding and grasp candidates itself after loading the relevant
  skills. Keep candidate lists local in `EVIDENCE` or artifacts.
- Use the Verifier SubAgent only for state checks or failure diagnosis, for example
  whether the object is actually held after lift or why the last attempt missed.
- Act remains responsible for selecting the next code block and executing robot motion.

## Local Checks

- Grounding should identify the named object, not a neighbor or target container.
- Grasp strategy should match the live geometry, not only the object class.
- Post-lift visual state should check that the named object moves with the gripper.
- If the model only printed observation structure or shapes, that did not advance the
  physical task; load a relevant skill, run focused Act analysis, or execute a bounded
  action next.

## Failure Modes

- Wrong object grounded: sharpen the target expression and enumerate candidates.
- Top-down miss or push: re-observe and switch depth/strategy.
- Object pose changed: recompute geometry and avoid reusing stale points.
- Thin rim or edge grasp looks closed: inspect visual holding state before declaring
  failure.

## Clean Reusable Rules

- Pick is evidence-driven: current-observation grounding, local affordance evidence,
  bounded execution, and lift-state check.
- Skills are workflow memory. They explain how to reason and what to save, not a
  benchmark answer script.
- Compact artifacts are for Act review and debugging; do not promote geometry as
  cross-agent state.

## Weak Priors

- Object family can suggest an initial grasp strategy, but live geometry and prior
  attempt outcomes dominate.
- Known common confusions can guide candidate review, but should not be encoded as
  fixed layout rules.

## Prohibited Shortcuts

- Do not hardcode target coordinates, pixels, object order, or seed layouts.
- Do not blindly repeat a failed action without changed evidence.
- Do not use gripper width as the sole success criterion.

## Artifacts to Save

- Grounding bbox/mask overlay.
- Grasp candidate overlay for nontrivial picks.
- Attempt record with selected strategy and post-lift state.

## Completion Signal

The object is visibly coupled to the gripper after lift and no longer resting on the
original support. If uncertain, return uncertainty with evidence instead of pretending
the pick completed.
