---
name: Pick Object
category: task
description: "Compose perception, affordance, motion, and state checks to pick up a named object."
---

# Pick Object

## Purpose

Pick a named object from a surface. This skill is progressively loaded as composition
guidance: the Manager must generate a graph that fits the live scene rather than load a
predefined sequence.

## When to use

Use when the task requires the robot to grasp an object that is not currently held. If
the object may already be held, first perform a state check instead of re-localizing it
on the table.

## When NOT to use

Do not compose a pick graph when the target is already verified as held, when the
requested postcondition is only placement, or when no world-changing action is
authorized.

## Strategy Selection

| Live evidence | Preferred affordance | Why |
|---|---|---|
| Compact, upright, closed top | `top_grasp_with_tcp_to_bottom_offset` | Interpretable canonical top grasp |
| Open upward-facing bowl or cup | `grasp_open_bowl` | Uses observed rim support |
| Elongated, lying, or top-unstable body | `grasp_pca_side` | Aligns contact to body geometry |
| Irregular, cluttered, or repeated heuristic miss | `grasp_graspnet` | Ranks learned 6-DoF candidates |

## Workflow

1. Establish current state: target not held, visible or recoverable, and gripper ready.
2. Add Grounding when fresh target evidence is required.
3. Add geometry analysis only when shape or pose affects strategy, then choose an
   Affordance specialist:
   - `top_grasp_with_tcp_to_bottom_offset` for compact upright objects;
   - `grasp_open_bowl` for open upward-facing bowls;
   - `grasp_pca_side` for elongated/lying objects;
   - `grasp_graspnet` for complex or uncertain cases.
4. Add a MotionPlanner to convert the selected affordance into a feasible trajectory without
   changing robot state.
5. Add an ActionExecutor to execute one bounded trajectory with `grasp_object`.
6. End with an independent Verifier checking fresh post-lift state. Only its hard passed report
   can satisfy the graph's success exit.

## Candidate Generation

- Grounding and Affordance generate only their declared typed outputs.
- MotionPlanner ranks executable candidates but cannot move the robot.
- ActionExecutor consumes the selected trajectory; it cannot segment or replan.
- Verifier independently checks whether the object is held after lift.

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

### `failed_grasp` recovery: bounded offsets and strategy change

Treat each retry as one changed physical hypothesis. Re-observe first; never run
several blind offsets in one executor call. For a plausible grasp family with
uncertain contact depth, vary one bounded depth or XY offset, execute one attempt,
lift, and verify. Persist the attempted offset and verdict. Repeated
`still_on_surface` should change depth or grasp family; `empty_or_slipped` should
move contact toward the observed body; slippage after lift should prefer a deeper
body/rim grasp. If the verdict is `wrong_object`, route `wrong_grounding` and
re-ground rather than adjusting geometry.

### `infeasible` and `execution_fault` recovery

`infeasible` routes from motion planning back to a different affordance candidate.
`execution_fault` must stop execution and route to replanning from the actual robot
state. Neither event permits the executor to invent a replacement pose.

## Multimodal Evidence Contract

- Grounding consumes a fresh observation epoch and publishes a mask/box overlay plus
  world-frame points.
- Affordance consumes those same-epoch points and publishes a grasp overlay for
  nontrivial candidates.
- Motion planning consumes typed geometry only; it does not refresh perception.
- Execution publishes primitive status and runtime-captured terminal robot state.
- Verification must capture a newer post-action observation epoch and publish the
  selected scene/wrist evidence. Stale evidence routes `stale_observation`.

## Clean Reusable Rules

- Pick is evidence-driven: current-observation grounding, local affordance evidence,
  bounded execution, and lift-state check.
- Skills are workflow memory. They explain how to reason and what to save, not a
  benchmark answer script.
- Cross-agent geometry moves only through typed `$ref` bindings in ArtifactStore.

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
