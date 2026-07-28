---
name: Place Object
category: task
description: "Compose placement affordance, release motion, and state checks to put a held object on or in a target."
---

# Place Object

## Purpose

Place the object currently held by the gripper into, onto, or relative to a named
target. This progressively loaded task skill guides dynamic composition; it does not
own a predefined node sequence or recovery graph.

## When to use

Use when a pick has succeeded and the object is believed to be held. If holding state
is uncertain, check it first. Do not require a large gripper width: edge, rim, cup,
bowl, and plate grasps can be nearly closed while still valid.

## When NOT to use

Do not compose placement while grasp state is unverified, while the requested
object is visibly not held, or when the task asks only for a pick.

## Strategy Selection

| Target/evidence | Recommended graph choice | Why |
|---|---|---|
| Open container with clear cavity | `find_placement` with `open_container` relation | Keeps object center inside the cavity |
| Plate, tray, shelf, or support | `find_placement` with `support_surface` relation | Uses support height and footprint |
| Off-center grasp with known held frame | Preserve `held_object_frame` through affordance | Compensates TCP-to-object-center offset |
| Occluded or shifted target | Re-ground before affordance | Prevents stale target-center motion |

## Workflow

1. Confirm the held-object belief is still plausible; keep `held_object_frame` available.
2. Ground the placement target with current observation and task language.
3. Affordance runs `find_placement` / `compute_drop_affordance` so typed
   `placement_affordance.position` is the TCP drop pose and `desired_object_center`
   is object-state evidence inside the zone.
4. MotionPlanner builds the fixed place template via `build_place_trajectory`
   (transport → descend to TCP → open/settle → retreat) and IK-checks it without
   changing state.
5. ActionExecutor executes only that trajectory with `release_at`, opening only after
   descend converges and settling before retreat.
6. An independent Verifier performs a fresh state-first check: at target, outside
   target, still held, not visible, or uncertain.

## Candidate Generation

- Use `open_container` reasoning for baskets, bins, bowls, cups, and other interior
  targets (object center inside the cavity, not above the rim).
- Use `support_surface` reasoning for plates, trays, tabletops, cabinet tops, and
  stacking-style placements.
- Affordance publishes the placement choice as a typed artifact.
- MotionPlanner owns trajectory feasibility; ActionExecutor cannot redo geometry.
- Verifier owns the hard post-release verdict.

## Local Checks

- The chosen target should match the task language and live scene.
- Desired placement is an object-center target; the typed affordance `position` is the
  TCP after offset compensation.
- Post-release success is object-target relation plus empty/open gripper, not the
  existence of a segmentation mask.
- If the object appears already correctly placed, finish rather than regrasping due to
  stale beliefs.

## Failure Modes

- Placed off-center: check object-center versus TCP alignment and target center choice.
- Bounced/toppled: lower release height via affordance recompute, not executor inventing
  heights.
- Still held/stuck: open or retreat before any table re-localization.
- Wrong target: re-ground target with the full referring expression.

### `failed_placement` recovery: hover, re-observe, align

When the target center or held-object offset is uncertain, compose a fresh
perception/affordance cycle before release: observe at safe hover, re-ground the
support region, recompute the object-center-to-TCP offset, then plan a small bounded
correction. Large disagreement routes `wrong_grounding`; it is not a large XY
correction. Descend only after the corrected hover converges, open at the canonical
release TCP, settle, retreat vertically, and verify.

If the object remains held, open/retreat safely before re-segmentation. If it
bounces or tips, route back to placement affordance to lower release geometry; the
executor must not invent a new height.

## Multimodal Evidence Contract

- Target grounding uses a current observation epoch and publishes a mask/box overlay.
- Placement affordance publishes desired object center, executable TCP, held-frame
  compensation, and an inspectable overlay.
- Motion planning consumes the typed affordance and emits the canonical place
  trajectory without changing the world.
- Execution publishes primitive evidence and terminal robot state.
- Verification uses a newer post-release observation epoch and classifies at target,
  misplaced, still held, wrong object, or uncertain. Stale evidence cannot certify
  success.

## Clean Reusable Rules

- Place is target-state driven: current-observation placement affordance, center-aware
  TCP, settle-before-retreat, and post-release state check.
- Cross-agent geometry moves only through typed `$ref` bindings.
- Verifier outputs are hard verdicts; recovery follows only validated edges generated
  for the current subgoal.

## Weak Priors

- Top-down release is the default for many tabletop placements.
- Prior offsets and release heights may seed a candidate but should be checked against
  live geometry.

## Prohibited Shortcuts

- Do not hardcode target centers, pixels, or benchmark layouts.
- Do not regrasp before classifying the current state.
- Do not treat `desired_object_center` as the motion TCP.
- Do not ask `query_vlm` for coordinates, boxes, masks, or points during verification.

## Artifacts to Save

- Placement target overlay and optional 3D visualization.
- Release attempt record with desired object center, TCP compensation summary, and
  post-release state.
- Diagnostic artifact if the object is outside the target or still held.

## Completion Signal

The named object rests in/on the named target and the gripper is open/empty. If the
state is uncertain, return the uncertainty and visible evidence instead of escalating
to a blind recovery script.
