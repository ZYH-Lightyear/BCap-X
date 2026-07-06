---
name: Place Object
category: task
description: "Compose placement affordance, release motion, and state checks to put a held object on or in a target."
---

# Place Object

## Purpose

Place the object currently held by the gripper into, onto, or relative to a named
target. This task skill guides composition; it does not impose a fixed planner phase or
fixed SubAgent type.

## When to use

Use when a pick has succeeded and the object is believed to be held. If holding state
is uncertain, check it first. Do not require a large gripper width: edge, rim, cup,
bowl, and plate grasps can be nearly closed while still valid.

## Workflow

1. Confirm the held-object belief is still plausible.
2. Ground the placement target with current observation and task language.
3. Estimate placement mode and desired object center with `find_placement`.
4. If an off-center grasp is known, preserve the held-object frame so `release_at` can
   compensate TCP position.
5. Execute a bounded release with `release_at`.
6. After retreat, perform a state-first check: at target, outside target, still held,
   not visible, or uncertain.

## Candidate Generation

- Use `open_container` reasoning for baskets, bins, bowls, cups, and other interior
  targets.
- Use `support_surface` reasoning for plates, trays, tabletops, cabinet tops, and
  stacking-style placements.
- Act computes placement affordance itself after loading `find_placement`; save a 2D
  or 3D artifact when the choice is not obvious.
- Use the Verifier SubAgent only for post-release state checks or failure diagnosis
  when the result is visually ambiguous.

## Local Checks

- The chosen target should match the task language and live scene.
- Desired placement is an object-center target; release motion may require TCP offset
  compensation.
- Post-release success is object-target relation plus empty/open gripper, not the
  existence of a segmentation mask.
- If the object appears already correctly placed, finish rather than regrasping due to
  stale beliefs.

## Failure Modes

- Placed off-center: check object-center versus TCP alignment and target center choice.
- Bounced/toppled: lower release height or choose a more stable target point after
  state check.
- Still held/stuck: open or retreat before any table re-localization.
- Wrong target: re-ground target with the full referring expression.

## Clean Reusable Rules

- Place is target-state driven: current-observation placement affordance, center-aware release, and
  post-release state check.
- Do not rely on cross-agent placed/held geometry. Re-check the current observation
  before finishing.
- Verifier outputs are verdicts only; Act executes motion and decides recovery.

## Weak Priors

- Top-down release is the default for many tabletop placements.
- Prior offsets and release heights may seed a candidate but should be checked against
  live geometry.

## Prohibited Shortcuts

- Do not hardcode target centers, pixels, or benchmark layouts.
- Do not regrasp before classifying the current state.
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
