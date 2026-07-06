---
name: PCA / Side Grasp
category: affordance
description: "Propose side or body grasp candidates from object point-cloud principal geometry."
---

# PCA / Side Grasp

## Purpose

Generate side/body grasp affordances from a segmented object's coarse 3D geometry. This
skill is useful when the contact should be around the body rather than straight down on
the top surface.

## When to use

Use after `segment_object` and usually after `estimate_object_geometry` when the object
is elongated, lying down, cylindrical, pushed over, or unstable under a top-down pinch.
Avoid it for open bowls/cups unless the rim-specific or GraspNet strategy is not
available.

Act runs this affordance analysis directly and keeps candidates local. Use the Verifier
SubAgent only to check a concrete visual alignment or diagnose a failed attempt.

## Workflow

1. Compute an OBB or PCA summary from the filtered target points.
2. Identify the long body axis and the shorter cross-body direction.
3. Place the grasp near the body center or slightly above mid-height, not at the very
   top surface.
4. Generate both side-approach directions and optionally a top-down fallback.
5. Filter by IK, save an overlay, and return the best compact candidate.

## Candidate Generation

- Try both signs of the side approach direction.
- Keep the grasp point inside the object's visible body extent.
- Use body height rather than raw `top_z` for lying objects.
- If PCA axes are unstable, fall back to simpler geometry or GraspNet.

## Local Checks

- The candidate should not point through the table, support surface, or container wall.
- The selected orientation must be IK-feasible before execution.
- If the object rolled or shifted during a previous attempt, recompute points first.
- The overlay should make the body contact and approach direction inspectable.

## Failure Modes

- Side pose unreachable: try the opposite side, reduce approach depth, or switch to
  GraspNet.
- Candidate misses body thickness: move toward mid-height or regenerate geometry.
- Wrong object moved: re-ground with a clearer referring expression.
- Object slips after lift: choose a deeper body contact or switch strategy.

## Clean Reusable Rules

- PCA/OBB axes are grasp hints, not ground truth.
- Do not repeat a failed pose without fresh observation or a strategy change.
- Body grasps need post-lift state checks just like top-down grasps.

## Weak Priors

- Lying bottles/cylinders and elongated objects often prefer body/side contact.
- Compact upright objects usually do not need this affordance first.

## Prohibited Shortcuts

- Do not hardcode yaw, side, or grasp height from one scene.
- Do not use PCA if the mask clearly contains background or multiple objects.

## Artifacts to Save

- Affordance overlay with selected side/body point, approach axis, jaw axis, and IK
  status.
- 3D review plot when projection is ambiguous or the candidate may be detached from
  the segmented body.
- Optional geometry summary when axis choice is ambiguous.
- Return actual image paths in `artifact_refs`; a directory path alone is not enough
  for Act to review the affordance.

## Optional Sidecars

No sidecar is required. Implement the PCA/OBB candidate generation directly with the
current segmented points and available geometry/overlay utilities such as
`robomex.perception.save_grasp_affordance_overlay` and
`robomex.perception.save_grasp_affordance_3d`.
