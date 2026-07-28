# RoboMEx Skill Library

Each skill is a self-contained package. The runtime loads `SKILL.md` through
progressive disclosure; optional sidecars are ordinary files resolved from the
loaded skill's base directory. Categories are not fixed execution phases or
SubAgent profiles.

## Package Structure

```text
<category>/<skill_id>/
├── SKILL.md       # compact workflow memory
├── references/    # optional explanatory material or non-runnable reference code
├── assets/        # optional visual/static assets; may intentionally be empty
└── scripts/       # optional runnable helper files with documented entry points
```

## Standard SKILL.md Shape

Every built-in skill should include Purpose, When to use, When NOT to use, Workflow, Candidate
Generation, Local Checks, Failure Modes, Clean Reusable Rules, Weak Priors,
Prohibited Shortcuts, Artifacts to Save, and Optional Sidecars when relevant.

## Inventory

| Category | Count | Skills |
|----------|------:|--------|
| perception | 2 | `estimate_object_geometry` (Estimate Object Geometry); `segment_object` (Segment Object by Language) |
| affordance | 6 | `compute_obb_short_axis_grasp` (Compute OBB Short Axis Grasp); `find_placement` (Find Placement); `grasp_graspnet` (GraspNet Candidate Grasp); `grasp_open_bowl` (Open Bowl Rim Grasp); `grasp_pca_side` (PCA / Side Grasp); `top_grasp_with_tcp_to_bottom_offset` (Top Grasp With TCP To Bottom Offset) |
| motion | 4 | `grasp_object` (Grasp Object); `plan_bounded_motion` (Plan Bounded Motion); `release_at` (Release At); `safe_return_home` (Safe Return Home) |
| verification | 2 | `verify_grasp_and_lift_via_robot_state` (Verify Grasp And Lift Via Robot State); `verify_placement` (Verify Placement) |
| task | 2 | `pick_object` (Pick Object); `place_object` (Place Object) |

## Directories

- `affordance/compute_obb_short_axis_grasp` — Compute a top-down grasp pose aligned to the short horizontal axis of an object's oriented bounding box.
- `affordance/find_placement` — GaP-style drop affordance: object center in zone + executable TCP pose.
- `affordance/grasp_graspnet` — Use Contact-GraspNet candidates as read-only 6-DoF grasp affordance evidence.
- `affordance/grasp_open_bowl` — Propose top-down rim or side-wall grasp candidates for an upward-facing open bowl.
- `affordance/grasp_pca_side` — Propose side or body grasp candidates from object point-cloud principal geometry.
- `affordance/top_grasp_with_tcp_to_bottom_offset` — Compute a top grasp and the TCP-to-object-bottom offset needed for later center-aware placement.
- `motion/grasp_object` — Execute a bounded grasp attempt from grounding and affordance evidence, with lift-state checking.
- `motion/plan_bounded_motion` — Convert one affordance into a canonical, bounded trajectory without changing the world.
- `motion/release_at` — Release a held object at a placement affordance, compensating off-center grasps when available.
- `motion/safe_return_home` — Retreat vertically before returning home to avoid sweeping through tabletop objects.
- `perception/estimate_object_geometry` — Summarize segmented 3D points into coarse shape, pose, dimensions, and grasp-relevant geometry hints.
- `perception/segment_object` — Ground a language-described object into compact bbox, mask, point-cloud, and artifact evidence.
- `task/pick_object` — Compose perception, affordance, motion, and state checks to pick up a named object.
- `task/place_object` — Compose placement affordance, release motion, and state checks to put a held object on or in a target.
- `verification/verify_grasp_and_lift_via_robot_state` — Fuse gripper width, end-effector lift state, external view, and wrist view to judge whether the target is held.
- `verification/verify_placement` — Independently verify the requested post-release object-target relation.
