# RoboMEx Skill Library

Each skill is a self-contained package. The runtime loads `SKILL.md` through
progressive disclosure; optional sidecars are ordinary files resolved from the loaded
skill's base directory. Categories organize the library for humans and menus. They are
not fixed execution phases, SubAgent profiles, or routing rules.

## Package Structure

```text
<category>/<skill_id>/
├── SKILL.md       # compact workflow memory
├── references/    # optional explanatory material or non-runnable reference code
├── assets/        # optional visual/static assets; may intentionally be empty
└── scripts/       # optional runnable helper files with documented entry points
```

## Standard SKILL.md Shape

Every built-in skill should use the same short workflow-memory shape:

- `Purpose`
- `When to use`
- `Workflow`
- `Candidate Generation`
- `Local Checks`
- `Failure Modes`
- `Clean Reusable Rules`
- `Weak Priors`
- `Prohibited Shortcuts`
- `Artifacts to Save`
- `Optional Sidecars` when the package has helper files

Keep the main file concise. Long code templates, object prompt registries, benchmark
case notes, and failed-trial logs belong in `references/` or trace artifacts, not in
the main `SKILL.md`.

## Runtime Memory Contract

`EVIDENCE` is local scratchpad memory inside the current Act/sub-goal or Verifier
call. Use it for masks, point clouds, candidate lists, and intermediate variables that
should not be copied into prompts. Geometry produced by a skill is local working
memory, not promoted world state. Cross-turn communication should stay compact:

- `artifact_refs` for overlays, masks, videos, or 3D review images.
- `primitive_traces` for action or sidecar execution summaries.
- `attempt_records` for tried strategies and outcomes.
- verifier `local_verdict` / `diagnoses` for focused checks and failure attribution.

Do not use `state_patch` to promote bbox, masks, points, grasp poses, or placement
poses into persistent control context.

## Skill Design Rules

Skills are workflow memory, not a callable API catalog. A good skill records how to
generate candidates, how to check them locally, which failure modes matter, which weak
priors are acceptable, and which shortcuts are forbidden.

Sidecar scripts are optional convenience code. Prefer the entry points and usage
patterns named in `SKILL.md`; do not spend agent turns printing full source before
trying a documented call. If a script is not directly runnable in the live sandbox,
put it under `references/` instead of `scripts/`.

Fixed pixel thresholds, seed-specific coordinates, object order tables, and benchmark
answer sheets must not become default skill rules. If a past run suggests a useful
lesson, rewrite it as a clean rule, weak prior, or shortcut rejection before admitting
it into a skill.

## Inventory

| Category | Count | Skills |
|----------|------:|--------|
| perception | 2 | `estimate_object_geometry` (Estimate Object Geometry); `segment_object` (Segment Object by Language) |
| affordance | 4 | `find_placement` (Find Placement); `grasp_graspnet` (GraspNet Candidate Grasp); `grasp_open_bowl` (Open Bowl Rim Grasp); `grasp_pca_side` (PCA / Side Grasp) |
| motion | 2 | `grasp_object` (Grasp Object); `release_at` (Release At) |
| task | 2 | `pick_object` (Pick Object); `place_object` (Place Object) |

## Directories

- `perception/segment_object` — Ground a language-described object with dedicated bbox/point/SAM3 APIs, categorical VLM checks, and compact bbox/mask/point evidence.
- `perception/estimate_object_geometry` — Summarize segmented points into coarse shape, pose, and dimensions.
- `affordance/find_placement` — Estimate object-center placement affordance for open containers and support surfaces.
- `affordance/grasp_open_bowl` — Propose top-down rim or side-wall grasp candidates and held-object center offsets for upward-facing bowls.
- `affordance/grasp_pca_side` — Propose side/body grasps from PCA or OBB geometry.
- `affordance/grasp_graspnet` — Use Contact-GraspNet as read-only 6-DoF candidate evidence with IK and artifact checks.
- `motion/grasp_object` — Execute one bounded grasp attempt and check post-lift state.
- `motion/release_at` — Execute center-aware release and post-release state checking.
- `task/pick_object` — Compose grounding, affordance, grasp execution, and lift-state checks for picking.
- `task/place_object` — Compose placement affordance, center-aware release, and object-target state checks for placing.
