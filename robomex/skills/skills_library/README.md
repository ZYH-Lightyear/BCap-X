# RoboMEx Skill Library

Each skill is a self-contained package, split by *reader*: `SKILL.md` for the
executor agent, `ref/` for the Verifier Agent, `scripts/` for deterministic code.

## Package Structure

```text
<category>/<skill_id>/
├── SKILL.md       # executor agent: when-to-use, procedure/decomposition, recovery
├── ref/           # Verifier Agent: verify.md rubric (+ optional visual references)
└── scripts/       # optional: deterministic verifier-as-code (verify.py)
```

## Inventory

| Category | Count | Skills |
|----------|------:|--------|
| high_level | 2 | `pick_object` (Pick Object); `place_in_container` (Place Held Object In Container) |
| observation | 3 | `estimate_geometry` (Estimate Object Geometry); `grasp_candidates` (Generate and Vet Grasp Candidates); `segment_object` (Segment Object by Language) |
| action | 2 | `grasp_object` (Grasp Object); `place_into_container` (Place Held Object Into Container) |

## Directories

Sidecars: `V` = ref/verify.md, `C` = scripts/verify.py.

- `[V-] action/grasp_object` — Approach a vetted grasp pose from above, close the gripper, and confirm the object is actually held by lifting slightly.
- `[V-] action/place_into_container` — Carry the held object over a container and release it inside, with the drop height computed from measured geometry instead of guessed constants.
- `[V-] high_level/pick_object` — High-level grasp of a named object off a surface, orchestrating perception (segment, geometry, grasp candidates) and the grasp action end-to-end.
- `[V-] high_level/place_in_container` — High-level placement of an already-held object into a named container, orchestrating container perception (segment + geometry) and the release action with a measured drop height.
- `[V-] observation/estimate_geometry` — Turn filtered 3D points into a verified size/position estimate (center, extents, top/bottom z) so heights and clearances are computed, never guessed.
- `[V-] observation/grasp_candidates` — Get GraspNet candidates, prefer a top-down aligned one, and prove IK feasibility BEFORE any motion. Never trust the top-1 raw candidate.
- `[V-] observation/segment_object` — Ground a named object into a 2D mask and noise-filtered 3D world points (SAM3 text prompt), the prerequisite for any geometric reasoning.
