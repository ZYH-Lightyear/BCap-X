# RoboMEx Skill Library

Each skill is a self-contained package, split by *reader*: `SKILL.md` for the
executor agent, `reference/` for the Verifier Agent, `scripts/` for deterministic code.

## Package Structure

```text
<category>/<skill_id>/
├── SKILL.md       # executor agent: when-to-use, procedure/decomposition, recovery
├── reference/     # Verifier Agent: verify.md rubric (+ optional visual references)
└── scripts/       # optional: deterministic verifier-as-code (verify.py)
```

## Inventory

| Category | Count | Skills |
|----------|------:|--------|
| high_level | 2 | `pick_object` (Pick Object); `place_object` (Place Object) |
| observation | 2 | `find_placement` (Find Placement); `segment_object` (Segment Object by Language) |
| action | 2 | `grasp_object` (Grasp Object); `release_at` (Release At) |

## Directories

Sidecars: `V` = reference/verify.md, `C` = scripts/verify.py.

- `[V-] action/grasp_object` — Derive a simple top-down grasp from an object's segmented 3D points, execute it, and confirm the hold by lifting slightly.
- `[V-] action/release_at` — Release a held object at a given 3D place position by moving above it, lowering with clearance, opening the gripper, and retreating.
- `[V-] high_level/pick_object` — High-level grasp of a named object off a surface, orchestrating segmentation and the grasp action end-to-end.
- `[V-] high_level/place_object` — High-level placement of a held object into/onto a named target, orchestrating placement localization and the release action end-to-end.
- `[V-] observation/find_placement` — Ground a named placement target (container / region / surface) and compute a safe 3D release point above it, the prerequisite for placing a held object.
- `[V-] observation/segment_object` — Ground a named object into a 2D mask and noise-filtered 3D world points (VLM localization then SAM3), the prerequisite for locating or grasping it.

