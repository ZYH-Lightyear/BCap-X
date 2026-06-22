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
| high_level | 1 | `pick_object` (Pick Object) |
| observation | 1 | `segment_object` (Segment Object by Language) |
| action | 1 | `grasp_object` (Grasp Object) |

## Directories

Sidecars: `V` = reference/verify.md, `C` = scripts/verify.py.

- `[V-] action/grasp_object` — Derive a simple top-down grasp from an object's segmented 3D points, execute it, and confirm the hold by lifting slightly.
- `[V-] high_level/pick_object` — High-level grasp of a named object off a surface, orchestrating segmentation and the grasp action end-to-end.
- `[V-] observation/segment_object` — Ground a named object into a 2D mask and noise-filtered 3D world points (VLM localization then SAM3), the prerequisite for locating or grasping it.

