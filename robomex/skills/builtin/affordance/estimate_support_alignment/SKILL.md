---
name: Estimate Support Alignment
category: affordance
description: "Measure held-bowl bottom alignment to a plate support region from one synchronized checkpoint."
---

# Estimate Support Alignment

## Purpose

Turn one admitted bowl/plate checkpoint plus typed attachment evidence into a
measured `AlignmentError`. At the normal servo checkpoint the node also republishes
the validated held-bowl and plate target geometry; at the pre-release checkpoint
it emits only the error requested by that node.

## When to use

Use after the deterministic attachment verifier confirms `verified_held` and
before the deterministic servo or pre-release gate makes a control decision.

## When NOT to use

Do not poll sensors, mix observations, choose a correction, run IK, author a
trajectory, or estimate an unobserved bowl/TCP offset.

## Workflow

1. Read `observation` and `attachment_evidence` from `INPUTS`.
2. Call the contracted deterministic helper with graph-owned tolerances.
3. The helper validates entity, visibility, frame, snapshot, generation, full
   revision vector, attachment status, support containment, and uncertainty.
4. Select only the output ports requested for this phase.
5. Finish with `result_var: NODE_RESULT`.

## Candidate Generation

There are no action candidates. The single candidate is the exact
target-minus-held measurement derived from admitted geometry.

## Local Checks

Both geometry records must be visible and share frame, snapshot ID, generation,
and revisions. Attachment evidence must name the expected bowl and the same
camera revision. Yaw is wrapped to `[-pi, pi]`; containment uses the plate safety
margin; translation and orientation uncertainty are propagated.

## Failure Modes

Use `uncertain` for unavailable geometry, `wrong_grounding` for identity mismatch,
and `stale_observation` for mixed snapshot/revision evidence. Never fabricate a
successful alignment payload on those paths.

## Clean Reusable Rules

Geometry estimation reports what is measured. The deterministic gate owns
tolerance/cumulative-budget decisions, and the motion author owns a later sealed
joint path.

## Weak Priors

The graph's tolerances are defaults, not permission to smooth away a measured
error. Fresh typed geometry always dominates a prior pose.

## Prohibited Shortcuts

- Do not use a fixed TCP-to-bowl offset or hard-coded correction direction.
- Do not compare geometry from different snapshots or camera revisions.
- Do not call motion, gripper, active-perception, or simulator-oracle APIs.
- Do not emit an executable action from this node.

## Artifacts to Save

Publish `alignment_error` and, outside `pre_release`, the exact admitted
`held_bowl` and `plate_target` payloads. Their data-plane lineage is attached by
the provider.

## Multimodal Evidence Contract

`observation` is the synchronized primary checkpoint. `attachment_evidence` is
an independent typed predicate bound to that checkpoint and state revision. Both
are mandatory; free-text claims and action success are not alignment evidence.

## Reference Code

`estimate_support_alignment` from `scripts/support_alignment.py` is already
bound. It returns strict schema payloads and raises on mixed evidence.

```python
measured = estimate_support_alignment(
    INPUTS["observation"]["payload"],
    INPUTS["attachment_evidence"]["payload"],
    tolerance_xy_m=NODE_CONFIG_V1["tolerance_xy_m"],
    tolerance_z_m=NODE_CONFIG_V1["tolerance_z_m"],
    tolerance_yaw_rad=NODE_CONFIG_V1["tolerance_yaw_rad"],
)
outputs = {"alignment_error": {"payload": measured["alignment_error"]}}
if NODE_CONFIG_V1.get("phase") != "pre_release":
    outputs["held_bowl"] = {"payload": measured["held_bowl"]}
    outputs["plate_target"] = {"payload": measured["plate_target"]}
NODE_RESULT = {"outputs": outputs, "control_outcome": "success"}
```

Then finish exactly with
`{"tool":"finish","args":{"claim":"synchronized alignment measured","result_var":"NODE_RESULT"}}`.
