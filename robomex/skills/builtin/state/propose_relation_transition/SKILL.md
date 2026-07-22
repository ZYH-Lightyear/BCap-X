---
name: Propose Relation Transition
category: verification
description: "Convert final support evidence and verdict into a strict reducer proposal without committing state."
---

# Propose Relation Transition

## Purpose

Mechanically author a `supported_by(bowl, plate)=asserted` state proposal only
after the final synchronized observation, typed `RelationEvidence`, and
independent `PlacementVerdict` all agree.

## When to use

Use after final relation verification and before the runtime-owned relation
reducer.

## When NOT to use

Do not use before release/settle/retreat, for a failed or uncertain verdict, or
to infer support from execution completion alone.

## Workflow

1. Pass the complete observation, relation-evidence, and verdict `INPUTS`
   wrappers to the contracted helper.
2. Pass the provider-owned episode ID from `RUNTIME_CONTEXT_V1`.
3. The helper cross-checks predicate/value, subject/target/track identities,
   observation/camera/state revisions, action identity, verdict, and lineage.
4. Publish the exact wire proposal; finish with `result_var: NODE_RESULT`.

## Candidate Generation

There is one legal proposal: the evidence-declared `supported_by` assertion.
Never translate `on_top_of` wording into a different reducer predicate.

## Local Checks

Verdict status must be `succeeded`, relation `asserted`, and its bowl, target,
source observation, and state revision must equal the relation evidence. The
evidence must be `supported_by/asserted` for the expected tracked bowl and plate.

## Failure Modes

Use `failed_placement` for a negative verdict, `wrong_grounding` for identity or
track mismatch, and `stale_observation` for revision/action/lineage mismatch.
Never emit a proposal on those paths.

## Clean Reusable Rules

A final verifier proves a relation; this worker only adapts that proof to the
closed reducer schema. The reducer independently validates and commits it.

## Weak Priors

None. Geometric proximity or a successful retreat is not a relation proof.

## Prohibited Shortcuts

- Do not invent IDs, refs, digests, predicate values, or state revisions.
- Do not omit the independent verdict or transitive evidence lineage.
- Do not write state or call motion, gripper, perception, or backend APIs.
- Do not accept subject and target from free text.

## Artifacts to Save

Publish one strict `proposal`; the separate reducer activation publishes the
only authoritative state commit receipt.

## Multimodal Evidence Contract

The final primary checkpoint carries bowl/plate geometry. Relation evidence
binds the physical predicate to that checkpoint, and the verdict independently
states task success. All three are mandatory and must share causal identity.

## Reference Code

`build_relation_transition_proposal` from
`scripts/relation_transition.py` is already bound.

```python
proposal = build_relation_transition_proposal(
    INPUTS["observation"],
    INPUTS["relation_evidence"],
    INPUTS["verdict"],
    episode_id=RUNTIME_CONTEXT_V1["episode_id"],
)
NODE_RESULT = {
    "outputs": {"proposal": {"payload": proposal}},
    "control_outcome": "success",
}
```

Then finish exactly with
`{"tool":"finish","args":{"claim":"support relation proposed","result_var":"NODE_RESULT"}}`.
