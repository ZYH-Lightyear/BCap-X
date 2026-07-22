---
name: Propose Attachment Transition
category: verification
description: "Convert admitted attachment evidence into a strict typed reducer proposal without committing state."
---

# Propose Attachment Transition

## Purpose

Mechanically adapt synchronized `AttachmentEvidence` into
`robomex.state_transition_proposal.v1`. This node proposes either
`verified_held` or `not_held`; the runtime-owned reducer remains the only state
writer.

## When to use

Use immediately after the deterministic attachment verifier and before the
corresponding reducer activation.

## When NOT to use

Do not infer attachment from an execution receipt, create an `attempted` or
`unknown` transition, repair stale evidence, or commit state directly.

## Workflow

1. Pass the complete `INPUTS` wrappers—not only payloads—to the helper so their
   content-addressed refs and admitted causal lineage are preserved.
2. Pass only provider-owned `RUNTIME_CONTEXT_V1["episode_id"]` as the episode ID.
3. The helper validates observation/evidence identity and constructs the closed
   domain proposal plus exact wire payload.
4. Publish only `proposal` and finish with `result_var: NODE_RESULT`.

## Candidate Generation

There is exactly one legal transition: the determinate status stated by the
typed evidence. Do not generate alternatives.

## Local Checks

Require matching bowl entity, track, snapshot ID, camera revision, attachment
status, action ID, base state revision, observation ref, evidence ref, and
evidence-artifact lineage. Duplicate refs are removed by content identity.

## Failure Modes

Use `wrong_grounding` for entity/track mismatch and `stale_observation` for any
observation, revision, lineage, action, or state-revision mismatch. A failed
proposal must emit no artifact.

## Clean Reusable Rules

Generated code may choose whether evidence is usable; it may not author the
authoritative episode envelope or mutate state. The strict wire model and reducer
revalidate every field.

## Weak Priors

None. Only the admitted typed evidence is authoritative for the proposed value.

## Prohibited Shortcuts

- Do not invent `episode_id`, `before_revision`, refs, digests, action IDs, or tracks.
- Do not strip transitive evidence lineage.
- Do not call perception, geometry, motion, gripper, or state-write APIs.
- Do not use an executor success string as attachment evidence.

## Artifacts to Save

Publish one `proposal` payload. The provider binds its artifact lineage to all
resolved inputs; the reducer produces the durable commit receipt separately.

## Multimodal Evidence Contract

The observation supplies the synchronized physical envelope. The typed evidence
supplies the determinate attachment predicate and causal action. Their identities,
revisions, and lineage must agree exactly.

## Reference Code

`build_attachment_transition_proposal` from
`scripts/attachment_transition.py` is already bound and returns a validated
`StateTransitionProposalWire` mapping.

```python
proposal = build_attachment_transition_proposal(
    INPUTS["observation"],
    INPUTS["attachment_evidence"],
    episode_id=RUNTIME_CONTEXT_V1["episode_id"],
)
NODE_RESULT = {
    "outputs": {"proposal": {"payload": proposal}},
    "control_outcome": "success",
}
```

Then finish exactly with
`{"tool":"finish","args":{"claim":"attachment transition proposed","result_var":"NODE_RESULT"}}`.
