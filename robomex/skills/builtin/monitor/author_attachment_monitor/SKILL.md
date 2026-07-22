---
name: Author Attachment Monitor
category: verification
description: "Author a compiler-checked online guard for held-object loss, occlusion, and identity swap."
---

# Author Attachment Monitor

## Purpose

Produce the sealed `robomex.monitor_program.v1` consumed in parallel with the
transport, correction, and descend actions. The monitor is code-shaped, but it
is not an actor with physical authority: the trusted runtime compiles it,
evaluates fresh control samples, and owns every stop request.

## When to use

Use once at the start of the bowl placement workflow when the node requests the
three graph-owned signals `attachment_status`, `held_entity_visible`, and
`identity_match`.

## When NOT to use

Do not poll a camera, infer attachment, move a sensor, execute a stop, or author
a monitor for undeclared signals. Monitoring evidence and state transitions are
separate nodes.

## Workflow

1. Read the versioned `NODE_CONFIG_V1` injected by the provider.
2. Pass its exact `allowed_signals` and `debounce_count` to the contracted helper.
3. The helper builds and compiles the restricted `evaluate(sample)` program.
4. Publish only `monitor_program`; finish with `result_var: NODE_RESULT`.

## Candidate Generation

There is exactly one audited baseline program. Do not make stylistic variants:
identity mismatch, loss of visibility, and loss of `verified_held` each produce
a critical finding; an ordinary sample returns `None`.

## Local Checks

Require the control hook, the exact three-signal ordered tuple, a debounce count
in `[1, 100]`, no imports/loops/I/O in monitor source, and successful
`MonitorCompiler` validation.

## Failure Modes

Return `infeasible` without an output if the graph config asks for any other
signal set or invalid debounce count. Missing/malformed samples fail closed at
runtime; do not add recovery logic to the authored program.

## Clean Reusable Rules

A monitor proposes a deterministic predicate over declared samples. It never
produces perceptual truth and never changes the world or embodied state.

## Weak Priors

One anomalous sample is normally sufficient (`debounce_count=1`). A larger
graph-owned count may suppress noisy detections but must never be invented here.

## Prohibited Shortcuts

- Do not hard-code a different program or signal name.
- Do not call perception, motion, gripper, backend, file, or process APIs.
- Do not treat action completion as proof that the object remained attached.
- Do not overwrite `RUNTIME_CONTEXT_V1` or `NODE_CONFIG_V1`.

## Artifacts to Save

Publish one `monitor_program` payload. The data plane and action receipt retain
its content digest and execution lineage; no free-form side artifact is needed.

## Multimodal Evidence Contract

This node consumes no observation artifact. It declares which runtime-provided
signals later action samples must contain. Absence, non-finite data, oracle-like
fields, or an identity/visibility/attachment anomaly is handled fail-closed by
the monitor runtime.

## Reference Code

`build_attachment_monitor_program` is already bound from
`scripts/attachment_monitor.py`; call it directly. The versioned node config is
provider-owned and cannot be supplied through an artifact port.

```python
program = build_attachment_monitor_program(
    allowed_signals=NODE_CONFIG_V1["allowed_signals"],
    debounce_count=NODE_CONFIG_V1["debounce_count"],
)
NODE_RESULT = {
    "outputs": {"monitor_program": {"payload": program}},
    "control_outcome": "success",
}
```

Then finish exactly with
`{"tool":"finish","args":{"claim":"attachment monitor compiled","result_var":"NODE_RESULT"}}`.
