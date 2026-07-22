---
name: Author Sealed Phase Motion
category: motion
description: "Plan and seal exactly one v2 arm phase as an immutable exact joint path without executing it."
---

# Author Sealed Phase Motion

## Purpose

Produce one `robomex.motion_plan.v2` for exactly one graph phase:
`transport_to_safe_hover`, `bounded_correction`, `descend_to_release`, or
`safe_retreat`. IK/collision planning happens here once; the trusted action
runner later executes the sealed joint samples without another IK solve.

## When to use

Use only after a runtime-owned `robomex.admission_snapshot.v1` has been captured
for the arm and the phase-specific evidence/guard inputs have been admitted.

## When NOT to use

Do not combine phases, open/close the gripper, settle, verify attachment, poll a
new observation, or execute a trajectory. Do not reuse this plan with another
snapshot.

## Workflow

1. Read the exact arm snapshot and graph-owned `plan_kind`.
   Read the runtime-owned `robot_model_digest` pin from the same closed node
   config; never infer it from the snapshot configuration digest.
2. Validate phase evidence: a verified attachment guard for held phases; a
   bounded servo decision for correction; within-tolerance authorization for
   descend; committed release evidence for retreat.
3. Ask the configured IK/collision planner for a joint trajectory using the
   snapshot joint vector as its start. Planning may simulate candidates but may
   not command hardware.
4. Prepend the exact admitted start vector if the planner omits it; reject a
   trajectory whose first state represents a different start.
5. Call the contracted sealer and publish one `action_spec`.
6. Finish with `result_var: NODE_RESULT`.

## Candidate Generation

Candidates may differ in collision-free joint route, but not in phase semantics,
target identity, bounded correction, or snapshot. Rank them by feasibility,
clearance, path length, and start continuity. Publish only the selected path.

## Local Checks

Require finite `(T, N)` joints in snapshot joint order, exact admitted start,
collision/IK success, phase bounds, verified attachment where required, and a
schema-valid sealed digest. `bounded_correction` must not exceed the supplied
translation/yaw step. Descend requires a within-tolerance servo decision.

## Failure Modes

Use `infeasible` when IK/collision planning finds no legal path and
`stale_observation` when snapshot, guard, servo decision, or state receipt is
inconsistent. Never replace an infeasible target with an unmeasured offset.

## Clean Reusable Rules

One graph action equals one immutable action spec and one execution receipt.
Motion authors propose exact joint samples; only the system-action lane owns the
physical write.

## Weak Priors

Prefer short, smooth, high-clearance paths. These are ranking priors only;
phase evidence, collision checks, and the sealed snapshot dominate.

## Prohibited Shortcuts

- Do not load the legacy `robomex.trajectory.v1` multi-phase place template.
- Do not put transport, descend, open, settle, or retreat into one plan.
- Do not emit Cartesian targets for the executor to solve later.
- Do not call `move_to_joints`, `execute_joint_trajectory`, gripper, or step APIs.
- Do not re-read current joints after the admitted snapshot; mismatch fails stale.

## Artifacts to Save

Publish one `action_spec` containing joint order, exact joint samples, execution
policy, plan kind, expected snapshot, affected revisions, and canonical digest.
Planner diagnostics may be saved separately but never become executable input.

## Multimodal Evidence Contract

Every phase is bound to the admission snapshot and its typed graph inputs.
Camera geometry may guide the planning candidate, but cannot replace attachment,
servo, or reducer evidence. A simulator rollout is hypothetical evidence only.

## Reference Code

`build_sealed_phase_motion` from `scripts/sealed_phase_motion.py` is already
bound. `joint_traj` below must be the successful output of the configured
proposal-safe IK/collision planner, never a hand-written or executed path.

```python
snapshot = INPUTS["snapshot"]["payload"]
plan_kind = NODE_CONFIG_V1["plan_kind"]

# `joint_traj` is produced above by the phase-specific proposal-safe planner.
# Normalize to plain lists and bind the exact admitted start mechanically.
planned = [list(map(float, row)) for row in joint_traj]
start = list(map(float, snapshot["joint_positions_rad"]))
if not planned or planned[0] != start:
    planned.insert(0, start)

sealed = build_sealed_phase_motion(
    snapshot,
    plan_kind=plan_kind,
    joint_waypoints=planned,
    tcp_frame_id=NODE_CONFIG_V1.get("tcp_frame_id", "panda_hand"),
    planner_backend=NODE_CONFIG_V1.get("planner_backend", "curobo"),
    robot_model_digest=NODE_CONFIG_V1["robot_model_digest"],
)
NODE_RESULT = {
    "outputs": {"action_spec": {"payload": sealed}},
    "control_outcome": "success",
}
```

Then finish exactly with
`{"tool":"finish","args":{"claim":"one phase sealed as MotionPlan.v2","result_var":"NODE_RESULT"}}`.
