---
name: Release At
category: motion
description: "Execute one contracted transport-and-release trajectory and publish execution evidence."
---

# Release At

## Purpose

Execute a bounded release attempt from a supplied trajectory artifact. Placement
geometry, object-center compensation, and feasibility belong to upstream Affordance and
MotionPlanner stages.

## When to use

Use after MotionPlanner has published a feasible transport, release, open, and retreat
sequence (for place: the `build_place_trajectory` template).

## When NOT to use

Do not use to compute placement geometry, refresh perception, verify the final
relation, or release when grasp/trajectory state is uncertain.

## Workflow

1. Read the supplied trajectory and reject missing, non-finite, or unbounded waypoints.
2. Execute only its declared transport and release sequence.
   Joint-space cuRobo waypoints use `move_to_joints` or
   `execute_joint_trajectory`; Cartesian waypoints use `goto_pose`. Never solve IK
   again inside this executor.
3. Move to transport hover, then descend to the release TCP. Record each
   `goto_pose` / `move_to_joints` status dict (`converged`, `settled`, `stalled`,
   `timed_out`, `steps`, `step_cap`, and `final_error`). A non-converged,
   timed-out, or stalled critical motion stops the sequence immediately: opening
   or descending from an unknown pose can drop the object outside the target.
4. Treat the descend status as evidence for deciding whether to open, not as a
   hard-coded gate. Preserve the decision and status for downstream verification.
5. Settle after open before any retreat: call `settle_after_open` on the gripper-open
   API (or open with settle steps). Do not retreat immediately after opening — the
   object must land/settle first (GaP-style).
6. Execute retreat upward only after settle.
7. Publish primitive status (including per-primitive `converged`/`stalled`/
   `final_error`) and any API failure as ExecutionEvidence. The runtime
   automatically captures `terminal_robot_state` after every state-changing block
   and injects it into your ExecutionEvidence; never call `get_observation()` —
   this node has no `perception_read` capability.

## Candidate Generation

- Candidate generation is outside this execution stage.
- Never derive a replacement release pose or change object-center compensation locally.

## Local Checks

- Check the trajectory feasibility flag and bounded waypoint count.
- Check the status dict returned by every `goto_pose` / `move_to_joints` call:
  `converged=False`, `timed_out=True`, or `stalled=True` must be recorded as
  motion evidence, but none automatically forbids the next declared action.
- Confirm release TCP came from the trajectory / affordance `position`, not from
  inventing `desired_object_center` as TCP.
- Gripper/robot terminal state is captured by the runtime and echoed back in
  execution feedback as `terminal_robot_state`. Leave the object-target verdict
  to Verifier.

## Failure Modes

- Motion failure: record the failed primitive and the executor's subsequent
  decision; failure does not automatically terminate the declared sequence.
- Collision/workspace refusal: return to MotionPlanner.
- Post-release relation failure: preserve evidence for graph-level recovery.

## Clean Reusable Rules

- Execute one supplied candidate; do not combine planning and execution.
- Keep motion outcome and control-flow decision separate; settle after open before
  retreat.
- Executor self-check is soft evidence; independent verification certifies placement.

## Weak Priors

- Smooth short transport and retreat segments are preferable when already declared.
- CapX `open_gripper` already steps the sim; still keep retreat after open completes.

## Prohibited Shortcuts

- Do not hardcode release coordinates or target-specific offsets from one seed.
- Do not call `get_observation()` or any perception function; terminal state arrives
  via runtime-captured `terminal_robot_state`.
- Do not ground, query visual coordinates, recompute placement, or replan motion.
- Do not invent retries inside this node.
- Do not open the gripper and immediately retreat without settle.

## Artifacts to Save

- Executed API code and ordered primitive trace.
- Any primitive failure. Terminal robot/gripper state is attached automatically
  by the runtime.

## Multimodal Evidence Contract

This node consumes typed trajectory evidence and no camera observations. Publish
ordered primitive status plus runtime-captured terminal robot state. On any
critical primitive failure, publish the partial trace with
`failure_kind: execution_fault`; only an uninterrupted release proceeds to fresh
placement verification.

## Optional Sidecars

`scripts/release.py` provides `compute_tcp_release_pos`, `settle_after_open`,
and `release_execution_checklist`. Geometry still belongs
upstream; these helpers only enforce execution order.

## Reference Code

`compute_tcp_release_pos` and `place_quat_from_affordance` are contracted,
pre-bound functions. Call them directly; do not import or inspect them.

```python
# Prefer INPUTS for trajectory / affordance — do not retype pose literals.
placement_affordance = INPUTS.get("placement_affordance", {}).get("payload") or placement_affordance
tcp = compute_tcp_release_pos(placement_affordance, EVIDENCE.get("held_object_frame"))
quat = place_quat_from_affordance(placement_affordance)  # never invent [1,0,0,0]
status = goto_pose(tcp, quat)     # record status["converged"] / ["stalled"] in the trace
motion_ok = (
    bool(status.get("converged"))
    and not bool(status.get("timed_out"))
    and not bool(status.get("stalled"))
)
if motion_ok:
    open_gripper()
NODE_RESULT = {
    "outputs": {
        "execution_evidence": {
            "payload": {
                "primitives": [
                    {"name": "goto_release", "status": "succeeded" if motion_ok else "failed",
                     "converged": bool(status.get("converged")), "stalled": bool(status.get("stalled"))},
                    *([{"name": "open_gripper", "status": "succeeded"}] if motion_ok else []),
                ],
                "all_primitives_ok": motion_ok,
            },
            "confidence": 0.8,
            "artifacts": {},
        }
    },
    "recommended_next": "verify" if motion_ok else "replan",
    "failure_kind": "" if motion_ok else "execution_fault",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
