---
name: Grasp Object
category: motion
description: "Execute one contracted grasp trajectory and publish primitive-level execution evidence."
---

# Grasp Object

## Purpose

Execute one grasp attempt from a supplied, feasible trajectory artifact. This skill
owns bounded robot and gripper calls, but not grounding, affordance generation, motion
planning, or the independent success verdict.

## When to use

Use only after a MotionPlanner has published the selected approach, contact, close,
lift, and retreat candidate.

## When NOT to use

Do not use this skill to repair poses, inspect camera observations, choose a grasp
family, or certify physical success. Those responsibilities belong to planning,
affordance, perception, and verification nodes.

## Workflow

1. Read the supplied trajectory artifact and reject missing or non-finite waypoints.
2. Execute only the declared bounded sequence.
3. Open the gripper, approach with the planned clearance, descend to the grasp pose, and
   close.
4. `goto_pose` / `move_to_joints` return a motion status dict. Record
   `converged`, `settled`, `stalled`, `timed_out`, `steps`, `step_cap`, and
   `final_error` per primitive. A non-converged, timed-out, or stalled critical
   motion is a hard execution gate: stop before dependent motion or gripper
   commands. Continuing from an unknown pose can collide with the table or close
   on empty space.
5. Publish the partial ExecutionEvidence with
   `failure_kind: execution_fault`. A downstream Verifier judges physical grasp
   success only after an uninterrupted sequence.
6. Lift a short distance before doing any transport.
7. Publish primitive status (including per-primitive `converged`/`stalled`/
   `final_error`), timestamps, and any API failure as ExecutionEvidence.
   The runtime automatically captures `terminal_robot_state` after every
   state-changing block and injects it into your ExecutionEvidence; it also appears
   in the execution feedback. Never call `get_observation()` — this node has no
   `perception_read` capability. A separate Verifier consumes that evidence and
   fresh observations.

## Candidate Generation

- Candidate generation is outside this execution stage.
- Do not substitute a new pose, retry arbitrary quaternions, or call segmentation.
- If the supplied trajectory is invalid, fail without motion so recovery can return to
  motion planning.

## Local Checks

- Confirm the trajectory's feasibility flag and bounded waypoint count.
- Check the status dict returned by every `goto_pose` / `move_to_joints` call.
  `converged=False`, `timed_out=True`, or `stalled=True` must be recorded and
  immediately stop the sequence.
- The terminal end-effector/gripper state is captured by the runtime and echoed back
  in execution feedback as `terminal_robot_state`.
- Do not reinterpret gripper width as a success verdict.

## Failure Modes

- Primitive/API failure: stop and publish the failed primitive.
- Collision or workspace refusal: do not widen bounds locally; return to MotionPlanner.
- Apparent grasp miss: preserve terminal state for the independent Verifier.

## Clean Reusable Rules

- Execute one bounded supplied candidate; never combine planning and execution.
- Executor self-checks are soft evidence. Only the graph's hard Verifier can certify
  the postcondition.

## Weak Priors

- Smooth short segments are usually preferable, but the contracted trajectory is
  authoritative.

## Prohibited Shortcuts

- Do not hardcode grasp coordinates, pixels, object poses, or seed-specific locations.
- Do not call `get_observation()` or any perception function; terminal state arrives
  via runtime-captured `terminal_robot_state`.
- Do not perform SAM3, grounding, affordance search, or motion replanning.
- Do not repeat or modify a failed pose inside this node.
- Do not call `goto_home_joint_position` in a way that opens/releases the gripper while
  holding an object.

## Artifacts to Save

- Executed API code and ordered primitive trace.
- Any primitive failure. Terminal robot/gripper state is attached automatically
  by the runtime.

## Multimodal Evidence Contract

This node consumes typed trajectory evidence and no camera observations. Publish
ordered primitive status plus runtime-captured terminal robot state. On
`execution_fault`, preserve the partial trace and do not claim a grasp verdict;
the independent verifier must use a fresh post-action observation epoch.

## Optional Sidecars

No sidecar is required. Use only motion primitives and the supplied trajectory artifact.

## Reference Code

Read the trajectory from `INPUTS` — do not retype waypoint numbers. Ontology for this
Franka / LIBERO env: top-down = `[0, 1, 0, 0]` (π about x, wxyz); identity
`[1, 0, 0, 0]` is gripper skyward and is never a grasp orientation. Prefer the
quaternion already on each waypoint.

For a cuRobo trajectory, each waypoint contains `joints` instead of Cartesian
pose fields. Execute it with `move_to_joints` (or
`execute_joint_trajectory` when the dense joint array is supplied), apply the same
fail-fast gate, and never run IK again in the executor.

```python
import json, os
import numpy as np

traj = INPUTS["trajectory"]["payload"]  # or load from INPUTS[...]["artifacts"]
waypoints = traj["waypoints"]
primitives = []
for wp in waypoints:
    if "joints" in wp:
        status = move_to_joints(np.asarray(wp["joints"]))
    else:
        status = goto_pose(
            np.asarray(wp["position_xyz"]),
            np.asarray(wp["quaternion_wxyz"]),
        )
    motion_ok = (
        bool(status.get("converged"))
        and not bool(status.get("timed_out"))
        and not bool(status.get("stalled"))
    )
    primitives.append({
        "name": f"goto_{wp['name']}",
        "status": "succeeded" if motion_ok else "failed",
        "converged": bool(status.get("converged")),
        "timed_out": bool(status.get("timed_out")),
        "stalled": bool(status.get("stalled")),
        "final_error": float(status.get("final_error") or 0.0),
    })
    if not motion_ok:
        break
    if wp.get("gripper") == "close":
        close_gripper()
        primitives.append({"name": "close_gripper", "status": "succeeded"})
    elif wp.get("gripper") == "open":
        open_gripper()
        primitives.append({"name": "open_gripper", "status": "succeeded"})

trace_path = os.path.join(ARTIFACTS_DIR, "execution_evidence.json")
with open(trace_path, "w") as f:
    json.dump({"primitives": primitives, "all_ok": all(p["status"] == "succeeded" for p in primitives)}, f, indent=2)

all_ok = bool(primitives) and all(p["status"] == "succeeded" for p in primitives)
NODE_RESULT = {
    "outputs": {
        "execution_evidence": {
            "payload": {"primitives": primitives, "all_primitives_ok": all_ok},
            "confidence": 0.8,
            "artifacts": {"evidence_file": "execution_evidence.json"},
        }
    },
    "recommended_next": "verify" if all_ok else "replan",
    "failure_kind": "" if all_ok else "execution_fault",
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
