---
name: Generate and Vet Grasp Candidates
category: observation
description: Get GraspNet candidates, prefer a top-down aligned one, and prove IK feasibility BEFORE any motion. Never trust the top-1 raw candidate.
---

# Generate and Vet Grasp Candidates

Generate grasp candidates and vet them before any motion. GraspNet's raw top-1 is
frequently unreachable or badly oriented (bad quaternion) — vetting is the whole
point of this skill, producing a `grasp_pos` / `grasp_quat` that is proven feasible.

## When to use

After segmenting + measuring the object, immediately before executing a grasp.

## Procedure

```python
obs = get_observation()
cam = obs["agentview"]
grasps_cam, scores = plan_grasp(cam["images"]["depth"], cam["intrinsics"], mask.astype(np.int32))
# Candidates are 4x4 transforms IN THE CAMERA FRAME; selection handles the world transform:
g_world, g_score = select_top_down_grasp(grasps_cam, scores, cam["pose_mat"], vertical_threshold=0.8)
assert g_world is not None, "no sufficiently top-down candidate"
grasp_pos, grasp_quat = decompose_transform(g_world)     # quat is wxyz
joints = solve_ik(grasp_pos, grasp_quat)                  # feasibility proof; raises if unreachable
```

## Rules

- `plan_grasp` output is camera-frame; only use poses after transforming to world
  (`select_top_down_grasp` does this internally).
- Quaternions are wxyz everywhere in this stack; `decompose_transform` returns wxyz.
- Top-down convention if you need a fixed orientation: `np.array([0.0, 1.0, 0.0, 0.0])` (wxyz).
- Vet candidates in score order until one passes both the alignment and the IK check;
  do not loop forever — after ~5 failures, re-perceive instead.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: the chosen grasp is IK-feasible and sits on/near the object points.

## Failure modes

- `select_top_down_grasp` returns `(None, -inf)`: lower `vertical_threshold` to ~0.6,
  or fall back to the highest-score candidate that passes `solve_ik`.
- IK fails for the best candidate: iterate the next-best candidates instead of
  forcing the same pose.
