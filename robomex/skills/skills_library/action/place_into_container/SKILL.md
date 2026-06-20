---
name: Place Held Object Into Container
category: action
description: Carry the held object over a container and release it inside, with the drop height computed from measured geometry instead of guessed constants.
---

# Place Held Object Into Container

Place the held object into a container. Both geometries must come from the geometry
skill: the held object's height and the container's `top_z` / `center` (measured, not guessed).

## When to use

When an object is already held and a target container has been segmented and measured.

## Procedure

```python
top_down = np.array([0.0, 1.0, 0.0, 0.0])             # wxyz

clearance = 0.03
release_z = container_top_z + held_height / 2 + clearance
target = np.array([container_center[0], container_center[1], release_z + 0.10])

goto_pose(target, top_down, z_approach=0.05)
open_gripper()

# Retreat straight up so the gripper does not clip the container rim:
retreat = target.copy()
retreat[2] += 0.10
goto_pose(retreat, top_down)
```

## Rules

- NEVER hardcode a release height: `release_z` is derived from the container's
  measured `top_z` and the held object's measured `height`.
- Approach and retreat vertically over the container center; lateral motion near
  the rim knocks containers over.
- Release only when positioned over the opening; after release, verify before
  declaring success.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: the object's fresh mask center lies within the container footprint and the gripper is open.

## Failure modes

- Object landed outside: re-segment the object, re-run grasp candidates, re-grasp,
  and approach again with higher clearance over the container center.
- Container moved during the approach: re-estimate the container geometry before retrying.
