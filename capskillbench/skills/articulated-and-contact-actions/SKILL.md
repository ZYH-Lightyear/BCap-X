---
name: articulated-and-contact-actions
description: Use for non-pick-place actions involving drawers, doors, buttons, knobs, handles, pushing, pulling, pressing, or short contact motions.
---

# Articulated And Contact Actions

Use this skill for handles, drawers, doors, buttons, knobs, and push actions.

## APIs

```python
point = ground_point("drawer handle", camera="main")
p = pixel_to_world_point(x, y, camera="main")
direction = normalize_vector(v)
waypoints = interpolate_segment(start, end, step=0.02)
goto_pose(position, quaternion_wxyz)
open_gripper()
close_gripper()
```

## Workflow

1. Use `ground_point` or `query_vlm` to locate the actionable point.
2. Convert the contact pixel to a world point using `pixel_to_world_point`.
3. Choose a simple contact orientation, usually top-down or forward-facing enough for the fixture.
4. For push actions:

```python
direction = normalize_vector(target_point - contact_point)
waypoints = interpolate_segment(contact_point, contact_point + 0.08 * direction, 0.02)
for p in waypoints:
    goto_pose(p, quat)
```

5. For pull actions, close the gripper lightly on a handle if needed, then move in short segments.
6. Re-observe after the interaction before placing objects inside or deciding completion.

## Pitfalls

- Use short waypoints and re-observe often.
- If a contact point has invalid depth, sample nearby pixels or ask for a slightly different point.
- Do not use grasp-style lift after a push or press.

## Related Skills

Use `$language-grounding` to locate handles/buttons, `$geometry-and-frames` for paths, and `$debug-and-recovery` when contact motions leave the scene unchanged.
