---
name: libero-articulated-and-contact-actions
description: Use for non-pick-place CaP-X LIBERO actions involving open drawer, close drawer, open or close microwave, turn on stove, push plate, pull handle, press button, or short contact motions against an object or articulated fixture.
---

# LIBERO Articulated And Contact Actions

Use this skill for drawers, microwave doors, stove buttons/knobs, and pushing objects.

## APIs

```python
reply = query_vlm(prompt, images=rgb)
points = point_prompt_molmo(rgb, "drawer handle")
p = pixel_to_world_point(u, v, z, intrinsics, pose_mat)
direction = normalize_vector(v)
waypoints = interpolate_segment(start, end, step=0.02)
goto_pose(position, quaternion_wxyz)
open_gripper()
close_gripper()
```

## Workflow

1. Use `query_vlm` or `point_prompt_molmo` to locate the handle, button, knob, door edge, or push contact point.
2. If the contact point came from a Qwen-family VLM, check whether it is normalized to `0..1000` and convert to pixels:

```python
h, w = rgb.shape[:2]
if 0 <= u <= 1000 and 0 <= v <= 1000 and (u > w or v > h):
    u = int(round(u / 1000 * (w - 1)))
    v = int(round(v / 1000 * (h - 1)))
```

3. Convert the contact pixel to a world point using local depth.
4. Choose a simple contact orientation, usually top-down or forward-facing enough for the fixture.
5. For push actions:

```python
direction = normalize_vector(target_point - contact_point)
waypoints = interpolate_segment(contact_point, contact_point + 0.08 * direction, 0.02)
for p in waypoints:
    goto_pose(p, quat)
```

6. For drawers or microwave doors, close the gripper lightly on a handle if needed, then move along the pull or push direction in short segments.
7. Re-observe after the interaction before placing objects inside or deciding completion.

## Task Patterns

- `open drawer`: target the handle/front edge and pull outward.
- `close drawer` or `close microwave`: push the front face inward.
- `turn on stove`: locate the control/button/knob and press or push it.
- `push plate`: contact the plate side opposite the desired direction, then move a short line toward the goal.

## Pitfalls

- Articulated actions are sensitive; use short waypoints and re-observe often.
- If a contact point has invalid depth, sample nearby pixels or ask the VLM for a slightly different point.
- Do not feed Qwen-style normalized `0..1000` points directly into `pixel_to_world_point`; convert using the actual image width and height.
- Do not use grasp-style lift after a push or press.

## Related Skills

Use `$libero-language-grounding` to locate handles/buttons, `$libero-geometry-and-frames` for directions and paths, and `$libero-debug-and-recovery` when contact motions leave the scene unchanged.
