---
name: language-grounding
description: Use when task language mentions an object, receptacle, surface, handle, button, articulated fixture, or spatial relation that must be localized in an image.
---

# Language Grounding

Ground task language into an image point, box, or short scene judgment before segmentation, grasping, placing, opening, pressing, or pushing.

## APIs

```python
cam = get_camera("main")
point = ground_point("the target object", camera="main")
reply = query_vlm(prompt, images=cam["rgb"])
```

`ground_point` returns `(x, y)` pixel coordinates or `(None, None)`. The harness chooses the backend; do not call backend-specific point helpers directly.

`query_vlm` is better for JSON, disambiguation, relation reasoning, and completion checks.

## Coordinate Conventions

Some VLMs return points or boxes in a normalized `0..1000` coordinate system. Convert only when the values exceed the real image shape:

```python
def maybe_1000_to_pixel(point, rgb):
    x, y = point
    h, w = rgb.shape[:2]
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
        return int(round(x / 1000 * (w - 1))), int(round(y / 1000 * (h - 1)))
    return int(round(x)), int(round(y))
```

## Workflow

1. Start with `cam = get_camera("main")`.
2. For a single named object, call `ground_point(description, camera="main")`.
3. For ambiguous goals, ask `query_vlm` for strict JSON with one point or one object choice.
4. Pass the grounded point to `segment_object((x, y), camera="main")`.

## Pitfalls

- Do not pass normalized coordinates to segmentation or geometry helpers.
- If the wrong object is grounded, add color, label, relative position, or nearby references.
- If VLM output is malformed, re-ask with "ONLY JSON" and a smaller schema.

## Related Skills

Use `$scene-observation` before grounding, `$segmentation-to-points` after grounding, and `$articulated-and-contact-actions` for handles, buttons, doors, drawers, and push contact points.
