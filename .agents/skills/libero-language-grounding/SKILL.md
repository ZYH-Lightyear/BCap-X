---
name: libero-language-grounding
description: Use for CaP-X LIBERO tasks when natural language mentions an object, receptacle, surface, drawer, stove, microwave, plate, basket, cabinet, caddy compartment, handle, button, or spatial relation that must be localized in an image before segmentation or manipulation.
---

# LIBERO Language Grounding

Ground task language into an image point, box, or short scene judgment before running SAM3, grasping, placing, opening, turning, or pushing.

## APIs

```python
points = point_prompt_molmo(rgb, "the target object")
reply = query_vlm(prompt, images=rgb)
```

`point_prompt_molmo` returns a mapping from query text to `(x, y)` pixel coordinates or `(None, None)`.

`query_vlm` is better when you need JSON, disambiguation, relation reasoning, or a completion check.

## Coordinate Conventions

`point_prompt_molmo` returns real pixel coordinates. Pass those directly to point-prompt SAM3.

Some VLMs, especially Qwen-family vision models, often return points or boxes in a normalized `0..1000` image coordinate system even when the prompt asks for pixels. Convert normalized coordinates before using SAM3 or `pixel_to_world_point`:

```python
def maybe_qwen_1000_to_pixel(point, rgb):
    x, y = point
    h, w = rgb.shape[:2]
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
        return int(round(x / 1000 * (w - 1))), int(round(y / 1000 * (h - 1)))
    return int(round(x)), int(round(y))

def maybe_qwen_1000_box_to_pixel(box, rgb):
    x1, y1, x2, y2 = box
    h, w = rgb.shape[:2]
    if max(box) <= 1000 and (x2 > w or y2 > h):
        return [
            int(round(x1 / 1000 * (w - 1))),
            int(round(y1 / 1000 * (h - 1))),
            int(round(x2 / 1000 * (w - 1))),
            int(round(y2 / 1000 * (h - 1))),
        ]
    return [int(round(v)) for v in box]
```

OpenAI/GPT-style vision models may return true pixels if explicitly requested. Do not blindly rescale every answer; compare returned values with the actual image `width` and `height`.

## Workflow

1. Start from `obs = get_observation()` and `rgb = obs["agentview"]["images"]["rgb"]`.
2. For a single named object, call `point_prompt_molmo(rgb, object_description)`.
3. For ambiguous goals, ask `query_vlm` for strict JSON:

```python
prompt = (
    "Find the single object matching 'black bowl next to the ramekin'. "
    "Reply only as JSON: {\"point\": [x, y]} in pixel coordinates."
)
reply = query_vlm(prompt, images=rgb)
```

4. Pass the grounded point to `segment_sam3_point_prompt`, or use the answer to decide which object or region to manipulate.

## Task Language Hints

- For `next to`, `between`, `left`, `right`, `front`, or `back`, include the reference object in the query.
- For `drawer`, `microwave`, `stove`, `cabinet`, or `caddy`, ask for the actionable region: handle, button, top surface, compartment, or opening.
- For multi-object tasks such as "put both", ground and manipulate one object at a time, then re-observe.

## Pitfalls

- Do not pass normalized 0-1000 coordinates to SAM3 unless you rescale them to real pixels.
- If using `query_vlm` with Qwen-family models, assume point/box answers may be 0-1000 normalized until checked against image width and height.
- If VLM output is malformed, re-ask with "ONLY JSON" and a smaller schema.
- If the wrong object is grounded, add color, label, relative position, or nearby objects to the description.

## Related Skills

Use `$libero-scene-observation` before grounding, `$libero-segmentation-to-points` after grounding, and `$libero-articulated-and-contact-actions` when grounding handles, buttons, drawers, or push contact points.
