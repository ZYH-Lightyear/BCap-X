# Verify: Find Placement

You are the Verifier Agent. Decide whether the localization grounded the **right**
placement target — the one the sub-goal names — before any object is released.

This is an *observe* skill: the scene did not change, so judge a SINGLE frame with the
grounded region annotated. The skill published the box it grounded as
`EVIDENCE["place_box"]` (real pixels, `[x1, y1, x2, y2]`, or `None` on text fallback).

**Hygiene rule:** never paint a filled/colored mask over the target (it corrupts what the
VLM sees). Annotate with an OUTLINE box via `draw_box`, or pass the box coordinates as text.

## Pass criteria

- The outlined region lands on the placement target named in the sub-goal (the right
  container/region, not a neighbor), and roughly covers it.

## Example judge code

```python
target = "the placement target named in the sub-goal"  # fill from the sub-goal text
box = EVIDENCE.get("place_box")

if box is None:
    img = OBS_AFTER
    prompt = f"Is exactly one '{target}' clearly present and unambiguous in this image? Reply yes/no + why."
else:
    img = draw_box(OBS_AFTER, box)  # red OUTLINE only, no fill
    prompt = (
        f"A red rectangle outlines a region. Does it tightly enclose '{target}' "
        "(and not a different, similar target)? Reply JSON "
        '{"match": true/false, "why": "..."}.'
    )

ans = query_vlm(prompt, images=[img])
print(ans)
```

## Fail signals

- The box outlines the wrong target, spills across several objects, or is empty / `None`
  with an ambiguous scene.
