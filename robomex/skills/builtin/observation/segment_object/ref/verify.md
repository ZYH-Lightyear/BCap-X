# Verify: Segment Object by Language

You are the Verifier Agent. Decide whether the segmentation grounded the **right**
object — the one the sub-goal names — and nothing else.

This is an *observe* skill: the scene did not change, so judge a SINGLE frame with the
grounded region annotated. The executor published the box it grounded as
`EVIDENCE["target_box"]` (real pixels, `[x1, y1, x2, y2]`, or `None` if it fell back to
text grounding).

**Hygiene rule:** never paint a filled/colored mask over the object (it corrupts what the
VLM sees). Annotate with an OUTLINE box via `draw_box`, or pass the box coordinates as text.

## Pass criteria

- The outlined region lands on the object named in the sub-goal (not a neighbor of
  similar color/shape), and covers roughly that whole object.

## Example judge code

```python
import json

box = EVIDENCE.get("target_box")
target = "the object named in the sub-goal"  # fill from the sub-goal text

if box is None:
    # text-grounding fallback: no box to outline -> ask about the scene directly, low confidence
    img = OBS_AFTER
    prompt = f"Is exactly one '{target}' clearly present and unambiguous in this image? Reply yes/no + why."
else:
    img = draw_box(OBS_AFTER, box)  # red OUTLINE only, no fill
    prompt = (
        f"A red rectangle outlines a region. Does it tightly enclose '{target}' "
        "(and not a different, similar-looking object)? Reply with JSON "
        '{"match": true/false, "why": "..."}.'
    )

ans = query_vlm(prompt, images=[img])
print(ans)
```

## Fail signals

- The box outlines a different object (e.g. the milk carton when the task wanted the soup),
  spills across several objects, or is empty / `None` with an ambiguous scene.
