# Verify: Segment Object by Language

You are the Verifier Agent. Decide whether the segmentation grounded the **right**
object — the one the sub-goal names — and nothing else.

This is an *observe* skill: the scene did not change. The executor saved two annotated
debug images to disk and published their paths in `EVIDENCE`:

- `EVIDENCE["vlm_box_image"]` — the scene with a **red bounding box** drawn around the
  region the VLM picked as the target object.
- `EVIDENCE["sam3_mask_image"]` — the scene with a **green contour** of the SAM3 mask
  plus the red VLM box, showing what was actually segmented.

Load these images directly and ask the VLM whether the right object was grounded.

**Hygiene rule:** these images already contain clean outline annotations (box / contour);
do NOT paint additional filled masks over the object.

## Pass criteria

- The red box and the green contour both land on the object named in the sub-goal (not a
  neighbor of similar color/shape), and the contour covers roughly that whole object.

## Example judge code

```python
from PIL import Image
import numpy as np

target = "the object named in the sub-goal"  # fill from the sub-goal text

vlm_img_path = EVIDENCE.get("vlm_box_image")
mask_img_path = EVIDENCE.get("sam3_mask_image")

# Prefer the mask overlay (it shows both the box AND the segmentation result).
if mask_img_path:
    img = np.array(Image.open(mask_img_path))
    prompt = (
        f"This image shows a red bounding box and green segmentation contour. "
        f"Do they correctly outline '{target}' (and not a different object)? "
        'Reply with JSON {{"match": true/false, "why": "..."}}.'
    )
elif vlm_img_path:
    img = np.array(Image.open(vlm_img_path))
    prompt = (
        f"A red rectangle outlines a region. Does it tightly enclose '{target}' "
        '(and not a different object)? Reply with JSON {{"match": true/false, "why": "..."}}.'
    )
else:
    # No saved images — fall back to drawing the box on OBS_AFTER
    box = EVIDENCE.get("target_box")
    assert box, "No target_box or saved images — cannot verify segment"
    img = draw_box(OBS_AFTER, box)
    prompt = (
        f"A red rectangle outlines a region. Does it tightly enclose '{target}'? "
        'Reply with JSON {{"match": true/false, "why": "..."}}.'
    )

ans = query_vlm(prompt, images=[img])
print(ans)
```

## Fail signals

- The box / contour outlines a different object (e.g. the milk carton when the task wanted
  the soup), the contour spills across several objects, or no evidence images were saved.
