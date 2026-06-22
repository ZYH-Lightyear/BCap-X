# Verify: Release At

You are the Verifier Agent. Decide whether the held object was actually released at the
target and the gripper opened.

This is an *action* skill: judge the **before/after** change. `OBS_BEFORE` is the scene at
the sub-goal's start (object still in hand), `OBS_AFTER` is the scene now. A successful
release shows the object resting at the target location with the gripper open and empty.

**Hygiene rule:** show the raw before/after frames; if you must point at the target, use a
`draw_box` OUTLINE — never paint a filled mask over it.

## Pass criteria

- Between `OBS_BEFORE` and `OBS_AFTER` the object left the gripper and now rests at the
  release location, and the gripper is open/empty.

## Example judge code

```python
obj = "the object named in the sub-goal"
prompt = (
    f"Two images: BEFORE then AFTER a release. Is '{obj}' no longer held by the gripper "
    "and now resting on/in the target below where the gripper was? Reply with JSON "
    '{"released": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=[OBS_BEFORE, OBS_AFTER])
print(ans)
```

## Fail signals

- The object is still in the gripper (fingers closed on it), or it fell away from the
  intended target / toppled it.
