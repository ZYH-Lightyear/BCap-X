# Verify: Release At

You are the Verifier Agent. Decide whether the held object was actually released at the
target and the gripper opened.

This is an *action* skill: judge it from the **process**. Prefer watching the release unfold
via the recorded process clips (`CLIPS` + `process_frames`); fall back to the `OBS_BEFORE`
→ `OBS_AFTER` pair if no clip was recorded. A successful release shows the object leaving the
fingers and coming to rest at the target with the gripper open and empty.

**Hygiene rule:** show the raw frames; if you must point at the target, use a `draw_box`
OUTLINE — never paint a filled mask over it.

## Pass criteria

- Across the action the object left the gripper and now rests at the release location, and
  the gripper is open/empty.

## Example judge code

```python
obj = "the object named in the sub-goal"
frames = []
for c in CLIPS:
    frames += process_frames(c["start"], c["end"], k=2)
if not frames:
    frames = [OBS_BEFORE, OBS_AFTER]  # fallback: no process clip recorded
prompt = (
    f"These frames are in time order during a release. Is '{obj}' no longer held by the "
    "gripper and now resting on/in the target below where the gripper was? Reply with JSON "
    '{"released": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=frames)
print(ans)
```

## Fail signals

- The object is still in the gripper (fingers closed on it), or it fell away from the
  intended target / toppled it.
