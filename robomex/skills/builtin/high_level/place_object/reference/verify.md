# Verify: Place Object

You are the Verifier Agent. Decide whether the held object was actually placed into/onto
the named target and released.

This is an *action* (compound) skill: judge it from the **process**. Prefer watching the
place unfold via the recorded process clips (`CLIPS` + `process_frames`); fall back to the
`OBS_BEFORE` → `OBS_AFTER` pair if no clip was recorded. The decisive evidence is: the object
now rests in/on the named target, and the gripper is empty (open, no longer carrying it).

**Hygiene rule:** show the raw frames; if you must point at the target, use a `draw_box`
OUTLINE — never paint a filled mask over it.

## Pass criteria

- By the end the named object is resting in/on the named target (not on the floor, not still
  in the gripper, not knocked over), and the gripper is open/empty.

## Example judge code

```python
obj = "the object named in the sub-goal"
target = "the placement target named in the sub-goal"
frames = []
for c in CLIPS:
    frames += process_frames(c["start"], c["end"], k=2)
if not frames:
    frames = [OBS_BEFORE, OBS_AFTER]  # fallback: no process clip recorded
prompt = (
    f"These frames are in time order during a place attempt. Is '{obj}' now resting in/on "
    f"'{target}' and released by the gripper (gripper empty)? Reply with JSON "
    '{"placed": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=frames)
print(ans)
```

## Fail signals

- The object is still held by the gripper, fell outside the target, or knocked the target
  over; or it landed on/in the wrong target.
