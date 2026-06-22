# Verify: Place Object

You are the Verifier Agent. Decide whether the held object was actually placed into/onto
the named target and released.

This is an *action* (compound) skill: judge the **before/after** change. `OBS_BEFORE` is
the scene at the sub-goal's start (object still in hand), `OBS_AFTER` is the scene now.
The decisive evidence is: the object now rests in/on the named target, and the gripper is
empty (open, no longer carrying it).

**Hygiene rule:** show the raw before/after frames; if you must point at the target, use a
`draw_box` OUTLINE — never paint a filled mask over it.

## Pass criteria

- In `OBS_AFTER` the named object is resting in/on the named target (not on the floor,
  not still in the gripper, not knocked over), and the gripper is open/empty.

## Example judge code

```python
obj = "the object named in the sub-goal"
target = "the placement target named in the sub-goal"
prompt = (
    f"Two images: BEFORE then AFTER a place attempt. Is '{obj}' now resting in/on "
    f"'{target}' and released by the gripper (gripper empty)? Reply with JSON "
    '{"placed": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=[OBS_BEFORE, OBS_AFTER])
print(ans)
```

## Fail signals

- The object is still held by the gripper, fell outside the target, or knocked the target
  over; or it landed on/in the wrong target.
