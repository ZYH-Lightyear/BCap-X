# Verify: Pick Object

You are the Verifier Agent. Decide whether the named object was actually picked up.

This is an *action* (compound) skill: judge it from the **process**. Prefer watching the pick
unfold via the recorded process clips (`CLIPS` + `process_frames`); fall back to the
`OBS_BEFORE` → `OBS_AFTER` pair if no clip was recorded. The decisive evidence is: the named
object is no longer resting on the surface and is now held by / lifted with the gripper.

**Hygiene rule:** show the raw frames; if you must point at the object, use a `draw_box`
OUTLINE — never paint a filled mask over it.

## Pass criteria

- By the end the named object has clearly left its starting resting spot and is raised with
  the gripper (not still on the table, not dropped, not a different object).

## Example judge code

```python
target = "the object named in the sub-goal"  # fill from the sub-goal text
frames = []
for c in CLIPS:
    frames += process_frames(c["start"], c["end"], k=2)
if not frames:
    frames = [OBS_BEFORE, OBS_AFTER]  # fallback: no process clip recorded
prompt = (
    f"These frames are in time order during a robot pick attempt. Was '{target}' "
    "successfully picked up — i.e. it is now lifted/held by the gripper and no longer "
    'resting where it was? Reply with JSON {"picked": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=frames)
print(ans)
```

## Fail signals

- The object is still on the surface in `OBS_AFTER`, the gripper is empty (fingers fully
  closed on nothing), or a different object moved instead of the requested one.
