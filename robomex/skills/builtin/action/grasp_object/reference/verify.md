# Verify: Grasp Object

You are the Verifier Agent. Decide whether the grasp actually secured the object.

This is an *action* skill: judge it from the **process**. Prefer watching the grasp + lift
unfold via the recorded process clips (`CLIPS` + `process_frames`); fall back to the
`OBS_BEFORE` → `OBS_AFTER` pair if no clip was recorded. A secured grasp shows the object
rising with the gripper during the confirmation lift rather than staying on the surface.

**Hygiene rule:** show the raw frames; if you must point at the object, use a `draw_box`
OUTLINE — never paint a filled mask over it.

## Pass criteria

- Across the action the named object rose with the gripper (held between the fingers),
  rather than staying put or slipping out.

## Example judge code

```python
target = "the object named in the sub-goal"  # fill from the sub-goal text
# Watch the action as it happened: sample a few frames from each process clip (time-ordered).
frames = []
for c in CLIPS:
    frames += process_frames(c["start"], c["end"], k=2)
if not frames:
    frames = [OBS_BEFORE, OBS_AFTER]  # fallback: no process clip recorded
prompt = (
    f"These frames are in time order during a grasp + small lift. Is '{target}' now held "
    "between the gripper fingers and lifted off the surface (a secure grasp), or did the "
    'grasp miss / slip? Reply with JSON {"grasped": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=frames)
print(ans)
```

## Fail signals

- Fingers fully closed on nothing (empty grasp), or the object stayed on the surface /
  slipped out during the lift.
