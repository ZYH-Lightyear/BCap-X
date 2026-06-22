# Verify: Grasp Object

You are the Verifier Agent. Decide whether the grasp actually secured the object.

This is an *action* skill: judge the **before/after** change. `OBS_BEFORE` is the scene at
the sub-goal's start, `OBS_AFTER` is the scene now. A secured grasp shows the object moving
up with the gripper during the confirmation lift rather than staying on the surface.

**Hygiene rule:** show the raw before/after frames; if you must point at the object, use a
`draw_box` OUTLINE — never paint a filled mask over it.

## Pass criteria

- Between `OBS_BEFORE` and `OBS_AFTER` the named object rose with the gripper (held between
  the fingers), rather than staying put or slipping out.

## Example judge code

```python
target = "the object named in the sub-goal"  # fill from the sub-goal text
prompt = (
    f"Two images: BEFORE then AFTER a grasp + small lift. Is '{target}' now held between "
    "the gripper fingers and lifted off the surface (a secure grasp), or did the grasp miss "
    '/ slip? Reply with JSON {"grasped": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=[OBS_BEFORE, OBS_AFTER])
print(ans)
```

## Fail signals

- Fingers fully closed on nothing (empty grasp), or the object stayed on the surface /
  slipped out during the lift.
