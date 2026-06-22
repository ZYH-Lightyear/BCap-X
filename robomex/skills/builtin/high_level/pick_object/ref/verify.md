# Verify: Pick Object

You are the Verifier Agent. Decide whether the named object was actually picked up.

This is an *action* skill: judge the **before/after** change. `OBS_BEFORE` is the scene at
the sub-goal's start, `OBS_AFTER` is the scene now. The simplest decisive evidence is:
the named object is no longer resting on the surface and is now held by / lifted with the
gripper.

**Hygiene rule:** show the raw before/after frames; if you must point at the object, use a
`draw_box` OUTLINE — never paint a filled mask over it.

## Pass criteria

- In `OBS_AFTER` the named object has clearly left its `OBS_BEFORE` resting spot and is
  raised with the gripper (not still on the table, not dropped, not a different object).

## Example judge code

```python
target = "the object named in the sub-goal"  # fill from the sub-goal text
prompt = (
    f"Two images: BEFORE then AFTER a robot pick attempt. Was '{target}' successfully "
    "picked up — i.e. it is now lifted/held by the gripper and no longer resting where it "
    'was? Reply with JSON {"picked": true/false, "why": "..."}.'
)
ans = query_vlm(prompt, images=[OBS_BEFORE, OBS_AFTER])
print(ans)
```

## Fail signals

- The object is still on the surface in `OBS_AFTER`, the gripper is empty (fingers fully
  closed on nothing), or a different object moved instead of the requested one.
