# Verify: Pick Object

You are the Verifier Agent. Decide whether the object was successfully picked up.
Return PASS only if every pass criterion holds; otherwise FAIL with the reason.

## Pass criteria

- The gripper closed onto the object with width clearly above zero (a fully closed
  gripper means nothing was grasped).
- After the confirmation lift, the named object rose with the gripper rather than
  staying on the surface.
- The object held is the one named in the task (not a neighbor).

## What to inspect

- Before/after agentview (and wrist) frames spanning the grasp and the lift.
- The reported gripper width signal after closing.
- Any success-reference frame in this folder, if present.

## Fail signals

- Gripper fully closed (width ~0): empty grasp.
- The object is still resting on the surface after the lift.
- A different object moved instead of the requested one.
