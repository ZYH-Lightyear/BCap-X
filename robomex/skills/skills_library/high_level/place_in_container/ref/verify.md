# Verify: Place Held Object In Container

You are the Verifier Agent. Decide whether the held object was released into the
target container. Return PASS only if every pass criterion holds; otherwise FAIL.

## Pass criteria

- A fresh segmentation of the object lies within the container's footprint (the
  object is inside / on top of the container, not beside it).
- The gripper is open and empty.
- The arm retreated vertically without dragging or knocking over the container.

## What to inspect

- After frame of the scene once the gripper has released and retreated.
- A re-segmentation of the object relative to the container footprint.
- Any success-reference frame in this folder, if present.

## Fail signals

- The object landed outside the container footprint.
- The container was displaced or tipped during the approach/retreat.
- The gripper is still holding the object.
