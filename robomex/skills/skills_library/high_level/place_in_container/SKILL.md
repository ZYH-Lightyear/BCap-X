---
name: Place Held Object In Container
category: high_level
description: High-level placement of an already-held object into a named container, orchestrating container perception (segment + geometry) and the release action with a measured drop height.
---

# Place Held Object In Container

Release a held object into a named container. This is a compound skill: it
orchestrates the leaf skills below rather than calling raw APIs. Precondition: the
object is already grasped — if it is not, run `pick_object` first.

## When to use

An object is already held in the gripper and the task needs it released inside (or
on top of) a named container / receptacle / surface.

## Decomposition

Note: perception here targets the *container*, not the held object.

1. **segment_object** — segment the `container_name` into a mask + filtered 3D points.
2. **estimate_geometry** — estimate the container's center and `top_z`; the release
   height is computed from this.
3. **place_into_container** — carry the held object over the container center and
   release at `top_z + held_height/2 + clearance`.

The held object's geometry should already be known from the pick; only the container
needs fresh perception here.

## Postcondition

The previously held object rests inside the target container, and the gripper is
open and retreated.

## Verify

Authoritative success rubric: `ref/verify.md` (used by the Verifier Agent).
Quick self-check: a fresh mask of the object lies within the container footprint and the gripper is open.

## Failure modes

- Object landed outside: re-segment and re-grasp the object, then approach again
  with higher clearance over the container center.
- Container moved during the approach: re-segment + re-estimate the container before
  retrying the release.
