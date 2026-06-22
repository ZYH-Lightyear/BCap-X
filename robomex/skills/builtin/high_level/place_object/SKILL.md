---
name: Place Object
category: high_level
description: High-level placement of a held object into/onto a named target, orchestrating placement localization and the release action end-to-end.
---

# Place Object

Place the object currently held in the gripper into/onto a named target (a container,
a region, or another object). This is a compound skill: it does not call raw APIs
itself, it points you at the leaf skills you may need. Consult each leaf's guidance
when you write the code for that step; adapt it, do not copy it blindly. The live
observation is authoritative.

## When to use

The robot is already holding the target object (a pick has succeeded: gripper width is
above zero) and the task asks to put it somewhere named. If nothing is held yet, do the
pick first — this skill assumes the object is in hand.

## Building blocks (compose these yourself)

You — the Coding Agent — decide how to combine these, in what order, and whether to
loop or skip, based on each leaf skill's own "When to use" / experience and the live
scene. There is NO fixed observe-then-act pipeline and NO mandatory hand-off contract
between them; just write the code that gets this object placed.

- **find_placement** (observation) — ground the named placement target and compute a
  safe 3D release point above its opening / surface (with clearance). The usual starting
  point, since releasing needs to know where the target is.
- **release_at** (action) — move above that release point, lower with clearance, open the
  gripper to release, and retreat so the object stays put.

A typical composition is find_placement → release_at, but you are free to re-localize,
adjust the release height, or re-observe as the scene calls for it.

## Postcondition

The named object now rests in/on the named target and the gripper is empty (open).

## Verify

Authoritative success rubric: `reference/verify.md` (used by the Verifier Agent).
Quick self-check: the object is in/on the target in the after-frame and the gripper is open.

## Failure modes

- Object released too high and bounced out / toppled the target: lower the release point
  toward the target surface (smaller clearance) and retry once.
- Released over the wrong target: re-run find_placement with a more specific name
  (add color/position) before releasing.
- Gripper did not open / object still held: re-issue the release and confirm the
  gripper width is high before retreating.
