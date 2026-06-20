# Verify: Segment Object by Language

You are the Verifier Agent. Decide whether the segmentation correctly grounded the
named object. Return PASS only if every pass criterion holds; otherwise FAIL.

## Pass criteria

- The mask overlay covers exactly the named object and nothing else (no large
  spill onto the table or neighboring objects).
- The recovered point cloud is dense enough for geometry (at least a few hundred
  points after noise filtering).

## What to inspect

- The mask overlay debug image saved for this step.
- The point count reported after `filter_noise`.

## Fail signals

- Empty or near-empty mask / point set.
- The mask lands on the wrong object or bleeds across multiple objects.
