---
name: Verify Placement
category: verification
description: "Independently check that a named object satisfies the requested placement relation."
---

# Verify Placement

## Purpose

Independently judge whether a named object satisfies the requested placement relation.

## When to use

Use after a release or insertion attempt, before the subgoal can report success.

## When NOT to use

Do not use before release, to command corrective motion, or when only stale
pre-release observations are available.

## Workflow

Use fresh external and wrist observations to check the requested placement
postcondition. Treat execution evidence as a claim, not truth. Report pass, fail,
or uncertainty with image artifacts and the current observation epoch. Never move
the robot or change the gripper state.

## Candidate Generation

Generate competing interpretations when visibility is ambiguous: placed, misplaced,
still held, or occluded.

## Local Checks

Check both the destination relation and that the object is no longer held when the
postcondition requires release.

## Failure Modes

Return uncertainty for occlusion or contradictory views. Return failure only when
fresh evidence supports a concrete violated relation. Emit `failed_placement` for
a visible target that is still held or misplaced, and `wrong_grounding` when the
inspected object is not the requested target.

## Clean Reusable Rules

Execution evidence identifies what to inspect; fresh observations determine the verdict.

## Weak Priors

Container overlap or proximity may support a placement claim, but should not replace
visual relation checks.

## Prohibited Shortcuts

- Do not move the robot or gripper.
- Do not accept the executor's success claim as verification.
- Do not use stale images.

## Artifacts to Save

- Fresh external-view image.
- Optional wrist image or relation overlay.
- Structured verifier report with observation epoch.

## Multimodal Evidence Contract

Consume execution evidence and a strictly newer post-release observation epoch.
Publish the selected scene/wrist images, target-relation evidence, gripper state,
and epoch. Never certify from executor text alone; route stale inputs as
`stale_observation`.

## Output Contract

Publish exactly `verifier_report`. For a pass, leave `failure_kind` empty. For a
concrete failure, set `failure_kind` to the matching declared exit condition so
the Manager can route recovery:

## Reference Code

```python
status = verdict["status"]  # pass | fail | uncertain
failure_kind = ""
if status == "fail":
    failure_kind = (
        "wrong_grounding"
        if verdict.get("state") == "wrong_object"
        else "failed_placement"
    )
NODE_RESULT = {
    "outputs": {
        "verifier_report": {
            "payload": {"verdict": verdict, "observation_epoch": observation_epoch},
            "confidence": confidence,
            "artifacts": artifacts,
        }
    },
    "recommended_next": "done" if status == "pass" else "recover_or_escalate",
    "failure_kind": failure_kind,
}
```
