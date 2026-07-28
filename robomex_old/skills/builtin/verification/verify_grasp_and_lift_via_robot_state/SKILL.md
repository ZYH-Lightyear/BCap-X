---
name: Verify Grasp And Lift Via Robot State
category: verification
description: "Judge grasp success from a visible scene view plus gripper state, falling back to the wrist view only when the scene is occluded."
recommended_min_actions: 2
---

# Verify Grasp And Lift Via Robot State

## Purpose

Verify whether a recent grasp-and-lift attempt actually secured the target object.
This skill is a state verifier, not a motion primitive. It asks one VLM to jointly
interpret a camera image and the observed gripper state. The scene camera is primary;
the wrist camera is queried only when the scene is occluded.

## When to use

Use immediately after closing the gripper and lifting, before transport or placement.
Use it again after a suspicious motion, slip, or failed postcondition. It is especially
important for bowls, rims, handles, thin objects, and cases where gripper width alone is
ambiguous.

## When NOT to use

Do not call this verifier before a lift attempt, use it as a grasp planner, or
convert gripper width into a threshold-only verdict.

## Workflow

1. Capture one fresh observation and collect the complete observed gripper/robot state.
2. Call `verify_grasp_with_vlm(...)`. It first sends the scene image and gripper state
   together to the VLM.
3. The scene VLM first reports whether the gripper and relevant target region are
   visible and unobstructed. When they are visible, its grasp verdict is final and the
   wrist camera is not queried.
4. Only when the scene is occluded or visibility is uncertain, the helper sends the
   wrist image and the same gripper state to the VLM and uses that verdict.
5. Save the returned plain dict to `EVIDENCE["grasp_verification"]`.
6. Branch on the verdict: transport only on `held`, retry only after a concrete failure
   reason, and re-observe or escalate on `uncertain`.

## Candidate Generation

This skill generates a verification verdict, not a grasp candidate. The useful
candidate states are:

- `held`: target appears coupled to the gripper and is lifted.
- `still_on_surface`: target remains on the table or target support.
- `wrong_object`: gripper holds an object that is not the intended target.
- `empty_or_slipped`: gripper is closed or open but target is not visibly held.
- `uncertain`: views disagree or the target is occluded.

## Local Checks

- The upstream ExecutionEvidence contains runtime-captured `terminal_robot_state`;
  pass that state and the fresh state to the VLM as evidence, without converting either
  into a hand-written threshold verdict.
- The sidecar deliberately contains no hard-coded gripper-width or open-ratio ranges.
  Width semantics vary by robot, gripper, object, controller, and environment. Let the
  VLM interpret the supplied state jointly with the selected image.
- The scene image is the primary view. Do not query both cameras by default.
- Use the wrist image only after the scene reports `occluded` or uncertain visibility.
  Account for mounting offset: a held object may be off-center or near an image edge.
- Require structured categorical JSON; do not ask for coordinates.

## Failure Modes

- Scene reports visible: use its grasp verdict and do not ask the wrist camera.
- Scene reports occluded or uncertain visibility: do not accept its grasp guess; fall
  back to the wrist image with the same gripper state.
- Wrist evidence remains insufficient: return `uncertain`; do not fabricate success or
  failure.
- Wrist view sees an object but target identity is uncertain: return `uncertain` or
  `wrong_object`, with a target-specific reason.

## Clean Reusable Rules

- A grasp succeeds when the selected unobstructed view and gripper state jointly support
  `held`.
- Camera selection is conditional, not an evidence vote: scene first, wrist only after
  scene occlusion.
- Verification must happen at the grasp/place boundary, before committing to the next
  phase.

## Weak Priors

- Gripper state is contextual evidence for the VLM, never an independently thresholded
  verdict.
- A clearly visible target moving with the gripper supports `held`; a clearly visible
  target remaining on its support surface supports `still_on_surface`.
- Any environment-specific gripper calibration belongs in runtime state or learned
  context, not as constants in this reusable skill.

## Prohibited Shortcuts

- Do not declare success from gripper width alone.
- Do not encode fixed gripper-width or open-ratio thresholds in the sidecar or prompt.
- Do not query the wrist camera when the scene view is visible and unobstructed.
- Do not ignore `task_completed: false` or an unmet postcondition.
- Do not retry the same grasp pose without a changed observation or changed depth.
- Do not ask VLM for precise coordinates during verification.

## Artifacts to Save

- Store the scene verification image. Store the wrist image only when the fallback ran.
- Record the selected `view_used`, complete supplied `gripper_state`, visibility, state,
  confidence, reason, branch, and per-view VLM observations.

## Multimodal Evidence Contract

Consume upstream execution evidence plus a fresh post-lift observation epoch.
Scene RGB is primary; wrist RGB is conditional on occlusion. Publish the chosen
view, categorical VLM evidence, robot state, and epoch. `wrong_object` maps to
`wrong_grounding`; visible grasp misses map to `failed_grasp`; unresolved views
remain `uncertain`.

## Optional Sidecars

`scripts/verify_grasp_state.py` performs conditional VLM verification. It does not call
robot APIs or apply numeric gripper thresholds. The caller supplies images, state, and
the sandbox `query_vlm` function. All helpers return plain dicts.

Signatures:

- `build_scene_verification_question(target_name, gripper_state) -> str`.
- `build_wrist_verification_question(target_name, gripper_state) -> str`.
- `parse_verification_response(response) -> dict` returns
  `{"visibility", "status", "confidence", "reason"}` with validated categorical values.
- `verify_grasp_with_vlm(*, target_name, gripper_state, scene_image, wrist_image,
  query_vlm) -> dict` queries scene first and wrist only after scene occlusion. It
  returns `{"success", "state", "confidence", "reason", "view_used", "visibility",
  "branch", "gripper_state", "observations"}`.

Call pattern (fits in one or two run_python blocks):

- Capture a fresh observation; obtain scene image, wrist image, and gripper state.
- Call `verdict = verify_grasp_with_vlm(target_name=target_name,
  gripper_state=gripper_state, scene_image=scene_rgb, wrist_image=wrist_rgb,
  query_vlm=query_vlm)`.
- Do not call `query_vlm` separately and do not implement your own threshold fusion.
- Save `verdict` to `EVIDENCE["grasp_verification"]` and report it in finish.

## Reference Code

`verify_grasp_with_vlm` is a contracted canonical function: it is already defined in
your sandbox namespace when this skill loads — call it directly, no import. Do not
probe it with `dir()` or `inspect`. One block is enough.

```python
obs = get_observation()
verdict = verify_grasp_with_vlm(
    target_name=target_name,
    gripper_state=obs.get("gripper_state") or robot_state.get("gripper"),
    scene_image=scene_rgb,
    wrist_image=wrist_rgb,
    query_vlm=query_vlm,
)
EVIDENCE["grasp_verification"] = verdict
# verdict["state"] is one of: held, still_on_surface, wrong_object,
# empty_or_slipped, uncertain. Map held -> pass, uncertain -> uncertain,
# everything else -> fail.
status = (
    "pass" if verdict["state"] == "held"
    else "uncertain" if verdict["state"] == "uncertain"
    else "fail"
)
confidence_by_label = {"high": 0.9, "medium": 0.6, "low": 0.3}
confidence = confidence_by_label.get(str(verdict.get("confidence", "")).lower(), 0.2)
failure_kind = (
    ""
    if status == "pass"
    else "wrong_grounding"
    if verdict["state"] == "wrong_object"
    else "failed_grasp"
    if status == "fail"
    else ""
)
NODE_RESULT = {
    "outputs": {
        "verifier_report": {
            "payload": {
                "verdict": {"status": status, "reason": verdict.get("reason", "")},
                "state": verdict["state"],
                "view_used": verdict.get("view_used"),
                "confidence": verdict.get("confidence"),
            },
            "confidence": confidence,
            "artifacts": {},
        }
    },
    "recommended_next": "done" if status == "pass" else "recover_or_escalate",
    "failure_kind": failure_kind,
}
# finish with: {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}
```
