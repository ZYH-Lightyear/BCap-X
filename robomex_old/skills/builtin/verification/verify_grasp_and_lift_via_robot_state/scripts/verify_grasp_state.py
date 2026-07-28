"""Conditional visual grasp verification.

The scene camera is used whenever it clearly shows the gripper and target.  The
wrist camera is queried only when the scene view is occluded.  Gripper state is
passed to the VLM as context and is never interpreted with hard-coded numeric
thresholds in this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any


VERIFICATION_STATES = frozenset(
    {"held", "still_on_surface", "wrong_object", "empty_or_slipped", "uncertain"}
)
VISIBILITY_STATES = frozenset({"visible", "occluded", "uncertain"})
CONFIDENCE_STATES = frozenset({"high", "medium", "low"})


def _render_gripper_state(gripper_state: Mapping[str, Any] | None) -> str:
    """Render observed state verbatim; deliberately apply no numeric policy."""

    return json.dumps(
        dict(gripper_state or {}),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def build_scene_verification_question(
    target_name: str,
    gripper_state: Mapping[str, Any] | None,
) -> str:
    """Build the primary scene-camera verification question."""

    target = target_name.strip() or "the target object"
    state = _render_gripper_state(gripper_state)
    return f"""Verify whether the robot successfully grasped {target}.

Observed gripper/robot state:
{state}

Use the image and the observed state jointly. Do not invent fixed thresholds
for gripper width or open ratio.

First decide whether both the gripper and the relevant target region are
clearly visible enough in this scene image to judge their physical coupling.
Then judge the grasp state.

Return exactly one JSON object:
{{
  "visibility": "visible | occluded | uncertain",
  "status": "held | still_on_surface | wrong_object | empty_or_slipped | uncertain",
  "confidence": "high | medium | low",
  "reason": "one short evidence-based reason"
}}

If the gripper or relevant target region is occluded, set visibility to
"occluded" and status to "uncertain"; do not guess."""


def build_wrist_verification_question(
    target_name: str,
    gripper_state: Mapping[str, Any] | None,
) -> str:
    """Build the fallback wrist-camera question used after scene occlusion."""

    target = target_name.strip() or "the target object"
    state = _render_gripper_state(gripper_state)
    return f"""The external scene view was occluded. Using this wrist-camera
image, verify whether the robot successfully grasped {target}.

Observed gripper/robot state:
{state}

Use the image and the observed state jointly. Do not invent fixed thresholds
for gripper width or open ratio. Account for wrist-camera mounting offset: a
held object may appear off-center or near an image edge.

Return exactly one JSON object:
{{
  "visibility": "visible | occluded | uncertain",
  "status": "held | still_on_surface | wrong_object | empty_or_slipped | uncertain",
  "confidence": "high | medium | low",
  "reason": "one short evidence-based reason"
}}

If the evidence is insufficient, return status "uncertain"; do not guess."""


def parse_verification_response(response: Any) -> dict[str, str]:
    """Parse and validate a VLM JSON response into a stable plain dictionary."""

    if isinstance(response, Mapping):
        raw = dict(response)
    else:
        text = str(response or "").strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        try:
            raw = json.loads(candidate)
        except (TypeError, ValueError):
            embedded = re.search(r"\{.*\}", text, re.DOTALL)
            try:
                raw = json.loads(embedded.group(0)) if embedded else {}
            except (TypeError, ValueError):
                raw = {}

    visibility = str(raw.get("visibility", "uncertain")).strip().lower()
    status = str(raw.get("status", "uncertain")).strip().lower()
    confidence = str(raw.get("confidence", "low")).strip().lower()
    return {
        "visibility": visibility if visibility in VISIBILITY_STATES else "uncertain",
        "status": status if status in VERIFICATION_STATES else "uncertain",
        "confidence": confidence if confidence in CONFIDENCE_STATES else "low",
        "reason": str(raw.get("reason", "")).strip(),
    }


def _branch_for(status: str) -> str:
    if status == "held":
        return "transport_or_place"
    if status == "still_on_surface":
        return "retry_with_changed_depth_or_grasp_family"
    if status in {"wrong_object", "empty_or_slipped"}:
        return "open_reobserve_and_reground"
    return "reobserve_or_escalate_uncertainty"


def verify_grasp_with_vlm(
    *,
    target_name: str,
    gripper_state: Mapping[str, Any] | None,
    scene_image: Any,
    wrist_image: Any,
    query_vlm: Callable[..., Any],
) -> dict[str, Any]:
    """Verify a grasp with one scene query and an occlusion-only wrist fallback.

    ``query_vlm`` must support ``query_vlm(question, images=image)``.  No wrist
    query is made when the scene result reports ``visibility="visible"``.
    """

    scene_raw = query_vlm(
        build_scene_verification_question(target_name, gripper_state),
        images=scene_image,
    )
    scene = parse_verification_response(scene_raw)
    observations = {"scene": scene}

    if scene["visibility"] == "visible":
        result = scene
        view_used = "scene"
    else:
        wrist_raw = query_vlm(
            build_wrist_verification_question(target_name, gripper_state),
            images=wrist_image,
        )
        result = parse_verification_response(wrist_raw)
        observations["wrist"] = result
        view_used = "wrist"

    status = result["status"]
    return {
        "success": status == "held",
        "state": status,
        "confidence": result["confidence"],
        "reason": result["reason"],
        "view_used": view_used,
        "visibility": result["visibility"],
        "branch": _branch_for(status),
        "gripper_state": dict(gripper_state or {}),
        "observations": observations,
    }
