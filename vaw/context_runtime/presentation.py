"""Compile command state into policy-visible presentation data.

This module is the boundary between semantic/runtime state and raster code.
It deliberately owns no sensor capture and performs no physical action.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaw.context_runtime.model import ActionTarget, ContextState, Pose
from vaw.context_runtime.near_field import NearFieldPreview
from vaw.context_runtime.private import (
    ActionReviewArtifacts,
    ImaginationArtifacts,
    build_edit_summary,
)
from vaw.context_runtime.workspace import ContextWorkspace

PresentationArtifacts = ImaginationArtifacts | ActionReviewArtifacts


def compile_active_presentation(
    workspace: ContextWorkspace,
) -> tuple[ActionTarget | None, PresentationArtifacts | None, dict[str, Any] | None]:
    """Select the one virtual target currently visible to the policy."""

    state = workspace.state
    if state.imagination is not None:
        artifacts = workspace._private.imagination_artifacts
        target = state.imagination.target
        return target, artifacts, _target_presentation(
            workspace,
            target,
            artifacts,
            status="editing",
            action_id=None,
        )
    if state.action_review is not None:
        review = state.action_review
        artifacts = workspace._private.review_artifacts.get(review.action_id)
        return review.target, artifacts, _target_presentation(
            workspace,
            review.target,
            artifacts,
            status="review",
            action_id=review.action_id,
            intent=review.intent,
        )
    return None, None, None


def _target_presentation(
    workspace: ContextWorkspace,
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
    *,
    status: str,
    action_id: str | None,
    intent: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "target": target.summary(),
        "target_role": target_role(target, artifacts),
    }
    source_delta = source_surface_delta_base_m(workspace, target, artifacts)
    if source_delta is not None:
        result["source_surface_delta_base_m"] = _rounded(source_delta, 4)
    if action_id is not None:
        result["action_id"] = action_id
    if intent is not None:
        result["intent"] = intent
    plan = artifact_plan(artifacts)
    if plan is not None:
        result["prediction"] = plan.prediction.summary()
    if isinstance(artifacts, ImaginationArtifacts) and artifacts.latest_visual_edit:
        result["latest_edit"] = artifacts.latest_visual_edit.summary()
    edit_summary = (
        build_edit_summary(target, artifacts)
        if isinstance(artifacts, ImaginationArtifacts)
        else artifacts.edit_summary
        if isinstance(artifacts, ActionReviewArtifacts)
        else None
    )
    if edit_summary is not None:
        result["edit_summary"] = edit_summary.summary()
    if (
        isinstance(artifacts, ImaginationArtifacts)
        and artifacts.rotation_gizmo_frame is not None
    ):
        result["rotation_gizmo_frame"] = artifacts.rotation_gizmo_frame
    return result


def target_role(
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
) -> str:
    if target.pose is None:
        return "gripper_only"
    context = artifacts.planning_context if artifacts is not None else None
    if context is not None and context.source_kind == "grasp":
        return "grasp_contact"
    if context is not None and context.source_kind == "point":
        return "point_pose"
    return "relative_pose"


def source_surface_delta_base_m(
    workspace: ContextWorkspace,
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
) -> tuple[float, float, float] | None:
    """Return target-TCP to nearest current source surface in BASE frame."""

    if target.pose is None or artifacts is None or artifacts.planning_context is None:
        return None
    region_id = artifacts.planning_context.region_id
    if region_id is None:
        return None
    geometry = workspace._private.region_geometry.get(region_id)
    if geometry is None:
        return None
    points = np.asarray(geometry.filtered_object_points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        points = np.asarray(geometry.object_points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return None
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    target_xyz = np.asarray(target.pose.position_xyz, dtype=np.float64)
    deltas = points - target_xyz
    return tuple(float(value) for value in deltas[np.argmin(np.linalg.norm(deltas, axis=1))])


def observed_source_ref(artifacts: PresentationArtifacts | None) -> str | None:
    if artifacts is None or artifacts.planning_context is None:
        return None
    context = artifacts.planning_context
    return context.region_id or context.source_ref


def artifact_plan(artifacts: PresentationArtifacts | None):
    if isinstance(artifacts, ImaginationArtifacts):
        return artifacts.preview_plan
    if isinstance(artifacts, ActionReviewArtifacts):
        return artifacts.motion_plan
    return None


def compile_near_field_preview(
    state: ContextState,
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
) -> NearFieldPreview | None:
    """Compile current and previous virtual gripper geometry for one RGB-D view."""

    robot = state.robot
    if robot is None or robot.gripper_opening is None:
        return None
    plan = artifact_plan(artifacts)
    joints = (
        plan.prediction.joint_positions_rad
        if plan is not None and plan.prediction.solve_ik == "returned"
        else None
    )
    if target.pose is None:
        joints = robot.joint_positions_rad
    opening = (
        1.0
        if target.gripper == "open"
        else 0.0
        if target.gripper == "closed"
        else robot.gripper_opening
    )
    visual_edit = (
        artifacts.latest_visual_edit
        if isinstance(artifacts, ImaginationArtifacts)
        else None
    )
    previous_pose: Pose | None = (
        visual_edit.reference_pose
        if visual_edit is not None and visual_edit.kind in {"delta_move", "rotate"}
        else None
    )
    return NearFieldPreview(
        target_pose=target.pose,
        joint_positions_rad=joints,
        gripper_opening=opening,
        visual_edit=visual_edit,
        previous_target_pose=previous_pose,
        previous_gripper_opening=(
            robot.gripper_opening if previous_pose is not None else None
        ),
        rotation_gizmo_frame=(
            artifacts.rotation_gizmo_frame
            if isinstance(artifacts, ImaginationArtifacts)
            and artifacts.rotation_gizmo_frame in {"base", "tool"}
            else None
        ),
    )


def _rounded(values: tuple[float, ...], digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


__all__ = [
    "PresentationArtifacts",
    "artifact_plan",
    "compile_active_presentation",
    "compile_near_field_preview",
    "observed_source_ref",
    "source_surface_delta_base_m",
    "target_role",
]
