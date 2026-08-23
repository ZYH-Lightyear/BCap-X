"""Compile command state into policy-visible presentation data.

This module is the boundary between semantic/runtime state and raster code.
It deliberately owns no sensor capture and performs no physical action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime.gripper_mesh import load_panda_urdf_fk
from vaw.context_runtime.model import ActionTarget, ContextState, Pose
from vaw.context_runtime.near_field import (
    NearFieldPreview,
    PreviewGripperStyle,
    gravity_stable_contact_frame_quaternion,
)
from vaw.context_runtime.private import ActionArtifacts, build_edit_summary
from vaw.context_runtime.workspace import ContextWorkspace

PresentationArtifacts = ActionArtifacts


def compile_active_presentation(
    workspace: ContextWorkspace,
) -> tuple[ActionTarget | None, PresentationArtifacts | None, dict[str, Any] | None]:
    """Select the one virtual target currently visible to the policy."""

    state = workspace.state
    if state.action_proposal is not None:
        action = state.action_proposal
        artifacts = workspace._private.action_artifacts
        return (
            action.target,
            artifacts,
            _target_presentation(
                workspace,
                action.target,
                artifacts,
                status=(
                    "refining"
                    if state.imagination is not None
                    else "refined"
                    if action.refined
                    else "planned"
                ),
                action_id=action.action_id,
                intent=action.intent,
            ),
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
    if action_id is not None:
        result["action_id"] = action_id
    if intent is not None:
        result["intent"] = intent
    plan = artifact_plan(artifacts)
    if plan is not None:
        result["prediction"] = plan.prediction.summary()
    if artifacts is not None and artifacts.latest_visual_edit:
        result["latest_edit"] = artifacts.latest_visual_edit.summary()
    edit_summary = build_edit_summary(target, artifacts) if artifacts is not None else None
    if edit_summary is not None:
        result["edit_summary"] = edit_summary.summary()
    if (
        artifacts is not None and artifacts.rotation_gizmo_frame is not None
        and artifacts.rotation_gizmo_axis is not None
    ):
        result["rotation_gizmo_frame"] = artifacts.rotation_gizmo_frame
        result["rotation_gizmo_axis"] = artifacts.rotation_gizmo_axis
    return result


def target_role(
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
) -> str:
    context = artifacts.planning_context if artifacts is not None else None
    if context is not None and context.source_kind == "grasp":
        return "grasp_contact"
    if context is not None and context.source_kind == "point":
        return "point_pose"
    return "relative_pose"


def observed_source_ref(artifacts: PresentationArtifacts | None) -> str | None:
    if artifacts is None or artifacts.planning_context is None:
        return None
    context = artifacts.planning_context
    return context.region_id or context.source_ref


def artifact_plan(artifacts: PresentationArtifacts | None):
    return artifacts.motion_plan if artifacts is not None else None


def compile_near_field_preview(
    state: ContextState,
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
    *,
    gripper_style: PreviewGripperStyle = "fk-mesh",
    tcp_to_hand_local_xyz: np.ndarray | None = None,
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
    opening = robot.gripper_opening
    visual_edit = (
        artifacts.latest_visual_edit if artifacts is not None else None
    )
    previous_pose: Pose | None = (
        visual_edit.reference_pose
        if visual_edit is not None and visual_edit.kind in {"delta_move", "rotate"}
        else None
    )
    contact_frame_pose = _contact_frame_pose(target, artifacts, robot.tcp_pose)
    realized_tcp_pose = _realized_tcp_pose(joints, tcp_to_hand_local_xyz)
    return NearFieldPreview(
        target_pose=target.pose,
        joint_positions_rad=joints,
        gripper_opening=opening,
        gripper_style=gripper_style,
        realized_tcp_pose=realized_tcp_pose,
        visual_edit=visual_edit,
        previous_target_pose=previous_pose,
        previous_gripper_opening=(robot.gripper_opening if previous_pose is not None else None),
        rotation_gizmo_frame=(
            artifacts.rotation_gizmo_frame
            if artifacts is not None and artifacts.rotation_gizmo_frame in {"base", "tool"}
            else None
        ),
        rotation_gizmo_axis=(
            artifacts.rotation_gizmo_axis
            if artifacts is not None and artifacts.rotation_gizmo_axis in {"x", "y", "z"}
            else None
        ),
        contact_frame_quaternion_xyzw=(
            gravity_stable_contact_frame_quaternion(contact_frame_pose)
            if contact_frame_pose is not None
            else None
        ),
        contact_frame_position_xyz=(
            contact_frame_pose.position_xyz if contact_frame_pose is not None else None
        ),
    )


def _realized_tcp_pose(
    joints: tuple[float, ...] | None,
    tcp_to_hand_local_xyz: np.ndarray | None,
) -> Pose | None:
    """Recover the TCP actually realised by returned joints for line-art mode."""

    if joints is None or tcp_to_hand_local_xyz is None:
        return None
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        hand = fk.frame(np.asarray(joints, dtype=np.float64), "panda_hand")
        rotation = hand[:3, :3]
        position = hand[:3, 3] - rotation @ np.asarray(
            tcp_to_hand_local_xyz, dtype=np.float64
        ).reshape(3)
        quaternion = Rotation.from_matrix(rotation).as_quat()
        return Pose(tuple(position), tuple(quaternion))
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _contact_frame_pose(
    target: ActionTarget,
    artifacts: PresentationArtifacts | None,
    observed_tcp: Pose | None,
) -> Pose | None:
    """Return the immutable pose that defines one Contact Camera session."""

    if artifacts is not None:
        initial = artifacts.initial_target
        if initial is not None and initial.pose is not None:
            return initial.pose
    return target.pose or observed_tcp


__all__ = [
    "PresentationArtifacts",
    "artifact_plan",
    "compile_active_presentation",
    "compile_near_field_preview",
    "observed_source_ref",
    "target_role",
]
