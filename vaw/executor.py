"""Commit execution: the only place the physical world changes.

Executes the selected candidate through the low-level controller, reads the
post-execution observation into a Receipt, and compares it against the
candidate's preview. A large deviation on a candidate whose preview said
"feasible" is an *unpredicted failure*: it is written back into the state
(failure attribution at inference time) and is the event the asymmetric
P_viol penalty counts at training time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaw.geometry import CONTACT_TO_HAND_M, shift_along_approach
from vaw.ik import solve_arm_ik
from vaw.state import ActionState
from vaw.types import Candidate, Pose, Receipt

# Positional deviation beyond this, on a feasible-previewed commit, counts
# as an unpredicted failure. Tolerance is deliberately loose for contact.
POS_TOLERANCE_M = 0.03

# Grasp commits approach from this height above the target in a second short
# motion. The controller's joint interpolation is rudimentary and one long
# motion visibly undershoots (~6 cm laterally in the first live run); the API
# docstring itself recommends keeping pre-grasp offsets around 0.075 m.
GRASP_APPROACH_M = 0.075

# The joint tracker's own convergence tolerance (``move_to_joints_blocking``
# stops at 0.01 rad, or silently at its step cap). Past this the arm is not
# where it was commanded, so a pose error is a stall rather than a bad solution.
JOINT_TOLERANCE_RAD = 0.02

# Normalized finger opening below which a closed gripper is holding nothing.
# Measured, not guessed: closing on air settles at 0.02, on the alphabet soup
# can at 0.13-0.64 (the fingers stop on the object).
EMPTY_GRIP_OPENING = 0.05


def _drive_to(api: Any, pose: Pose, z_approach: float) -> np.ndarray | None:
    """Move the arm to ``pose``; return the joint configuration commanded last.

    Spelled out as IK + ``move_to_joints`` rather than delegated to
    ``api.goto_pose`` (which does these same two steps) for two reasons: the IK
    has to be ours, see :mod:`vaw.ik`; and the receipt needs the configuration
    that was actually commanded. A 7-DoF arm has a null space of configurations
    per pose, so joints from any *other* IK call differ from the commanded ones
    by an arbitrary amount even when tracking was perfect — which makes them
    useless for telling "the arm never arrived" from "it arrived and the pose is
    still off".
    """
    if z_approach > 0.0:
        hover = Pose(pose.position + np.array([0.0, 0.0, z_approach]), pose.quat_wxyz)
        api.move_to_joints(solve_arm_ik(api, hover))
    joints = solve_arm_ik(api, pose)
    api.move_to_joints(joints)
    return joints


def _joint_error(obs: dict[str, Any], commanded: np.ndarray | None) -> float | None:
    """Norm of (achieved - commanded) arm joints, or None if either is missing."""
    if commanded is None or "robot_joint_pos" not in obs:
        return None
    achieved = np.asarray(obs["robot_joint_pos"], dtype=np.float64).reshape(-1)[:7]
    if achieved.size != 7 or commanded.size != 7:
        return None
    return float(np.linalg.norm(achieved - commanded))


def _read_ee(obs: dict[str, Any]) -> tuple[Pose, float]:
    state = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(-1)
    pose = Pose(state[:3].copy(), state[3:7].copy())
    gripper_opening = float(state[7]) if state.size > 7 else float("nan")
    return pose, gripper_opening


def _expected_hand_position(pose: Pose) -> np.ndarray:
    """Where the panda_hand link should end up if the fingers close at ``pose``.

    Candidate positions are fingertip-convention while the observation reports
    the hand link, a measured 1.1 cm behind the fingertips. Comparing the two
    raw makes every receipt report a phantom deviation — the first live run
    showed 0.14 m of "error" that was purely this convention gap.
    """
    return shift_along_approach(pose.position, pose.quat_wxyz, CONTACT_TO_HAND_M)


def execute_commit(
    api: Any,
    state: ActionState,
    cand: Candidate,
    *,
    z_approach: float | None = None,
) -> Receipt:
    """Move the real gripper to the candidate pose and settle the receipt.

    Grasp candidates get a two-stage approach (hover at ``GRASP_APPROACH_M``,
    then descend) unless the caller overrides ``z_approach``.
    """
    requested = cand.pose.copy()
    if z_approach is None:
        z_approach = GRASP_APPROACH_M if cand.kind == "grasp" else 0.0
    error_note = None
    commanded_joints = None
    try:
        commanded_joints = _drive_to(api, requested, z_approach)
    except Exception as exc:  # controller/IK failure is a receipt, not a crash
        error_note = f"{type(exc).__name__}: {exc}"

    obs = api.get_observation()
    achieved, gripper_opening = _read_ee(obs)
    # Both errors are measured between *hand* poses: achieved is a hand pose,
    # so the requested/predicted targets are mapped through the TCP offset.
    expected_hand = _expected_hand_position(requested)
    pos_error = float(np.linalg.norm(achieved.position - expected_hand))

    preview = state.previews.get(cand.candidate_id)
    discrepancy: dict[str, Any] = {}
    unpredicted = False
    if preview is not None and preview.predicted_ee is not None:
        pred_error = float(
            np.linalg.norm(
                achieved.position - _expected_hand_position(preview.predicted_ee)
            )
        )
        discrepancy = {
            "pred_pos_error_m": round(pred_error, 4),
            "preview_feasible": preview.feasible,
        }
        # Which of the two ways a commit misses its target? Comparing achieved
        # joints against the ones IK returned separates them, and they call for
        # opposite recoveries: a stalled arm means something is in the way (try
        # another approach), a tracked-but-wrong pose means the pose was never
        # achievable (try another candidate). Without this the receipt only says
        # "deviated" and the two are indistinguishable — the first live Canvas v2
        # run lost a grasp to a stall that looked exactly like an IK residual.
        joint_error = _joint_error(obs, commanded_joints)
        if joint_error is not None and pred_error > POS_TOLERANCE_M:
            discrepancy["joint_error_rad"] = round(joint_error, 4)
            discrepancy["cause"] = (
                "arm stalled before reaching the previewed joint configuration "
                "(something blocked it)"
                if joint_error > JOINT_TOLERANCE_RAD
                else "arm reached the previewed joints but the pose is still off "
                "(IK solution does not meet the target)"
            )
        unpredicted = preview.feasible and (
            pred_error > POS_TOLERANCE_M or error_note is not None
        )
    elif error_note is not None or pos_error > POS_TOLERANCE_M:
        discrepancy = {"note": "committed without preview"}

    if error_note:
        discrepancy["execution_error"] = error_note

    receipt = Receipt(
        receipt_id=state.next_id("r"),
        op="commit",
        candidate_id=cand.candidate_id,
        requested=requested,
        achieved=achieved,
        gripper_opening=gripper_opening,
        pos_error_m=pos_error,
        discrepancy=discrepancy,
        unpredicted_failure=unpredicted,
    )
    state.add_receipt(receipt)
    state.virtual_gripper = achieved.copy()
    return receipt


def execute_gripper(api: Any, state: ActionState, action: str) -> Receipt:
    """Open or close the gripper as its own physical operation (VIA-style)."""
    if action not in ("open", "close"):
        raise ValueError(f"gripper action must be 'open' or 'close', got '{action}'")
    if action == "open":
        api.open_gripper()
    else:
        api.close_gripper()

    obs = api.get_observation()
    achieved, gripper_opening = _read_ee(obs)
    state.gripper_open = action == "open"

    discrepancy: dict[str, Any] = {}
    # Closing onto nothing snaps near fully closed; a near-closed opening is a
    # strong "grasped nothing" signal the agent must see (v1 note failure mode).
    if action == "close" and np.isfinite(gripper_opening) and gripper_opening < EMPTY_GRIP_OPENING:
        discrepancy["warning"] = "gripper nearly fully closed: likely grasped nothing"

    receipt = Receipt(
        receipt_id=state.next_id("r"),
        op="commit_gripper",
        achieved=achieved,
        gripper_opening=gripper_opening,
        discrepancy=discrepancy,
    )
    state.add_receipt(receipt)
    return receipt
