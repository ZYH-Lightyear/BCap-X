"""Core data structures of the Visual Action Workspace (VAW).

Design rule: everything the agent can reference carries a short string id
(``obj1``, ``g2``, ``r3``), and the whole state can be summarised into a
compact JSON dict that goes into the model prompt. Heavy arrays (masks,
point clouds) stay in memory only: they are rendered onto the canvas but
never serialised into prompts or trace JSON.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _round_list(arr: Any, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in np.asarray(arr).reshape(-1)]


def _yaw_deg(rotation: Any) -> float:
    """Heading of a 3x3 rotation about world Z, in degrees.

    Reported instead of the raw matrix: nine numbers of an orthonormal basis are
    unreadable in a prompt, while "this box is rotated 23 deg" is actionable.
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


@dataclass
class Pose:
    """World-frame pose: position (3,) + quaternion wxyz (4,)."""

    position: np.ndarray
    quat_wxyz: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.quat_wxyz = np.asarray(self.quat_wxyz, dtype=np.float64).reshape(4)

    def copy(self) -> "Pose":
        return Pose(self.position.copy(), self.quat_wxyz.copy())

    def to_dict(self) -> dict[str, list[float]]:
        return {"position": _round_list(self.position), "quat_wxyz": _round_list(self.quat_wxyz)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Pose":
        return Pose(np.asarray(d["position"]), np.asarray(d["quat_wxyz"]))


# Canonical top-down orientation for the Franka/LIBERO env (pi about x).
TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class ObjectEntry:
    """A grounded scene object: referenceable evidence in the workspace."""

    object_id: str
    name: str
    score: float
    obs_revision: int
    mask: np.ndarray | None = None          # (H, W) bool, agentview frame
    box: list[float] | None = None          # [x1, y1, x2, y2] pixels
    points_world: np.ndarray | None = None  # (N, 3) world frame
    obb: dict[str, Any] | None = None       # {"center", "extent", "R"}

    @property
    def centroid_world(self) -> np.ndarray | None:
        if self.points_world is None or len(self.points_world) == 0:
            return None
        return np.median(self.points_world, axis=0)

    def summary(self, detail: bool = False) -> dict[str, Any]:
        """Compact by default; ``detail`` is for the focused object (see
        ``ActionState.summary``), which gets every number the canvas refuses to
        render as text."""
        out: dict[str, Any] = {
            "id": self.object_id,
            "name": self.name,
            "obs_revision": self.obs_revision,
        }
        c = self.centroid_world
        if c is not None:
            out["centroid_xyz"] = _round_list(c, 3)
        if self.obb is not None:
            out["obb_extent"] = _round_list(self.obb["extent"], 3)
        if not detail:
            return out

        out["score"] = round(self.score, 3)
        if self.points_world is not None:
            out["n_points"] = int(len(self.points_world))
        if self.box is not None:
            out["pixel_box"] = _round_list(self.box, 1)
        if self.obb is not None:
            out["obb_center"] = _round_list(self.obb["center"], 3)
            out["obb_yaw_deg"] = round(_yaw_deg(self.obb["R"]), 1)
        return out


@dataclass
class Candidate:
    """A proposed physical action, instantiated in the workspace.

    ``kind``:
      - "grasp":    approach + close at ``pose`` (from GraspNet or manual)
      - "place":    move + open at ``pose``
      - "waypoint": just move end-effector to ``pose``
    """

    candidate_id: str
    kind: str
    pose: Pose
    score: float = 0.0
    source: str = ""            # tool that produced it, e.g. "plan_grasp"
    object_id: str | None = None
    obs_revision: int = 0
    edited: bool = False        # True after nudge/rotate

    def summary(self, detail: bool = False) -> dict[str, Any]:
        """Compact drops the orientation and provenance: for a candidate the
        agent is not currently working on, position and score are what matter,
        and the full pose is one ``inspect``/``select`` away."""
        out: dict[str, Any] = {
            "id": self.candidate_id,
            "kind": self.kind,
            "position": _round_list(self.pose.position),
            "score": round(self.score, 3),
        }
        if self.object_id:
            out["object_id"] = self.object_id
        if self.edited:
            out["edited"] = True
        if detail:
            out["quat_wxyz"] = _round_list(self.pose.quat_wxyz)
            out["source"] = self.source
            out["obs_revision"] = self.obs_revision
        return out


@dataclass
class PreviewResult:
    """Terminal IK evidence for a candidate before motion planning is connected."""

    candidate_id: str
    ik_ok: bool
    orientation_used: str = "requested"     # from solve_ik fallback info
    predicted_ee: Pose | None = None
    #: Exact seven-joint solution returned by the same IK call that was checked
    #: during preview.  Rendering feeds this into URDF FK; it never reconstructs
    #: the hand link by guessing a TCP offset from ``predicted_ee``.
    joint_positions_rad: np.ndarray | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.joint_positions_rad is None:
            return
        joints = np.asarray(self.joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("preview joint_positions_rad must contain seven finite joints")
        self.joint_positions_rad = joints

    @property
    def feasible(self) -> bool:
        """Endpoint feasibility only; no trajectory/collision claim is implied."""

        return self.ik_ok

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "endpoint_ik_ok": self.ik_ok,
            "orientation_used": self.orientation_used,
            "trajectory_planned": False,
            "collision_checked": False,
            "notes": self.notes,
        }


@dataclass
class Receipt:
    """Execution receipt of a physical operation.

    ``discrepancy`` compares the receipt against what the operation claimed
    would happen — the candidate's preview for a commit, the requested offset
    for a step. It powers both failure attribution (written back into the state)
    and the asymmetric P_viol penalty at training time.
    """

    receipt_id: str
    op: str                                  # "commit" | "move_xyz" | "commit_gripper"
    candidate_id: str | None = None
    requested: Pose | None = None
    achieved: Pose | None = None
    #: Normalized finger opening, 0 (closed) .. 1 (fully open). Deliberately not
    #: called a width: it is a fraction, and the first live traces reported it to
    #: the model as "width=0.992", which reads as metres.
    gripper_opening: float | None = None
    pos_error_m: float | None = None
    discrepancy: dict[str, Any] = field(default_factory=dict)
    unpredicted_failure: bool = False
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.receipt_id, "op": self.op}
        if self.candidate_id:
            out["candidate_id"] = self.candidate_id
        if self.achieved is not None:
            out["achieved"] = self.achieved.to_dict()
        if self.gripper_opening is not None:
            out["gripper_opening"] = round(self.gripper_opening, 3)
        if self.pos_error_m is not None:
            out["pos_error_m"] = round(self.pos_error_m, 4)
        if self.discrepancy:
            out["discrepancy"] = self.discrepancy
        if self.unpredicted_failure:
            out["unpredicted_failure"] = True
        return out


@dataclass
class StepResult:
    """What one workspace step returns to the agent loop."""

    ok: bool
    op: str
    args: dict[str, Any]
    receipt_text: str
    canvas: np.ndarray | None                # (H, W, 3) uint8
    state_summary: dict[str, Any]
    physical: bool = False
