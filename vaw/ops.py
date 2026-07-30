"""The workspace operation set: action space = tool protocol = log schema.

Each op is registered with a JSON-schema-ish parameter spec via ``@op``.
Cognitive ops mutate belief (state + canvas) only; ops with ``physical=True``
change the world. Op docstrings are agent-facing: they are exported verbatim
into the tool definitions in ``protocol.py``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from vaw.camera import resolve_view
from vaw.executor import execute_commit, execute_gripper, execute_move
from vaw.geometry import (
    GRASP_POSE_TO_CONTACT_M,
    mask_to_world_points,
    rotate_quat_wxyz,
    shift_along_approach,
)
from vaw.preview import run_preview
from vaw.types import Candidate, ObjectEntry, Pose, TOP_DOWN_QUAT_WXYZ

if TYPE_CHECKING:
    from vaw.workspace import Workspace


class OpError(RuntimeError):
    """Agent-visible operation failure (bad reference, tool returned nothing)."""


@dataclass
class OpSpec:
    name: str
    fn: Callable[..., str]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    physical: bool = False

    @property
    def description(self) -> str:
        return inspect.getdoc(self.fn) or ""


OPS: dict[str, OpSpec] = {}


def op(name: str, params: dict[str, dict[str, Any]] | None = None, physical: bool = False):
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        OPS[name] = OpSpec(name=name, fn=fn, params=params or {}, physical=physical)
        return fn

    return deco


# ===================================================================== #
# Cognitive ops: change belief and canvas only.                          #
# ===================================================================== #

@op("observe")
def observe(ws: "Workspace") -> str:
    """Take a fresh observation. Bumps the observation revision: evidence from
    earlier revisions is marked stale on the canvas and in the state summary.
    Use after any physical operation, or whenever the scene may have changed."""
    ws.refresh_observation()
    return f"observation refreshed (revision {ws.state.obs_revision})"


@op(
    "ground",
    params={"text": {"type": "string", "description": "object description, e.g. 'red mug'"}},
)
def ground(ws: "Workspace", text: str) -> str:
    """Locate an object by text. Detects it (VLM bbox), segments it (SAM3), and
    lifts its mask to 3D points. Adds a referenceable object (obj1, obj2, ...)
    to the workspace; its mask and label appear on the canvas."""
    cam = ws.obs[ws.camera_name]
    rgb = np.asarray(cam["images"]["rgb"])
    box = ws.api.vlm_bbox_detection(rgb, text)
    results = ws.api.segment_sam3_box_prompt(rgb, box)
    valid = [r for r in results if r.get("mask") is not None]
    if not valid:
        raise OpError(f"segmentation returned no mask for '{text}' (box={box})")
    best = max(valid, key=lambda r: float(r.get("score", 0.0)))
    mask = np.asarray(best["mask"], dtype=bool)

    points = mask_to_world_points(
        cam["images"]["depth"], mask, cam["intrinsics"], cam["pose_mat"]
    )
    obb = None
    if len(points) >= 16:
        try:
            obb = ws.api.get_oriented_bounding_box_from_3d_points(points)
        except Exception:
            obb = None

    entry = ws.state.add_object(
        ObjectEntry(
            object_id=ws.state.next_id("obj"),
            name=text,
            score=float(best.get("score", 0.0)),
            obs_revision=ws.state.obs_revision,
            mask=mask,
            box=[float(v) for v in box],
            points_world=points if len(points) else None,
            obb=obb,
        )
    )
    parts = [f"{entry.object_id} '{text}' grounded (score={entry.score:.2f})"]
    c = entry.centroid_world
    if c is not None:
        parts.append(f"centroid xyz={np.round(c, 3).tolist()}")
    if obb is not None:
        parts.append(f"obb extent={np.round(np.asarray(obb['extent']), 3).tolist()}")
    return "; ".join(parts)


@op(
    "inspect",
    params={"object_id": {"type": "string", "description": "object to inspect, e.g. 'obj1'"}},
)
def inspect_object(ws: "Workspace", object_id: str) -> str:
    """Look closely at one grounded object. Three things happen at once: the
    canvas focus inset zooms onto it (with every candidate of that object and
    their approach axes drawn), the state summary expands it to full detail
    while other objects stay compact, and this receipt reports its geometry.
    Use it before choosing between nearby grasps, or when the main view is too
    coarse to tell what you are looking at."""
    entry = ws.state.get_object(object_id)
    ws.state.focus_id = object_id
    parts = [f"focus set to {object_id} '{entry.name}' (rev {entry.obs_revision})"]
    c = entry.centroid_world
    if c is not None:
        parts.append(f"centroid={np.round(c, 3).tolist()}")
    if entry.points_world is not None:
        parts.append(f"{len(entry.points_world)} pts")
    if entry.obb is not None:
        parts.append(f"obb extent={np.round(np.asarray(entry.obb['extent']), 3).tolist()}")
    cands = ws.state.candidates_of(object_id)
    if cands:
        parts.append(f"candidates: {', '.join(c.candidate_id for c in cands)}")
    if entry.obs_revision < ws.state.obs_revision:
        parts.append("WARNING: stale (from an older observation)")
    return "; ".join(parts)


@op(
    "view",
    params={
        "preset": {
            "type": "string",
            "description": (
                "'agentview' (back to the physical camera) | 'top' | 'left' | 'right' "
                "| 'low' | 'close'"
            ),
            "required": False,
        },
        "azimuth_deg": {
            "type": "number",
            "description": "absolute world azimuth; within 75 deg of the physical camera",
            "required": False,
        },
        "elevation_deg": {
            "type": "number",
            "description": "0 = table level, 90 = straight down; allowed 10-85",
            "required": False,
        },
        "zoom": {"type": "number", "description": "1.0 = default, up to 2.5", "required": False},
    },
)
def view(
    ws: "Workspace",
    preset: str | None = None,
    azimuth_deg: float | None = None,
    elevation_deg: float | None = None,
    zoom: float | None = None,
) -> str:
    """Move the viewpoint of the main canvas view. This does not move the robot
    or take a new observation — it re-renders the same scene geometry from
    another angle, which is how you resolve occlusion and depth ambiguity ("is
    the gripper actually above the object, or just in front of it?").

    'agentview' shows the real camera image; any other angle shows the scene
    rebuilt as a point cloud, which is geometrically accurate but sparser, so
    come back to 'agentview' for appearance questions. Requests outside the
    supported envelope are clamped and the clamp is reported."""
    if preset is None and azimuth_deg is None and elevation_deg is None and zoom is None:
        raise OpError("view needs at least one of: preset, azimuth_deg, elevation_deg, zoom")
    notes = resolve_view(
        ws.state.view,
        preset=preset,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        zoom=zoom,
    )
    v = ws.state.view
    msg = (
        f"view={v.preset} az={v.azimuth_deg:.0f} el={v.elevation_deg:.0f} zoom={v.zoom:.1f}"
        f" ({'physical camera image' if v.is_physical else 'reconstructed point cloud'})"
    )
    return msg + ("; " + "; ".join(notes) if notes else "")


@op(
    "propose_grasps",
    params={
        "object_id": {"type": "string", "description": "grounded object to grasp"},
        "top_k": {"type": "integer", "description": "max candidates (default 5)", "required": False},
    },
)
def propose_grasps(ws: "Workspace", object_id: str, top_k: int = 5) -> str:
    """Run the grasp planner on a grounded object's mask. Adds up to top_k grasp
    candidates (g1, g2, ...) to the workspace, drawn on the canvas with ids."""
    entry = ws.state.get_object(object_id)
    if entry.mask is None:
        raise OpError(f"{object_id} has no mask; re-ground it first")
    cam = ws.obs[ws.camera_name]
    depth = cam["images"]["depth"]
    try:
        poses_cam, scores = ws.api.plan_grasp(
            depth, cam["intrinsics"], entry.mask.astype(np.int64)
        )
    except AssertionError as exc:
        raise OpError(f"grasp planner found no candidates for {object_id}: {exc}") from exc

    pose_mat = np.asarray(cam["pose_mat"], dtype=np.float64)
    order = np.argsort(np.asarray(scores))[::-1][: int(top_k)]
    ids = []
    for idx in order:
        world = pose_mat @ np.asarray(poses_cam[idx], dtype=np.float64)
        from scipy.spatial.transform import Rotation

        quat_xyzw = Rotation.from_matrix(world[:3, :3]).as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1)
        # Into the workspace's own convention: a candidate's position is where
        # the fingers close, not where the planner's frame origin sits.
        contact = shift_along_approach(world[:3, 3], quat_wxyz, GRASP_POSE_TO_CONTACT_M)
        cand = ws.state.add_candidate(
            Candidate(
                candidate_id=ws.state.next_id("g"),
                kind="grasp",
                pose=Pose(contact, quat_wxyz),
                score=float(scores[idx]),
                source="plan_grasp",
                object_id=object_id,
                obs_revision=ws.state.obs_revision,
            )
        )
        ids.append(f"{cand.candidate_id}({cand.score:.2f})")
    return f"added {len(ids)} grasp candidates for {object_id}: {', '.join(ids)}"


@op(
    "propose_pose",
    params={
        "kind": {"type": "string", "description": "'place' or 'waypoint'"},
        "position": {"type": "array", "description": "[x, y, z] world frame, meters"},
        "quat_wxyz": {
            "type": "array",
            "description": "[w, x, y, z]; omit for canonical top-down",
            "required": False,
        },
        "object_id": {"type": "string", "description": "related object", "required": False},
    },
)
def propose_pose(
    ws: "Workspace",
    kind: str,
    position: list[float],
    quat_wxyz: list[float] | None = None,
    object_id: str | None = None,
) -> str:
    """Manually add a place/waypoint candidate at a world-frame pose (e.g. above
    a target object's centroid). Omit quat_wxyz to use the top-down orientation."""
    if kind not in ("place", "waypoint"):
        raise OpError(f"kind must be 'place' or 'waypoint', got '{kind}'")
    quat = np.asarray(quat_wxyz, dtype=np.float64) if quat_wxyz else TOP_DOWN_QUAT_WXYZ.copy()
    cand = ws.state.add_candidate(
        Candidate(
            candidate_id=ws.state.next_id("p"),
            kind=kind,
            pose=Pose(np.asarray(position, dtype=np.float64), quat),
            source="propose_pose",
            object_id=object_id,
            obs_revision=ws.state.obs_revision,
        )
    )
    return f"added {kind} candidate {cand.candidate_id} at {np.round(cand.pose.position, 3).tolist()}"


@op(
    "select",
    params={"candidate_id": {"type": "string", "description": "candidate to select"}},
)
def select(ws: "Workspace", candidate_id: str) -> str:
    """Select a candidate as the pending action. The virtual gripper moves to it
    on the canvas. Only the selected candidate can be committed."""
    cand = ws.state.select(candidate_id)
    preview = ws.state.previews.get(candidate_id)
    note = f"; preview: {preview.notes}" if preview else "; not previewed yet"
    return f"selected {candidate_id} ({cand.kind}){note}"


@op(
    "nudge",
    params={
        "candidate_id": {"type": "string", "description": "candidate to edit"},
        "dx": {"type": "number", "description": "meters, world X", "required": False},
        "dy": {"type": "number", "description": "meters, world Y", "required": False},
        "dz": {"type": "number", "description": "meters, world Z", "required": False},
    },
)
def nudge(ws: "Workspace", candidate_id: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    """Translate a candidate by a metric offset (each axis clamped to ±0.1 m).
    This edits a plan on the canvas and does not move the robot — for that, see
    move_xyz. Voids the candidate's previous preview."""
    delta = np.clip([dx, dy, dz], -0.1, 0.1)
    cand = ws.state.get_candidate(candidate_id)
    cand.pose.position = cand.pose.position + delta
    cand.edited = True
    ws.state.invalidate_preview(candidate_id)
    if candidate_id == ws.state.selected_id:
        ws.state.virtual_gripper = cand.pose.copy()
    return f"{candidate_id} moved by {np.round(delta, 3).tolist()} to {np.round(cand.pose.position, 3).tolist()}"


@op(
    "rotate",
    params={
        "candidate_id": {"type": "string", "description": "candidate to edit"},
        "axis": {"type": "string", "description": "'x' | 'y' | 'z' (world axes)"},
        "degrees": {"type": "number", "description": "signed angle, within ±90"},
    },
)
def rotate(ws: "Workspace", candidate_id: str, axis: str, degrees: float) -> str:
    """Rotate a candidate about a world axis by a signed angle (clamped to ±90°).
    Voids the candidate's previous preview."""
    degrees = float(np.clip(degrees, -90.0, 90.0))
    cand = ws.state.get_candidate(candidate_id)
    cand.pose.quat_wxyz = rotate_quat_wxyz(cand.pose.quat_wxyz, axis, degrees)
    cand.edited = True
    ws.state.invalidate_preview(candidate_id)
    if candidate_id == ws.state.selected_id:
        ws.state.virtual_gripper = cand.pose.copy()
    return f"{candidate_id} rotated {degrees:.0f} deg about {axis}"


@op(
    "preview",
    params={"candidate_id": {"type": "string", "description": "candidate to preview"}},
)
def preview_op(ws: "Workspace", candidate_id: str) -> str:
    """Check endpoint IK without moving the robot and render the resulting
    terminal gripper through URDF FK. No trajectory or collision claim is made
    until a motion planner supplies the joint sequence."""
    cand = ws.state.get_candidate(candidate_id)
    result = run_preview(ws.api, ws.state, cand)
    return f"preview {candidate_id}: {result.notes}"


# ===================================================================== #
# Physical ops: the only operations that change the world.               #
# ===================================================================== #

@op("commit", physical=True)
def commit(ws: "Workspace") -> str:
    """Execute the selected candidate on the real robot. Returns a receipt with
    the achieved pose and, if the candidate was previewed, the deviation from
    the prediction. Unpredicted failures are flagged explicitly."""
    cand = ws.state.selected
    if cand is None:
        raise OpError("no candidate selected; call select first")
    receipt = execute_commit(ws.api, ws.state, cand)
    ws.refresh_observation()
    parts = [f"{receipt.receipt_id}: reached {np.round(receipt.achieved.position, 3).tolist()}"]
    if receipt.discrepancy:
        parts.append(f"discrepancy={receipt.discrepancy}")
    if receipt.unpredicted_failure:
        parts.append("UNPREDICTED FAILURE: endpoint IK succeeded but execution deviated")
    return "; ".join(parts)


@op(
    "move_xyz",
    params={
        "dx": {"type": "number", "description": "meters, world X", "required": False},
        "dy": {"type": "number", "description": "meters, world Y", "required": False},
        "dz": {"type": "number", "description": "meters, world Z", "required": False},
    },
    physical=True,
)
def move_xyz(ws: "Workspace", dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    """Move the real gripper by a small world-frame offset, keeping its current
    orientation (each axis clamped to ±0.05 m). Unlike nudge, which only edits a
    candidate on the canvas, this moves the robot immediately — no candidate and
    no preview. Use it to close the last few centimeters onto a target, to back
    off after a failed grasp, or to correct a commit that stopped short; use
    candidates and commit for larger repositioning. The receipt reports where the
    gripper actually ended up and by how much it missed."""
    receipt = execute_move(ws.api, ws.state, np.array([dx, dy, dz], dtype=np.float64))
    ws.refresh_observation()
    parts = [
        f"{receipt.receipt_id}: gripper at {np.round(receipt.achieved.position, 3).tolist()}"
    ]
    if receipt.pos_error_m is not None and receipt.pos_error_m > 0.005:
        parts.append(f"missed the step by {receipt.pos_error_m:.3f} m")
    if receipt.discrepancy:
        parts.append(f"discrepancy={receipt.discrepancy}")
    return "; ".join(parts)


@op(
    "commit_gripper",
    params={"action": {"type": "string", "description": "'open' or 'close'"}},
    physical=True,
)
def commit_gripper(ws: "Workspace", action: str) -> str:
    """Open or close the gripper (its own physical step; do not combine with a
    move). The receipt reports the resulting width: near fully closed after a
    'close' means the gripper likely grasped nothing."""
    receipt = execute_gripper(ws.api, ws.state, action)
    ws.refresh_observation()
    msg = (
        f"{receipt.receipt_id}: gripper {action}, "
        f"opening={receipt.gripper_opening:.3f} (0=closed, 1=open)"
    )
    if receipt.discrepancy.get("warning"):
        msg += f"; WARNING: {receipt.discrepancy['warning']}"
    return msg


@op(
    "done",
    params={
        "success": {"type": "boolean", "description": "whether you believe the task is complete"}
    },
    physical=True,
)
def done(ws: "Workspace", success: bool) -> str:
    """End the episode, declaring task success or failure."""
    ws.finished = True
    ws.claimed_success = bool(success)
    return f"episode ended (claimed success={success})"
