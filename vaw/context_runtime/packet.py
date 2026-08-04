"""Deterministic compiler for the fixed-size M1.3.1 Dynamic Context Canvas."""

from __future__ import annotations

import base64
import io
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from vaw.context_runtime.geometry import project_world_to_pixel
from vaw.context_runtime.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    rasterize_silhouette,
)
from vaw.context_runtime.model import ContextState, PointEvidence, Pose, RobotState
from vaw.context_runtime.workspace import ContextWorkspace

CONTEXT_SCHEMA = "vaw-context-v2"
CONTEXT_WEB_SCHEMA_VERSION = 3
CONTEXT_WIDTH = 1440
CONTEXT_HEIGHT = 1080

BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
VIOLET = (124, 58, 237)

DecisionMode = Literal[
    "idle",
    "grounding",
    "candidates",
    "proposal",
    "receipt",
    "error",
    "terminal",
]


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    query: str
    bbox_xyxy_px: tuple[float, float, float, float]
    source_revision: int
    raster_id: str
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.region_id,
            "query": self.query,
            "bbox": _rounded(self.bbox_xyxy_px, 2),
            "sourceRevision": self.source_revision,
            "rasterId": self.raster_id,
        }
        if self.within_region_id is not None:
            out["withinRegionId"] = self.within_region_id
        return out


@dataclass(frozen=True)
class PointSpec:
    point_id: str
    query: str
    pixel_xy: tuple[float, float]
    position_xyz: tuple[float, float, float]
    source_revision: int
    raster_id: str
    within_region_id: str | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.point_id,
            "query": self.query,
            "pixel": _rounded(self.pixel_xy, 2),
            "position": _rounded(self.position_xyz, 5),
            "sourceRevision": self.source_revision,
            "rasterId": self.raster_id,
        }
        if self.within_region_id is not None:
            out["withinRegionId"] = self.within_region_id
        return out


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    kind: str
    source_ref: str | None
    target_pose: Pose
    delta_from_anchor_xyz_m: tuple[float, float, float] | None
    approach_vector_base: tuple[float, float, float] | None
    solve_ik: str
    source_revision: int
    raster_id: str

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.candidate_id,
            "kind": self.kind,
            "targetPose": self.target_pose.summary(),
            "deltaFromAnchor": (
                _rounded(self.delta_from_anchor_xyz_m, 5)
                if self.delta_from_anchor_xyz_m is not None
                else None
            ),
            "approachVector": (
                _rounded(self.approach_vector_base, 4)
                if self.approach_vector_base is not None
                else None
            ),
            "solveIk": self.solve_ik,
            "sourceRevision": self.source_revision,
            "rasterId": self.raster_id,
        }
        if self.source_ref is not None:
            out["sourceRef"] = self.source_ref
        return out


@dataclass(frozen=True)
class WorldContextSpec:
    task_prompt: str
    agentview_raster_id: str
    wrist_raster_id: str | None
    robot: RobotState | None
    active_action: dict[str, Any] | None
    latest_event: dict[str, Any] | None
    last_receipt: dict[str, Any] | None

    def summary(self) -> dict[str, Any]:
        return {
            "taskPrompt": self.task_prompt,
            "agentviewRasterId": self.agentview_raster_id,
            "wristRasterId": self.wrist_raster_id,
            "robot": self.robot.summary() if self.robot is not None else None,
            "activeAction": self.active_action,
            "latestEvent": self.latest_event,
            "lastReceipt": self.last_receipt,
        }


@dataclass(frozen=True)
class EvidenceCatalogSpec:
    regions: tuple[RegionSpec, ...]
    points: tuple[PointSpec, ...]
    candidates: tuple[CandidateSpec, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "regions": [item.summary() for item in self.regions],
            "points": [item.summary() for item in self.points],
            "candidates": [item.summary() for item in self.candidates],
        }


@dataclass(frozen=True)
class DecisionWorkspaceSpec:
    mode: DecisionMode
    region_ids: tuple[str, ...] = ()
    point_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    action_id: str | None = None
    primary_raster_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "regionIds": list(self.region_ids),
            "pointIds": list(self.point_ids),
            "candidateIds": list(self.candidate_ids),
            "actionId": self.action_id,
            "primaryRasterId": self.primary_raster_id,
        }


@dataclass(frozen=True)
class ContextPacket:
    revision: int
    world: WorldContextSpec
    catalog: EvidenceCatalogSpec
    decision: DecisionWorkspaceSpec
    rasters: dict[str, np.ndarray]
    schema: str = CONTEXT_SCHEMA

    def summary(self) -> dict[str, Any]:
        """JSON-safe packet metadata; RGB arrays are represented by ids."""

        return {
            "schema": self.schema,
            "revision": self.revision,
            "viewport": {"width": CONTEXT_WIDTH, "height": CONTEXT_HEIGHT},
            "world": self.world.summary(),
            "catalog": self.catalog.summary(),
            "decision": self.decision.summary(),
            "rasterIds": list(self.rasters),
        }

    def manifest(self) -> dict[str, Any]:
        active = self.world.active_action
        return {
            "revision": self.revision,
            "active_action_id": active.get("action_id") if active is not None else None,
            "valid_region_ids": [item.region_id for item in self.catalog.regions],
            "valid_point_ids": [item.point_id for item in self.catalog.points],
            "valid_candidate_ids": [item.candidate_id for item in self.catalog.candidates],
        }

    def web_snapshot(self, *, render_id: str) -> dict[str, Any]:
        snapshot = self.summary()
        snapshot.update(
            {
                "schemaVersion": CONTEXT_WEB_SCHEMA_VERSION,
                "renderId": render_id,
                "rasters": {key: encode_png_data_url(value) for key, value in self.rasters.items()},
            }
        )
        return snapshot


class ContextCompiler:
    """Trusted presenter: private geometry in, policy-visible packet out."""

    def compile(self, workspace: ContextWorkspace) -> ContextPacket:
        state = workspace.state
        private = workspace._private
        camera = private.camera(workspace.camera_name)
        rgb = _rgb(camera)
        wrist = None
        wrist_camera = None
        with suppress(RuntimeError, ValueError):
            wrist_camera = private.camera(workspace.wrist_camera_name)
            wrist = _rgb(wrist_camera)

        # The persistent world view is deliberately sensor-clean.  All
        # grounding, self and imagination overlays belong to the dynamic
        # decision workspace below it.
        rasters: dict[str, np.ndarray] = {"agentview": rgb.copy()}
        if wrist is not None:
            rasters["wrist"] = wrist.copy()

        region_specs = self._compile_regions(state, rgb, private.region_masks, rasters)
        point_specs = self._compile_points(state, rgb, rasters)
        candidate_specs = self._compile_candidates(state, rgb, camera, rasters)
        decision = _decision_spec(state)
        if decision.mode == "proposal" and state.active_action is not None:
            raster_id = "decision:proposal"
            rasters[raster_id] = _proposal_raster(rgb, state, camera, state.active_action)
            decision = DecisionWorkspaceSpec(
                mode=decision.mode,
                region_ids=decision.region_ids,
                point_ids=decision.point_ids,
                candidate_ids=decision.candidate_ids,
                action_id=decision.action_id,
                primary_raster_id=raster_id,
            )
        elif decision.mode == "receipt":
            raster_id = "decision:post_action"
            target = (
                state.last_spatial_target.target_pose
                if state.last_spatial_target is not None
                else None
            )
            rasters[raster_id] = _post_action_raster(rgb, camera, target)
            decision = DecisionWorkspaceSpec(
                mode=decision.mode,
                primary_raster_id=raster_id,
            )

        latest = state.recent_calls[-1] if state.recent_calls else None
        packet = ContextPacket(
            revision=state.observation_revision,
            world=WorldContextSpec(
                task_prompt=state.task_prompt,
                agentview_raster_id="agentview",
                wrist_raster_id="wrist" if wrist is not None else None,
                robot=state.robot,
                active_action=(
                    state.active_action.summary() if state.active_action is not None else None
                ),
                latest_event=latest.summary() if latest is not None else None,
                last_receipt=(
                    _receipt_presentation(state.last_receipt)
                    if state.last_receipt is not None
                    else None
                ),
            ),
            catalog=EvidenceCatalogSpec(
                regions=tuple(region_specs),
                points=tuple(point_specs),
                candidates=tuple(candidate_specs),
            ),
            decision=decision,
            rasters=rasters,
        )
        _validate_packet(packet)
        return packet

    @staticmethod
    def _compile_regions(
        state: ContextState,
        rgb: np.ndarray,
        region_masks: dict[str, np.ndarray],
        rasters: dict[str, np.ndarray],
    ) -> list[RegionSpec]:
        specs: list[RegionSpec] = []
        for region in state.regions.values():
            raster_id = f"region:{region.region_id}"
            linked_points = tuple(
                point
                for point in state.points.values()
                if point.within_region_id == region.region_id
            )
            rasters[raster_id] = _region_crop(
                rgb,
                region.bbox_xyxy_px,
                region_masks.get(region.region_id),
                linked_points,
            )
            specs.append(
                RegionSpec(
                    region_id=region.region_id,
                    query=region.query,
                    bbox_xyxy_px=region.bbox_xyxy_px,
                    source_revision=region.source_revision,
                    raster_id=raster_id,
                    within_region_id=region.within_region_id,
                )
            )
        return specs

    @staticmethod
    def _compile_points(
        state: ContextState,
        rgb: np.ndarray,
        rasters: dict[str, np.ndarray],
    ) -> list[PointSpec]:
        specs: list[PointSpec] = []
        for point in state.points.values():
            raster_id = f"point:{point.point_id}"
            rasters[raster_id] = _point_crop(rgb, point.pixel_xy, point.point_id)
            specs.append(
                PointSpec(
                    point_id=point.point_id,
                    query=point.query,
                    pixel_xy=point.pixel_xy,
                    position_xyz=point.position_xyz,
                    source_revision=point.source_revision,
                    raster_id=raster_id,
                    within_region_id=point.within_region_id,
                )
            )
        return specs

    @staticmethod
    def _compile_candidates(
        state: ContextState,
        rgb: np.ndarray,
        camera: dict[str, Any],
        rasters: dict[str, np.ndarray],
    ) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for candidate in state.candidates.values():
            raster_id = f"candidate:{candidate.candidate_id}"
            source = (
                state.regions.get(candidate.source_ref)
                if candidate.source_ref is not None
                else None
            )
            anchor = next(
                (
                    point
                    for point in state.points.values()
                    if point.within_region_id == candidate.source_ref
                ),
                None,
            )
            rotation = Rotation.from_quat(
                np.asarray(candidate.target_pose.quaternion_xyzw)
            ).as_matrix()
            approach = tuple(float(value) for value in rotation[:, 2])
            delta = None
            if anchor is not None:
                delta = tuple(
                    float(value)
                    for value in (
                        np.asarray(candidate.target_pose.position_xyz)
                        - np.asarray(anchor.position_xyz)
                    )
                )
            robot_mask = None
            gripper_mask = None
            if candidate.prediction.joint_positions_rad is not None:
                opening = (
                    state.robot.gripper_opening
                    if state.robot is not None and state.robot.gripper_opening is not None
                    else 1.0
                )
                robot_mask = _robot_mask(
                    candidate.prediction.joint_positions_rad,
                    opening,
                    camera,
                    rgb.shape[1],
                    rgb.shape[0],
                )
                gripper_mask = _gripper_mask(
                    candidate.prediction.joint_positions_rad,
                    opening,
                    camera,
                    rgb.shape[1],
                    rgb.shape[0],
                )
            rasters[raster_id] = _candidate_crop(
                rgb,
                source.bbox_xyxy_px if source is not None else None,
                candidate.target_pose,
                camera,
                robot_mask=robot_mask,
                gripper_mask=gripper_mask,
            )
            specs.append(
                CandidateSpec(
                    candidate_id=candidate.candidate_id,
                    kind=candidate.kind,
                    source_ref=candidate.source_ref,
                    target_pose=candidate.target_pose,
                    delta_from_anchor_xyz_m=delta,
                    approach_vector_base=approach,
                    solve_ik=candidate.prediction.solve_ik,
                    source_revision=candidate.source_revision,
                    raster_id=raster_id,
                )
            )
        return specs


def _decision_spec(state: ContextState) -> DecisionWorkspaceSpec:
    """Derive presentation from result shape, never from an action phase."""

    if not state.recent_calls:
        return DecisionWorkspaceSpec(mode="idle")
    latest = state.recent_calls[-1]
    result = latest.result
    if "error" in result:
        return DecisionWorkspaceSpec(mode="error")
    if latest.function_name == "done":
        return DecisionWorkspaceSpec(mode="terminal")

    action_id = result.get("action_id")
    action = state.active_action
    if isinstance(action_id, str) and action is not None and action.action_id == action_id:
        regions, points, candidates = _action_references(state, action.source_ref)
        return DecisionWorkspaceSpec(
            mode="proposal",
            region_ids=regions,
            point_ids=points,
            candidate_ids=candidates,
            action_id=action_id,
        )

    candidate_ids = result.get("candidate_ids")
    if isinstance(candidate_ids, list):
        valid_candidates = tuple(
            str(value) for value in candidate_ids if str(value) in state.candidates
        )[:5]
        region_ids = _candidate_source_regions(state, valid_candidates)
        return DecisionWorkspaceSpec(
            mode="candidates",
            region_ids=region_ids,
            candidate_ids=valid_candidates,
        )

    region_id = result.get("region_id")
    point_id = result.get("point_id")
    if isinstance(region_id, str) or isinstance(point_id, str):
        regions, points = _grounding_references(state, region_id, point_id)
        primary = None
        if isinstance(point_id, str) and point_id in state.points:
            primary = f"point:{point_id}"
        elif isinstance(region_id, str) and region_id in state.regions:
            primary = f"region:{region_id}"
        return DecisionWorkspaceSpec(
            mode="grounding",
            region_ids=regions,
            point_ids=points,
            primary_raster_id=primary,
        )

    # A refreshed observation is the semantic receipt event. The private
    # receipt identifier is an audit handle and must not control presentation.
    if latest.revision_after != latest.revision_before:
        return DecisionWorkspaceSpec(mode="receipt")
    return DecisionWorkspaceSpec(mode="idle")


def _receipt_presentation(receipt: Any) -> dict[str, Any]:
    """Return only receipt facts that the raster actually communicates."""

    out: dict[str, Any] = {"function_name": receipt.function_name}
    if receipt.action_id is not None:
        out["action_id"] = receipt.action_id
    if receipt.position_error_m is not None:
        out["position_error_m"] = round(float(receipt.position_error_m), 6)
    if receipt.gripper_opening is not None:
        out["gripper_opening"] = round(float(receipt.gripper_opening), 6)
    if receipt.discrepancy:
        out["discrepancy"] = dict(receipt.discrepancy)
    return out


def _grounding_references(
    state: ContextState,
    region_id: Any,
    point_id: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    regions: list[str] = []
    points: list[str] = []
    if isinstance(region_id, str) and region_id in state.regions:
        region = state.regions[region_id]
        if region.within_region_id in state.regions:
            regions.append(region.within_region_id)
        regions.append(region_id)
        points.extend(
            point.point_id for point in state.points.values() if point.within_region_id == region_id
        )
    if isinstance(point_id, str) and point_id in state.points:
        point = state.points[point_id]
        if point.within_region_id in state.regions:
            regions.append(point.within_region_id)
        points.append(point_id)
    return _unique(regions), _unique(points)


def _candidate_source_regions(
    state: ContextState, candidate_ids: tuple[str, ...]
) -> tuple[str, ...]:
    return _unique(
        candidate.source_ref
        for candidate_id in candidate_ids
        if (candidate := state.candidates.get(candidate_id)) is not None
        and candidate.source_ref in state.regions
    )


def _action_references(
    state: ContextState, source_ref: str | None
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if source_ref is None:
        return (), (), ()
    point = state.points.get(source_ref)
    if point is not None:
        regions = (point.within_region_id,) if point.within_region_id in state.regions else ()
        return regions, (point.point_id,), ()
    candidate = state.candidates.get(source_ref)
    if candidate is not None:
        related = tuple(
            item.candidate_id
            for item in state.candidates.values()
            if item.source_ref == candidate.source_ref
        )[:5]
        regions = (candidate.source_ref,) if candidate.source_ref in state.regions else ()
        points = tuple(
            point.point_id
            for point in state.points.values()
            if point.within_region_id == candidate.source_ref
        )
        return regions, points, related
    if source_ref in state.regions:
        return (source_ref,), (), ()
    return (), (), ()


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if isinstance(value, str)))


def _validate_packet(packet: ContextPacket) -> None:
    if packet.schema != CONTEXT_SCHEMA:
        raise ValueError(f"unsupported Context schema {packet.schema!r}")
    for raster_id, raster in packet.rasters.items():
        array = np.asarray(raster)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError(f"raster {raster_id!r} must be HxWx3 uint8, got {array.shape}")
    encoded_keys = {key.replace("_", "").lower() for key in _walk_keys(packet.summary())}
    for forbidden in (
        "intrinsics",
        "posemat",
        "rawmask",
        "depth",
        "cloud",
        "envsuccess",
        "reward",
        "privileged",
        "score",
        "obb",
    ):
        if forbidden in encoded_keys:
            raise ValueError(f"private field {forbidden!r} escaped into ContextPacket")


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _region_crop(
    rgb: np.ndarray,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray | None,
    linked_points: tuple[PointEvidence, ...] = (),
) -> np.ndarray:
    left, top, right, bottom = _expanded_bounds(bbox, rgb.shape[1], rgb.shape[0], ratio=0.18)
    image = Image.fromarray(rgb[top:bottom, left:right]).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    if mask is not None and mask.shape == rgb.shape[:2]:
        local = mask[top:bottom, left:right]
        overlay = np.zeros((*local.shape, 4), dtype=np.uint8)
        overlay[local] = (*BLUE, 42)
        image = Image.alpha_composite(
            image.convert("RGBA"), Image.fromarray(overlay, mode="RGBA")
        ).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        ys, xs = np.nonzero(mask_outline(local))
        for x, y in zip(xs, ys, strict=True):
            draw.point((int(x), int(y)), fill=(*BLUE, 255))
    for point in linked_points:
        local = (point.pixel_xy[0] - left, point.pixel_xy[1] - top)
        _cross(draw, local, GREEN, radius=4, width=2)
    return np.asarray(image, dtype=np.uint8)


def _point_crop(rgb: np.ndarray, pixel: tuple[float, float], label: str) -> np.ndarray:
    radius = max(64, min(rgb.shape[0], rgb.shape[1]) // 7)
    bbox = (
        pixel[0] - radius,
        pixel[1] - radius,
        pixel[0] + radius,
        pixel[1] + radius,
    )
    left, top, right, bottom = _expanded_bounds(bbox, rgb.shape[1], rgb.shape[0], ratio=0)
    image = Image.fromarray(rgb[top:bottom, left:right]).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    local = (pixel[0] - left, pixel[1] - top)
    _cross(draw, local, GREEN, radius=10, width=4)
    _label(draw, (local[0] + 11, local[1] - 10), label, GREEN)
    return np.asarray(image, dtype=np.uint8)


def _candidate_crop(
    rgb: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    pose: Pose,
    camera: dict[str, Any],
    *,
    robot_mask: np.ndarray | None,
    gripper_mask: np.ndarray | None,
) -> np.ndarray:
    has_robot_mask = robot_mask is not None and robot_mask.any()
    has_gripper_mask = gripper_mask is not None and gripper_mask.any()
    glyph = None if has_gripper_mask else _candidate_gripper_glyph(pose, camera)
    if bbox is not None:
        crop_box = bbox
    elif has_gripper_mask:
        ys, xs = np.nonzero(gripper_mask)
        crop_box = (
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        )
    elif glyph is not None:
        crop_box = (
            float(glyph[:, 0].min()),
            float(glyph[:, 1].min()),
            float(glyph[:, 0].max()),
            float(glyph[:, 1].max()),
        )
    else:
        crop_box = (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0]))
    # The crop follows the target object and hand, not the entire arm.  The
    # latter remains useful context inside the crop but must not zoom the
    # interaction down until all candidates look alike.
    if has_gripper_mask:
        ys, xs = np.nonzero(gripper_mask)
        crop_box = (
            min(float(crop_box[0]), float(xs.min())),
            min(float(crop_box[1]), float(ys.min())),
            max(float(crop_box[2]), float(xs.max())),
            max(float(crop_box[3]), float(ys.max())),
        )
    if glyph is not None:
        crop_box = (
            min(float(crop_box[0]), float(glyph[:, 0].min())),
            min(float(crop_box[1]), float(glyph[:, 1].min())),
            max(float(crop_box[2]), float(glyph[:, 0].max())),
            max(float(crop_box[3]), float(glyph[:, 1].max())),
        )
    crop_box = _fit_box_aspect(crop_box, target_aspect=1.15)
    left, top, right, bottom = _expanded_bounds(
        crop_box,
        rgb.shape[1],
        rgb.shape[0],
        ratio=0.24 if has_gripper_mask else 0.18,
    )
    raster = rgb[top:bottom, left:right].copy()
    local_robot = robot_mask[top:bottom, left:right] if has_robot_mask else None
    local_gripper = (
        gripper_mask[top:bottom, left:right] if has_gripper_mask else None
    )
    raster = _overlay_robot_imagination(
        raster,
        robot_mask=local_robot,
        gripper_mask=local_gripper,
    )
    image = Image.fromarray(raster).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    target = _project_pose(pose, camera)
    if glyph is not None:
        local = glyph - np.array([left, top], dtype=np.float64)
        stem, base, jaw_a, jaw_b, tip_a, tip_b, contact = local
        draw.line((*stem, *contact), fill=(*VIOLET, 150), width=2)
        for start, end in (
            (stem, base),
            (jaw_a, jaw_b),
            (jaw_a, tip_a),
            (jaw_b, tip_b),
        ):
            draw.line((*start, *end), fill=(*VIOLET, 255), width=4)
        x, y = contact
        draw.ellipse(
            (x - 4, y - 4, x + 4, y + 4),
            fill=(255, 255, 255, 220),
            outline=(*VIOLET, 255),
            width=2,
        )
    if target is not None:
        contact = (target[0] - left, target[1] - top)
        away = (target[2] - left, target[3] - top)
        _arrow(draw, away, contact, GREEN, width=3)
    return np.asarray(image, dtype=np.uint8)


def _candidate_gripper_glyph(pose: Pose, camera: dict[str, Any]) -> np.ndarray | None:
    rotation = Rotation.from_quat(np.asarray(pose.quaternion_xyzw)).as_matrix()
    approach = rotation[:, 2]
    across = rotation[:, 1]
    contact = np.asarray(pose.position_xyz, dtype=np.float64)
    base = contact - 0.055 * approach
    stem = contact - 0.095 * approach
    jaw_a = base + 0.028 * across
    jaw_b = base - 0.028 * across
    tip_a = jaw_a + 0.05 * approach
    tip_b = jaw_b + 0.05 * approach
    points = np.stack([stem, base, jaw_a, jaw_b, tip_a, tip_b, contact])
    try:
        projected = project_world_to_pixel(points, camera["intrinsics"], camera["pose_mat"])
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return None
    return projected[:, :2]


def _fit_box_aspect(
    bbox: tuple[float, float, float, float], *, target_aspect: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    return (
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    )


def _overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> np.ndarray:
    if mask is None or mask.shape != rgb.shape[:2] or not mask.any():
        return rgb
    out = rgb.astype(np.float64)
    out[mask] = out[mask] * (1.0 - alpha) + np.asarray(color) * alpha
    raster = np.clip(out, 0, 255).astype(np.uint8)
    raster[mask_outline(mask)] = color
    return raster


def _overlay_robot_imagination(
    rgb: np.ndarray,
    *,
    robot_mask: np.ndarray | None,
    gripper_mask: np.ndarray | None,
) -> np.ndarray:
    """Make the target hand dominant while keeping arm geometry translucent."""

    raster = _overlay_mask(rgb, robot_mask, VIOLET, alpha=0.45)
    return _overlay_mask(raster, gripper_mask, VIOLET, alpha=0.80)


def _mask_box(mask: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _union_boxes(
    boxes: list[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    valid = [box for box in boxes if box is not None]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def _pixel_box(
    pixel: tuple[float, float] | None,
    *,
    radius: float = 8.0,
) -> tuple[float, float, float, float] | None:
    if pixel is None:
        return None
    return (
        pixel[0] - radius,
        pixel[1] - radius,
        pixel[0] + radius,
        pixel[1] + radius,
    )


def _proposal_raster(
    rgb: np.ndarray,
    state: ContextState,
    camera: dict[str, Any],
    action,
) -> np.ndarray:
    robot = state.robot
    joints = action.prediction.joint_positions_rad
    opening = robot.gripper_opening if robot is not None else None
    robot_mask = None
    gripper_mask = None
    if joints is not None and opening is not None:
        robot_mask = _robot_mask(joints, opening, camera, rgb.shape[1], rgb.shape[0])
        gripper_mask = _gripper_mask(
            joints,
            opening,
            camera,
            rgb.shape[1],
            rgb.shape[0],
        )

    raster = rgb.copy()
    adjustment = action.adjustment
    if (
        adjustment is not None
        and adjustment.parent_action_id is None
        and robot is not None
        and robot.joint_positions_rad is not None
        and opening is not None
    ):
        observed_gripper = _gripper_mask(
            robot.joint_positions_rad,
            opening,
            camera,
            rgb.shape[1],
            rgb.shape[0],
        )
        raster = _overlay_mask(raster, observed_gripper, BLUE, alpha=0.58)
    raster = _overlay_robot_imagination(
        raster,
        robot_mask=robot_mask,
        gripper_mask=gripper_mask,
    )

    image = Image.fromarray(raster).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    reference_pixel = None
    target_pixel = _pose_origin_pixel(action.target_pose, camera)
    if adjustment is not None:
        reference = adjustment.reference_pose
        reference_pixel = _pose_origin_pixel(reference, camera)
        axes_rotation = (
            Rotation.identity()
            if adjustment.frame == "base"
            else Rotation.from_quat(np.asarray(reference.quaternion_xyzw, dtype=np.float64))
        )
        if adjustment.kind == "rotate":
            _draw_pose_axes(
                draw,
                reference.position_xyz,
                axes_rotation,
                camera,
                BLUE,
            )
            _draw_pose_axes(
                draw,
                action.target_pose.position_xyz,
                Rotation.from_quat(
                    np.asarray(action.target_pose.quaternion_xyzw, dtype=np.float64)
                ),
                camera,
                GREEN,
            )
        if (
            adjustment.kind == "delta_move"
            and reference_pixel is not None
            and target_pixel is not None
        ):
            _arrow(draw, reference_pixel, target_pixel, GREEN, width=4)
    else:
        source_bbox = _source_bbox(state, action.source_ref, camera)
        if source_bbox is not None:
            draw.rectangle(source_bbox, outline=(*BLUE, 230), width=2)

    annotated = np.asarray(image, dtype=np.uint8)
    source_bbox = _source_bbox(state, action.source_ref, camera)
    focus_box = _union_boxes(
        [
            source_bbox,
            _mask_box(gripper_mask),
            _pixel_box(reference_pixel, radius=14.0),
            _pixel_box(target_pixel, radius=14.0),
        ]
    )
    if focus_box is None:
        focus_box = _mask_box(robot_mask)
    if focus_box is None:
        focus_box = (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0]))
    focus_box = _fit_box_aspect(focus_box, target_aspect=1.8)
    left, top, right, bottom = _expanded_bounds(
        focus_box,
        rgb.shape[1],
        rgb.shape[0],
        ratio=0.28,
    )
    focus = Image.fromarray(annotated[top:bottom, left:right]).convert("RGB")
    focus = focus.resize((960, 480), Image.Resampling.BILINEAR)

    # A small whole-arm inset preserves global configuration evidence without
    # forcing the main interaction crop to zoom out.
    overview = Image.fromarray(annotated).convert("RGB")
    overview.thumbnail((260, 190), Image.Resampling.BILINEAR)
    inset = Image.new("RGB", (overview.width + 8, overview.height + 8), "white")
    inset.paste(overview, (4, 4))
    focus.paste(inset, (960 - inset.width - 10, 10))
    focus_draw = ImageDraw.Draw(focus, "RGBA")
    focus_draw.rectangle(
        (
            960 - inset.width - 10,
            10,
            960 - 10,
            10 + inset.height,
        ),
        outline=(*VIOLET, 255),
        width=2,
    )
    return np.asarray(focus, dtype=np.uint8)


def _pose_origin_pixel(pose: Pose, camera: dict[str, Any]) -> tuple[float, float] | None:
    projected = _project_pose(pose, camera)
    return None if projected is None else projected[:2]


def _draw_pose_axes(
    draw: ImageDraw.ImageDraw,
    origin_xyz: tuple[float, float, float],
    rotation: Rotation,
    camera: dict[str, Any],
    color: tuple[int, int, int],
    *,
    length_m: float = 0.045,
) -> None:
    origin = np.asarray(origin_xyz, dtype=np.float64)
    points = np.vstack([origin, origin + rotation.as_matrix().T * length_m])
    try:
        projected = project_world_to_pixel(points, camera["intrinsics"], camera["pose_mat"])
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return
    center = tuple(float(value) for value in projected[0, :2])
    for index, axis in enumerate("xyz", start=1):
        endpoint = tuple(float(value) for value in projected[index, :2])
        draw.line((*center, *endpoint), fill=(*color, 210), width=3)
        draw.text(
            (endpoint[0] + 2, endpoint[1] - 6),
            axis,
            fill=(*color, 255),
            font=ImageFont.load_default(),
        )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0:
        return
    unit = vector / length
    normal = np.array([-unit[1], unit[0]], dtype=np.float64)
    tip = np.asarray(end, dtype=np.float64)
    wing_a = tip - unit * 12.0 + normal * 6.0
    wing_b = tip - unit * 12.0 - normal * 6.0
    draw.line((*start, *end), fill=(*color, 230), width=width)
    draw.polygon(
        [tuple(tip), tuple(wing_a), tuple(wing_b)],
        fill=(*color, 230),
    )


def _post_action_raster(
    rgb: np.ndarray,
    camera: dict[str, Any],
    target_pose: Pose | None,
) -> np.ndarray:
    """Render a current-RGB target band for post-action world verification."""

    if target_pose is None:
        return rgb.copy()
    projected = _project_pose(target_pose, camera)
    if projected is None:
        return rgb.copy()
    target_x, target_y = projected[:2]
    height, width = rgb.shape[:2]
    if not (0.0 <= target_x < width and 0.0 <= target_y < height):
        return rgb.copy()

    crop_height = min(height, max(96, int(round(width / 3.6))))
    top = int(round(target_y - crop_height / 2))
    top = min(max(0, top), height - crop_height)
    return rgb[top : top + crop_height].copy()


def _source_bbox(
    state: ContextState,
    source_ref: str | None,
    camera: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    if source_ref is None:
        return None
    region = state.regions.get(source_ref)
    if region is not None:
        return region.bbox_xyxy_px
    point = state.points.get(source_ref)
    if point is not None:
        return _pixel_box(point.pixel_xy, radius=18.0)
    candidate = state.candidates.get(source_ref)
    if candidate is None:
        return None
    if candidate.source_ref is not None:
        source_region = state.regions.get(candidate.source_ref)
        if source_region is not None:
            return source_region.bbox_xyxy_px
    projected = _pose_origin_pixel(candidate.target_pose, camera)
    return _pixel_box(projected, radius=18.0)


def _robot_mask(
    joints: tuple[float, ...],
    opening: float,
    camera: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray | None:
    return _panda_mask(joints, opening, camera, width, height, whole_robot=True)


def _gripper_mask(
    joints: tuple[float, ...],
    opening: float,
    camera: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray | None:
    return _panda_mask(joints, opening, camera, width, height, whole_robot=False)


def _panda_mask(
    joints: tuple[float, ...],
    opening: float,
    camera: dict[str, Any],
    width: int,
    height: int,
    *,
    whole_robot: bool,
) -> np.ndarray | None:
    fk = load_panda_urdf_fk()
    if fk is None:
        return None
    try:
        triangle_fn = fk.robot_triangles if whole_robot else fk.triangles
        triangles = triangle_fn(np.asarray(joints), float(opening))
        projected = project_world_to_pixel(
            triangles.reshape(-1, 3),
            camera["intrinsics"],
            camera["pose_mat"],
        ).reshape(-1, 3, 3)
        return rasterize_silhouette(projected[:, :, :2], projected[:, :, 2], width, height)
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None


def _project_pose(pose: Pose, camera: dict[str, Any]) -> tuple[float, float, float, float] | None:
    rotation = Rotation.from_quat(np.asarray(pose.quaternion_xyzw)).as_matrix()
    origin = np.asarray(pose.position_xyz, dtype=np.float64)
    axis = origin + rotation[:, 2] * 0.06
    try:
        projected = project_world_to_pixel(
            np.stack([origin, axis]), camera["intrinsics"], camera["pose_mat"]
        )
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return None
    return tuple(float(value) for value in projected[:, :2].reshape(-1))


def _expanded_bounds(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    pad_x = max((x2 - x1) * ratio, 4.0)
    pad_y = max((y2 - y1) * ratio, 4.0)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return 0, 0, width, height
    return left, top, right, bottom


def _cross(
    draw: ImageDraw.ImageDraw,
    pixel: tuple[float, float],
    color: tuple[int, int, int],
    *,
    radius: int,
    width: int,
) -> None:
    x, y = (float(pixel[0]), float(pixel[1]))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(*color, 255),
        width=width,
    )
    draw.line(
        (x - radius - 4, y, x + radius + 4, y),
        fill=(*color, 255),
        width=width,
    )
    draw.line(
        (x, y - radius - 4, x, y + radius + 4),
        fill=(*color, 255),
        width=width,
    )


def _label(
    draw: ImageDraw.ImageDraw,
    pixel: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = ImageFont.load_default()
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    box = draw.textbbox((x, y), text, font=font, stroke_width=0)
    draw.rounded_rectangle(
        (box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3),
        radius=3,
        fill=(*color, 220),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _rgb(camera: dict[str, Any]) -> np.ndarray:
    try:
        rgb = np.asarray(camera["images"]["rgb"])
    except (KeyError, TypeError) as exc:
        raise ValueError("camera has no RGB image") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must be HxWx3, got {rgb.shape}")
    return np.asarray(rgb, dtype=np.uint8)


def _rounded(values: tuple[float, ...], digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def encode_png_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


__all__ = [
    "CONTEXT_HEIGHT",
    "CONTEXT_SCHEMA",
    "CONTEXT_WEB_SCHEMA_VERSION",
    "CONTEXT_WIDTH",
    "CandidateSpec",
    "ContextCompiler",
    "ContextPacket",
    "DecisionMode",
    "DecisionWorkspaceSpec",
    "EvidenceCatalogSpec",
    "PointSpec",
    "RegionSpec",
    "WorldContextSpec",
    "encode_png_data_url",
]
