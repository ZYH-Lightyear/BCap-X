"""Deterministic compiler for the fixed-size persistent-Waypoint Canvas."""

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
from vaw.context_runtime.model import (
    ActionTarget,
    ContextState,
    LastPhysicalAction,
    PointEvidence,
    Pose,
    RobotState,
)
from vaw.context_runtime.near_field import NearFieldPreview
from vaw.context_runtime.private import (
    ActionReviewArtifacts,
    ImaginationArtifacts,
    PrivateEnvContext,
)
from vaw.context_runtime.scene_view import render_scene_view
from vaw.context_runtime.workspace import ContextWorkspace

CONTEXT_SCHEMA = "vaw-context-v13-post-commit"
CONTEXT_WEB_SCHEMA_VERSION = 14
CONTEXT_WIDTH = 1920
CONTEXT_HEIGHT = 1080

BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
VIOLET = (124, 58, 237)

_CANDIDATE_FK_POSITION_TOLERANCE_M = 0.02
_CANDIDATE_FK_ROTATION_TOLERANCE_RAD = 0.10

DecisionMode = Literal[
    "idle",
    "grounding",
    "seeds",
    "editing",
    "reviewed",
    "post_commit",
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
class SeedSpec:
    seed_id: str
    target_pose: Pose
    delta_from_anchor_xyz_m: tuple[float, float, float] | None
    approach_vector_base: tuple[float, float, float] | None
    solve_ik: str
    source_revision: int
    raster_id: str

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.seed_id,
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
        return out


@dataclass(frozen=True)
class WorldContextSpec:
    agentview_raster_id: str
    observed_scene_raster_id: str
    imagination_scene_raster_id: str
    robot: RobotState | None
    owner: str
    action: dict[str, Any] | None
    refinement_goal: str | None
    latest_error: str | None
    last_physical_action: LastPhysicalAction | None = None
    post_commit_before_raster_id: str | None = None
    post_commit_current_raster_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "agentviewRasterId": self.agentview_raster_id,
            "observedSceneRasterId": self.observed_scene_raster_id,
            "imaginationSceneRasterId": self.imagination_scene_raster_id,
            "robot": self.robot.summary() if self.robot is not None else None,
            "owner": self.owner,
            "action": self.action,
            "refinementGoal": self.refinement_goal,
            "latestError": self.latest_error,
            "lastPhysicalAction": (
                self.last_physical_action.summary()
                if self.last_physical_action is not None
                else None
            ),
            "postCommitBeforeRasterId": self.post_commit_before_raster_id,
            "postCommitCurrentRasterId": self.post_commit_current_raster_id,
        }


@dataclass(frozen=True)
class EvidenceCatalogSpec:
    regions: tuple[RegionSpec, ...]
    points: tuple[PointSpec, ...]
    seeds: tuple[SeedSpec, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "regions": [item.summary() for item in self.regions],
            "points": [item.summary() for item in self.points],
            "seeds": [item.summary() for item in self.seeds],
        }


@dataclass(frozen=True)
class DecisionWorkspaceSpec:
    mode: DecisionMode
    region_ids: tuple[str, ...] = ()
    point_ids: tuple[str, ...] = ()
    seed_ids: tuple[str, ...] = ()
    action_id: str | None = None
    primary_raster_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "regionIds": list(self.region_ids),
            "pointIds": list(self.point_ids),
            "seedIds": list(self.seed_ids),
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
        return {
            "owner": self.world.owner,
            "review_action_id": (
                self.world.action.get("action_id")
                if self.world.action is not None
                else None
            ),
            "valid_region_ids": [item.region_id for item in self.catalog.regions],
            "valid_point_ids": [item.point_id for item in self.catalog.points],
            "valid_seed_ids": [item.seed_id for item in self.catalog.seeds],
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
        wrist_camera = None
        with suppress(RuntimeError, ValueError):
            wrist_camera = private.camera(workspace.wrist_camera_name)
        target, active_artifacts, action_presentation = _active_presentation(workspace)
        scene_preview = (
            _near_field_preview(state, target, active_artifacts)
            if target is not None
            else None
        )
        source_ref = _observed_source_ref(active_artifacts)
        source_mask = (
            private.region_masks.get(source_ref)
            if source_ref is not None
            else None
        )
        observed_scene = render_scene_view(
            camera,
            wrist_camera,
            state.robot,
            dark=False,
        )
        imagination_scene = render_scene_view(
            camera,
            wrist_camera,
            state.robot,
            scene_preview,
            dark=True,
            source_mask=source_mask,
        )

        # The persistent world view is deliberately sensor-clean.  All
        # grounding, self and imagination overlays belong to the dynamic
        # decision workspace below it.
        rasters: dict[str, np.ndarray] = {
            "agentview": rgb.copy(),
            "observed_scene": observed_scene,
            "imagination_scene": imagination_scene,
        }
        post_before_id, post_current_id = _compile_post_commit_rasters(
            workspace,
            camera,
            rasters,
        )

        region_specs = self._compile_regions(state, rgb, private.region_masks, rasters)
        point_specs = self._compile_points(state, rgb, rasters)
        seed_specs = self._compile_seeds(
            state,
            private,
            rgb,
            camera,
            rasters,
            tcp_to_hand_local_xyz=workspace._tcp_to_hand_local_xyz,
        )
        decision = _decision_spec(workspace)
        event = private.presentation_event
        packet = ContextPacket(
            revision=state.observation_revision,
            world=WorldContextSpec(
                agentview_raster_id="agentview",
                observed_scene_raster_id="observed_scene",
                imagination_scene_raster_id="imagination_scene",
                robot=state.robot,
                owner=state.owner,
                action=action_presentation,
                refinement_goal=(
                    state.imagination.refinement_goal
                    if state.imagination is not None
                    else None
                ),
                latest_error=event.error if event is not None else None,
                last_physical_action=state.last_physical_action,
                post_commit_before_raster_id=post_before_id,
                post_commit_current_raster_id=post_current_id,
            ),
            catalog=EvidenceCatalogSpec(
                regions=tuple(region_specs),
                points=tuple(point_specs),
                seeds=tuple(seed_specs),
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
    def _compile_seeds(
        state: ContextState,
        private: PrivateEnvContext,
        rgb: np.ndarray,
        camera: dict[str, Any],
        rasters: dict[str, np.ndarray],
        *,
        tcp_to_hand_local_xyz: np.ndarray,
    ) -> list[SeedSpec]:
        specs: list[SeedSpec] = []
        for seed in state.seeds.values():
            raster_id = f"seed:{seed.seed_id}"
            artifact = private.seed_artifacts.get(seed.seed_id)
            region_id = (
                artifact.planning_context.region_id
                if artifact is not None and artifact.planning_context is not None
                else None
            )
            source = state.regions.get(region_id) if region_id is not None else None
            pose = seed.target.pose
            if pose is None:
                continue
            anchor = next(
                (
                    point
                    for point in state.points.values()
                    if point.within_region_id == region_id
                ),
                None,
            )
            rotation = Rotation.from_quat(
                np.asarray(pose.quaternion_xyzw)
            ).as_matrix()
            approach = tuple(float(value) for value in rotation[:, 2])
            delta = None
            if anchor is not None:
                delta = tuple(
                    float(value)
                    for value in (
                        np.asarray(pose.position_xyz)
                        - np.asarray(anchor.position_xyz)
                    )
                )
            robot_mask = None
            gripper_mask = None
            prediction = (
                artifact.preview_plan.prediction
                if artifact is not None and artifact.preview_plan is not None
                else None
            )
            displayed_solve_ik = prediction.solve_ik if prediction is not None else "unavailable"
            if prediction is not None and prediction.joint_positions_rad is not None:
                preview_matches = _candidate_fk_matches_target(
                    prediction.joint_positions_rad,
                    pose,
                    tcp_to_hand_local_xyz,
                )
                if preview_matches:
                    opening = (
                        state.robot.gripper_opening
                        if state.robot is not None
                        and state.robot.gripper_opening is not None
                        else 1.0
                    )
                    robot_mask = _robot_mask(
                        prediction.joint_positions_rad,
                        opening,
                        camera,
                        rgb.shape[1],
                        rgb.shape[0],
                    )
                    gripper_mask = _gripper_mask(
                        prediction.joint_positions_rad,
                        opening,
                        camera,
                        rgb.shape[1],
                        rgb.shape[0],
                    )
                else:
                    displayed_solve_ik = "mismatch"
            rasters[raster_id] = _candidate_crop(
                rgb,
                source.bbox_xyxy_px if source is not None else None,
                pose,
                camera,
                robot_mask=robot_mask,
                gripper_mask=gripper_mask,
            )
            specs.append(
                SeedSpec(
                    seed_id=seed.seed_id,
                    target_pose=pose,
                    delta_from_anchor_xyz_m=delta,
                    approach_vector_base=approach,
                    solve_ik=displayed_solve_ik,
                    source_revision=seed.source_revision,
                    raster_id=raster_id,
                )
            )
        return specs


def _decision_spec(workspace: ContextWorkspace) -> DecisionWorkspaceSpec:
    state = workspace.state
    event = workspace._private.presentation_event
    if state.imagination is not None:
        return DecisionWorkspaceSpec(
            mode="editing",
            seed_ids=tuple(state.seeds)[:5],
        )

    if state.last_physical_action is not None:
        return DecisionWorkspaceSpec(mode="post_commit")

    # A Main-review target may remain available while Main gathers newer evidence.
    # The lower canvas must show that latest evidence instead of pinning an old
    # reviewed preview over every subsequent detection/proposal result.
    if event is not None:
        if event.error is not None:
            return DecisionWorkspaceSpec(mode="error")
        if event.function_name == "done":
            return DecisionWorkspaceSpec(mode="terminal")
        result = event.result
        seed_ids = result.get("seed_ids")
        if isinstance(seed_ids, list):
            valid = tuple(
                str(value) for value in seed_ids if str(value) in state.seeds
            )[:5]
            return DecisionWorkspaceSpec(mode="seeds", seed_ids=valid)
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

    if state.action_review is not None:
        return DecisionWorkspaceSpec(
            mode="reviewed",
            seed_ids=tuple(state.seeds)[:5],
            action_id=state.action_review.action_id,
        )
    if (
        state.last_handoff is not None
        and state.last_handoff.status == "failed"
        and state.seeds
    ):
        return DecisionWorkspaceSpec(
            mode="seeds",
            seed_ids=tuple(state.seeds)[:5],
        )

    return DecisionWorkspaceSpec(mode="idle")


def _compile_post_commit_rasters(
    workspace: ContextWorkspace,
    current_camera: dict[str, Any],
    rasters: dict[str, np.ndarray],
) -> tuple[str | None, str | None]:
    """Compile one policy-visible before/current comparison around the target.

    Sensor calibration remains private: it is used only to project the last
    physical target into each observation.  The packet receives two RGB crops
    at the same metric scale, never the old raw frame as a second model image.
    """

    if workspace.state.last_physical_action is None:
        return None, None
    artifacts = workspace._private.last_physical_artifacts
    try:
        previous_camera = workspace._private.camera(
            workspace.camera_name,
            previous=True,
        )
    except RuntimeError:
        previous_camera = current_camera
    focus_pose = artifacts.focus_pose if artifacts is not None else None
    before_id = "post_commit:before"
    current_id = "post_commit:current"
    rasters[before_id] = _metric_focus_crop(previous_camera, focus_pose)
    rasters[current_id] = _metric_focus_crop(current_camera, focus_pose)
    return before_id, current_id


def _metric_focus_crop(
    camera: dict[str, Any],
    focus_pose: Pose | None,
    *,
    span_m: float = 0.32,
    output_size: tuple[int, int] = (760, 390),
) -> np.ndarray:
    """Return a deterministic target-centred RGB crop or a full-view fallback."""

    rgb = _rgb(camera)
    bounds: tuple[int, int, int, int] | None = None
    if focus_pose is not None:
        try:
            projected = project_world_to_pixel(
                np.asarray([focus_pose.position_xyz], dtype=np.float64),
                camera["intrinsics"],
                camera["pose_mat"],
            )[0]
            depth = float(projected[2])
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
            if (
                np.isfinite(projected).all()
                and depth > 1e-4
                and 0.0 <= float(projected[0]) < rgb.shape[1]
                and 0.0 <= float(projected[1]) < rgb.shape[0]
            ):
                half_w = float(intrinsics[0, 0]) * span_m / (2.0 * depth)
                half_h = float(intrinsics[1, 1]) * span_m / (2.0 * depth)
                crop_box = _fit_box_aspect(
                    (
                        float(projected[0] - half_w),
                        float(projected[1] - half_h),
                        float(projected[0] + half_w),
                        float(projected[1] + half_h),
                    ),
                    target_aspect=output_size[0] / output_size[1],
                )
                bounds = _expanded_bounds(
                    crop_box,
                    rgb.shape[1],
                    rgb.shape[0],
                    ratio=0.0,
                )
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            bounds = None
    if bounds is None:
        crop = rgb
    else:
        left, top, right, bottom = bounds
        crop = rgb[top:bottom, left:right]
        if crop.size == 0:
            crop = rgb
    image = Image.fromarray(crop).convert("RGB")
    image = image.resize(output_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _active_presentation(
    workspace: ContextWorkspace,
) -> tuple[
    ActionTarget | None,
    ImaginationArtifacts | ActionReviewArtifacts | None,
    dict[str, Any] | None,
]:
    state = workspace.state
    if state.imagination is not None:
        artifacts = workspace._private.imagination_artifacts
        return (
            state.imagination.target,
            artifacts,
            _target_presentation(
                state.imagination.target,
                artifacts,
                status="editing",
                action_id=None,
            ),
        )
    if state.action_review is not None:
        artifacts = workspace._private.review_artifacts.get(
            state.action_review.action_id
        )
        return (
            state.action_review.target,
            artifacts,
            _target_presentation(
                state.action_review.target,
                artifacts,
                status="review",
                action_id=state.action_review.action_id,
                intent=state.action_review.intent,
            ),
        )
    return None, None, None


def _target_presentation(
    target: ActionTarget,
    artifacts: ImaginationArtifacts | ActionReviewArtifacts | None,
    *,
    status: str,
    action_id: str | None,
    intent: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "target": target.summary()}
    if action_id is not None:
        result["action_id"] = action_id
    if intent is not None:
        result["intent"] = intent
    plan = _artifact_plan(artifacts)
    if plan is not None:
        result["prediction"] = plan.prediction.summary()
    if isinstance(artifacts, ImaginationArtifacts) and artifacts.latest_visual_edit:
        result["latest_edit"] = artifacts.latest_visual_edit.summary()
    return result


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
    # Candidate cards have a stable portrait-ish viewport.  Keeping the
    # raster close to that aspect prevents one remaining seed from becoming
    # a full-width, heavily cropped strip in the Web renderer.
    crop_box = _fit_box_aspect(crop_box, target_aspect=0.82)
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
    # The hand/object relationship is the decision evidence.  Preserve the
    # true whole-arm silhouette, but keep it subordinate to the target hand.
    raster = _overlay_mask(raster, local_robot, VIOLET, alpha=0.10)
    raster = _overlay_mask(raster, local_gripper, VIOLET, alpha=0.80)
    image = Image.fromarray(raster).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    if bbox is not None:
        draw.rectangle(
            (
                bbox[0] - left,
                bbox[1] - top,
                bbox[2] - left,
                bbox[3] - top,
            ),
            outline=(*BLUE, 235),
            width=3,
        )
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

    raster = _overlay_mask(rgb, robot_mask, VIOLET, alpha=0.26)
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
    target: ActionTarget,
    artifacts: ImaginationArtifacts | ActionReviewArtifacts | None,
) -> np.ndarray:
    """Render current RGB plus the reviewed target without predicting dynamics."""

    robot = state.robot
    plan = _artifact_plan(artifacts)
    joints = plan.prediction.joint_positions_rad if plan is not None else None
    if target.pose is None and robot is not None:
        joints = robot.joint_positions_rad
    current_opening = robot.gripper_opening if robot is not None else None
    target_opening = (
        1.0
        if target.gripper == "open"
        else 0.0
        if target.gripper == "closed"
        else current_opening
    )
    robot_mask = None
    gripper_mask = None
    if joints is not None and target_opening is not None:
        robot_mask = _robot_mask(
            joints,
            target_opening,
            camera,
            rgb.shape[1],
            rgb.shape[0],
        )
        gripper_mask = _gripper_mask(
            joints,
            target_opening,
            camera,
            rgb.shape[1],
            rgb.shape[0],
        )

    raster = rgb.copy()
    visual_edit = (
        artifacts.latest_visual_edit
        if isinstance(artifacts, ImaginationArtifacts)
        else None
    )
    if (
        visual_edit is not None
        and robot is not None
        and robot.joint_positions_rad is not None
        and current_opening is not None
    ):
        observed_gripper = _gripper_mask(
            robot.joint_positions_rad,
            current_opening,
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
    source_ref = _observed_source_ref(artifacts)
    source_bbox = _source_bbox(state, source_ref, camera)
    reference_pixel = None
    target_pixel = (
        _pose_origin_pixel(target.pose, camera)
        if target.pose is not None
        else None
    )
    if visual_edit is not None and target.pose is not None:
        reference = visual_edit.reference_pose
        reference_pixel = _pose_origin_pixel(reference, camera)
        axes_rotation = (
            Rotation.identity()
            if visual_edit.frame == "base"
            else Rotation.from_quat(
                np.asarray(reference.quaternion_xyzw, dtype=np.float64)
            )
        )
        if visual_edit.kind == "rotate":
            _draw_pose_axes(draw, reference.position_xyz, axes_rotation, camera, BLUE)
            _draw_pose_axes(
                draw,
                target.pose.position_xyz,
                Rotation.from_quat(
                    np.asarray(target.pose.quaternion_xyzw, dtype=np.float64)
                ),
                camera,
                GREEN,
            )
            _draw_rotation_arc(
                draw,
                visual_edit,
                camera,
            )
        elif reference_pixel is not None and target_pixel is not None:
            _arrow(draw, reference_pixel, target_pixel, GREEN, width=4)
    elif source_bbox is not None:
        draw.rectangle(source_bbox, outline=(*BLUE, 230), width=2)
        if source_ref is not None:
            _label(
                draw,
                (source_bbox[0] + 3, max(2.0, source_bbox[1] - 18.0)),
                source_ref,
                BLUE,
            )

    annotated = np.asarray(image, dtype=np.uint8)
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
    focus_width, focus_height = 1280, 640
    focus = focus.resize((focus_width, focus_height), Image.Resampling.BILINEAR)

    overview = Image.fromarray(annotated).convert("RGB")
    overview.thumbnail((346, 253), Image.Resampling.BILINEAR)
    inset = Image.new("RGB", (overview.width + 8, overview.height + 8), "white")
    inset.paste(overview, (4, 4))
    focus.paste(inset, (focus_width - inset.width - 12, 12))
    ImageDraw.Draw(focus, "RGBA").rectangle(
        (
            focus_width - inset.width - 12,
            12,
            focus_width - 12,
            12 + inset.height,
        ),
        outline=(*VIOLET, 255),
        width=3,
    )
    return np.asarray(focus, dtype=np.uint8)


def _observed_source_ref(
    artifacts: ImaginationArtifacts | ActionReviewArtifacts | None,
) -> str | None:
    if artifacts is None or artifacts.planning_context is None:
        return None
    context = artifacts.planning_context
    # A grasp seed's source_ref identifies the virtual seed itself; region_id
    # identifies the observed object that seed is supposed to manipulate.
    # Prefer observed evidence so a bad seed remains visibly inconsistent.
    return context.region_id or context.source_ref


def _artifact_plan(
    artifacts: ImaginationArtifacts | ActionReviewArtifacts | None,
):
    if isinstance(artifacts, ImaginationArtifacts):
        return artifacts.preview_plan
    if isinstance(artifacts, ActionReviewArtifacts):
        return artifacts.motion_plan
    return None


def _near_field_preview(
    state: ContextState,
    target: ActionTarget,
    artifacts: ImaginationArtifacts | ActionReviewArtifacts | None,
) -> NearFieldPreview | None:
    """Compile the active virtual target for the observed near-field cloud.

    Point samples always come from the current sensor revision.  Only the
    robot geometry is virtual, so this preview cannot imply object motion or
    contact success.
    """

    robot = state.robot
    if robot is None or robot.gripper_opening is None:
        return None
    plan = _artifact_plan(artifacts)
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
    return NearFieldPreview(
        target_pose=target.pose,
        joint_positions_rad=joints,
        gripper_opening=opening,
        visual_edit=visual_edit,
    )


def _agentview_with_base_axes(
    rgb: np.ndarray,
    camera: dict[str, Any],
    robot: RobotState | None,
) -> np.ndarray:
    """Draw the fixed LIBERO agentview control legend used by base-frame edits.

    This is deliberately a command-direction legend rather than a projected
    3-D gizmo.  LIBERO-PRO uses one fixed agentview: base +Z raises the TCP,
    base +Y moves screen-right, and base +X moves toward the image bottom.
    """

    del camera, robot
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    origin = np.array([90.0, float(rgb.shape[0] - 112)], dtype=np.float64)
    directions = (
        np.array([0.0, 1.0]),   # BASE +X: image down
        np.array([1.0, 0.0]),   # BASE +Y: image right
        np.array([0.0, -1.0]),  # BASE +Z: image up / physical lift
    )
    colors = ((220, 38, 38), (22, 163, 74), (37, 99, 235))
    draw.ellipse(
        (origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7),
        fill=(255, 255, 255, 245),
        outline=(71, 85, 105, 255),
        width=2,
    )
    for index, (color, axis_name) in enumerate(zip(colors, "XYZ", strict=True)):
        direction = directions[index]
        length = float(np.linalg.norm(direction))
        label = f"+{axis_name}"
        unit = direction / length
        endpoint = origin + unit * 64.0
        _arrow(draw, tuple(origin), tuple(endpoint), color, width=7)
        _axis_label_box(draw, tuple(endpoint + unit * 20.0), label, color)
    return np.asarray(image, dtype=np.uint8)


def _axis_label_box(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    font = _font(17)
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0] + 14
    height = box[3] - box[1] + 10
    left = float(anchor[0]) - width * 0.5
    top = float(anchor[1]) - height * 0.5
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=4,
        fill=(255, 255, 255, 232),
        outline=(*color, 255),
        width=3,
    )
    draw.text(
        (left + 7, top + 4 - box[1]),
        label,
        fill=(*color, 255),
        font=font,
    )


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


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


def _draw_rotation_arc(
    draw: ImageDraw.ImageDraw,
    edit: Any,
    camera: dict[str, Any],
) -> None:
    if edit.axis not in {"x", "y", "z"} or edit.angle_deg is None:
        return
    axis_index = "xyz".index(edit.axis)
    axis_unit = np.eye(3, dtype=np.float64)[axis_index]
    reference_rotation = Rotation.from_quat(
        np.asarray(edit.reference_pose.quaternion_xyzw, dtype=np.float64)
    ).as_matrix()
    axis_base = axis_unit if edit.frame == "base" else reference_rotation @ axis_unit
    radial_base = (
        np.eye(3, dtype=np.float64)[(axis_index + 1) % 3]
        if edit.frame == "base"
        else reference_rotation[:, (axis_index + 1) % 3]
    )
    center = np.asarray(edit.reference_pose.position_xyz, dtype=np.float64)
    samples = np.linspace(
        0.0,
        np.deg2rad(float(edit.angle_deg)),
        num=25,
        dtype=np.float64,
    )
    arc = np.vstack(
        [
            center
            + Rotation.from_rotvec(axis_base * angle).apply(radial_base * 0.05)
            for angle in samples
        ]
    )
    try:
        projected = project_world_to_pixel(
            arc,
            camera["intrinsics"],
            camera["pose_mat"],
        )
    except (KeyError, ValueError, np.linalg.LinAlgError):
        return
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return
    points = [tuple(float(value) for value in pixel[:2]) for pixel in projected]
    colors = ((220, 38, 38), GREEN, BLUE)
    color = colors[axis_index]
    draw.line(points, fill=(*color, 240), width=5, joint="curve")
    if len(points) >= 2:
        _arrow(draw, points[-2], points[-1], color, width=5)


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
    seed = state.seeds.get(source_ref)
    if seed is None or seed.target.pose is None:
        return None
    projected = _pose_origin_pixel(seed.target.pose, camera)
    return _pixel_box(projected, radius=18.0)


def _candidate_fk_matches_target(
    joints: tuple[float, ...],
    target: Pose,
    tcp_to_hand_local_xyz: np.ndarray,
) -> bool:
    """Whether preview joints actually realise the candidate TCP pose.

    Reduced ``solve_ik`` may clamp the requested position or substitute a
    fallback orientation. Its returned joints remain valid robot joints, but
    rendering them as the original candidate would be misleading. Validate
    the exact URDF FK before allowing a whole-arm mask onto a candidate card.
    """

    fk = load_panda_urdf_fk()
    if fk is None:
        return False
    try:
        actual_hand = fk.frame(np.asarray(joints), "panda_hand")
        target_rotation = Rotation.from_quat(
            np.asarray(target.quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
        expected_hand_position = np.asarray(
            target.position_xyz, dtype=np.float64
        ) + target_rotation @ np.asarray(
            tcp_to_hand_local_xyz, dtype=np.float64
        ).reshape(3)
        position_error = float(
            np.linalg.norm(actual_hand[:3, 3] - expected_hand_position)
        )
        rotation_error = float(
            (
                Rotation.from_matrix(actual_hand[:3, :3]).inv()
                * Rotation.from_matrix(target_rotation)
            ).magnitude()
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return False
    return (
        position_error <= _CANDIDATE_FK_POSITION_TOLERANCE_M
        and rotation_error <= _CANDIDATE_FK_ROTATION_TOLERANCE_RAD
    )


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
    # The public approach vector is local +Z from panda_hand toward the TCP.
    # Draw the arrow from the hand side into the contact point so the raster
    # direction matches SeedSpec.approachVector exactly.
    hand_side = origin - rotation[:, 2] * 0.06
    try:
        projected = project_world_to_pixel(
            np.stack([origin, hand_side]),
            camera["intrinsics"],
            camera["pose_mat"],
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
    "SeedSpec",
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
