"""Deterministic compiler for Main and focused Imagination canvases."""

from __future__ import annotations

import base64
import io
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from vaw.context_runtime.attached_object import volume_triangles_base
from vaw.context_runtime.contact_camera import (
    ContactCameraRequest,
    OppositeSceneCameraRequest,
)
from vaw.context_runtime.geometry import project_world_to_pixel
from vaw.context_runtime.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    overlay_projected_mesh_outline,
    rasterize_silhouette,
)
from vaw.context_runtime.model import (
    ContextState,
    PointEvidence,
    Pose,
    RobotState,
)
from vaw.context_runtime.near_field import (
    CONTACT_FOCUS_WIDTH,
    CONTACT_PANEL_HEIGHT,
    NearFieldPreview,
    PreviewGripperStyle,
    _direct_target_gripper_triangles,
    gravity_stable_contact_frame_quaternion,
    render_contact_focus,
)
from vaw.context_runtime.presentation import (
    compile_active_presentation as _active_presentation,
)
from vaw.context_runtime.presentation import (
    compile_near_field_preview as _near_field_preview,
)
from vaw.context_runtime.presentation import (
    observed_source_ref as _observed_source_ref,
)
from vaw.context_runtime.private import PrivateEnvContext
from vaw.context_runtime.scene_view import (
    OBSERVED_SCENE_HEIGHT,
    OBSERVED_SCENE_WIDTH,
    render_scene_view,
)
from vaw.context_runtime.workspace import ContextWorkspace

CONTEXT_SCHEMA = "vaw-context-v37-working-memory"
CONTEXT_WEB_SCHEMA_VERSION = 38
CONTEXT_WIDTH = 2048
CONTEXT_HEIGHT = 1280

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
    "proposal",
    "contact",
    "terminal",
]
ContextProjection = Literal["main", "imagination"]


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
    family: str | None = None

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
        if self.family:
            out["family"] = self.family
        return out


@dataclass(frozen=True)
class WorldContextSpec:
    agentview_raster_id: str
    observed_scene_raster_id: str
    imagination_scene_raster_id: str
    contact_focus_raster_id: str | None
    robot: RobotState | None
    action: dict[str, Any] | None
    refinement_goal: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "agentviewRasterId": self.agentview_raster_id,
            "observedSceneRasterId": self.observed_scene_raster_id,
            "imaginationSceneRasterId": self.imagination_scene_raster_id,
            "contactFocusRasterId": self.contact_focus_raster_id,
            "robot": self.robot.summary() if self.robot is not None else None,
            "action": self.action,
            "refinementGoal": self.refinement_goal,
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
    projection: ContextProjection = "main"
    schema: str = CONTEXT_SCHEMA

    def summary(self) -> dict[str, Any]:
        """JSON-safe packet metadata; RGB arrays are represented by ids."""

        return {
            "schema": self.schema,
            "projection": self.projection,
            "revision": self.revision,
            "viewport": {"width": CONTEXT_WIDTH, "height": CONTEXT_HEIGHT},
            "world": self.world.summary(),
            "catalog": self.catalog.summary(),
            "decision": self.decision.summary(),
            "rasterIds": list(self.rasters),
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "pending_action": (
                {
                    "action_id": self.world.action.get("action_id"),
                    "intent": self.world.action.get("intent"),
                    "state": self.world.action.get("status"),
                }
                if self.world.action is not None
                else None
            ),
            "regions": [
                {"id": item.region_id, "query": item.query}
                for item in self.catalog.regions
            ],
            "points": [
                {
                    "id": item.point_id,
                    "query": item.query,
                    **(
                        {"within_region_id": item.within_region_id}
                        if item.within_region_id is not None
                        else {}
                    ),
                }
                for item in self.catalog.points
            ],
            "seed_ids": [item.seed_id for item in self.catalog.seeds],
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

    def __init__(self, *, preview_gripper_style: PreviewGripperStyle = "fk-mesh") -> None:
        if preview_gripper_style not in {"fk-mesh", "semantic-wireframe"}:
            raise ValueError(f"unsupported preview gripper style: {preview_gripper_style}")
        self.preview_gripper_style = preview_gripper_style

    def compile(
        self,
        workspace: ContextWorkspace,
        *,
        projection: ContextProjection = "main",
    ) -> ContextPacket:
        state = workspace.state
        private = workspace._private
        camera = private.camera(workspace.camera_name)
        rgb = _rgb(camera)
        wrist_camera = None
        with suppress(RuntimeError, ValueError):
            wrist_camera = private.camera(workspace.wrist_camera_name)
        target, active_artifacts, action_presentation = _active_presentation(workspace)
        scene_preview = (
            _near_field_preview(
                state,
                target,
                active_artifacts,
                gripper_style=self.preview_gripper_style,
                tcp_to_hand_local_xyz=workspace._tcp_to_hand_local_xyz,
            )
            if target is not None
            else None
        )
        contact_preview = scene_preview or _current_physical_contact_preview(workspace)
        carried_volume_triangles = None
        if target is not None and private.attachment_hypothesis is not None:
            carried_volume_triangles = volume_triangles_base(
                private.attachment_hypothesis,
                target.pose,
            )
        framing_carried_triangles = carried_volume_triangles
        if (
            framing_carried_triangles is None
            and private.attachment_hypothesis is not None
            and state.robot is not None
            and state.robot.tcp_pose is not None
        ):
            # Physical Contact views use the attachment only as a camera
            # framing prior at the current observed TCP.  The proxy is not
            # drawn as observed truth; the direct MuJoCo RGB remains the
            # policy-visible evidence.
            framing_carried_triangles = volume_triangles_base(
                private.attachment_hypothesis,
                state.robot.tcp_pose,
            )
        source_ref = _presentation_region_ref(state, active_artifacts)
        source_mask = private.region_masks.get(source_ref) if source_ref is not None else None
        contact_cameras = None
        if (
            contact_preview is not None
            and contact_preview.contact_frame_position_xyz is not None
            and contact_preview.contact_frame_quaternion_xyzw is not None
            and callable(private.contact_camera_provider)
        ):
            action_id = (
                state.pending_action.action_id if state.pending_action is not None else None
            )
            if (
                private.contact_camera_pair is None
                or private.contact_camera_action_id != action_id
            ):
                required_preview_points = _preview_contact_points(
                    state.robot,
                    contact_preview,
                )
                subject_points = None
                if source_ref is not None:
                    geometry = private.region_geometry.get(source_ref)
                    if geometry is not None:
                        subject_points = geometry.filtered_object_points_base
                if (
                    subject_points is None
                    and private.last_physical_artifacts is not None
                ):
                    subject_points = (
                        private.last_physical_artifacts.subject_points_base
                    )
                if framing_carried_triangles is not None:
                    carried_points = framing_carried_triangles.reshape(-1, 3)
                    last_subject = private.last_physical_artifacts
                    stale_carried_surface = (
                        target is None
                        and private.attachment_hypothesis is not None
                        and last_subject is not None
                        and last_subject.subject_query
                        == private.attachment_hypothesis.query
                    )
                    subject_points = (
                        carried_points
                        if subject_points is None or stale_carried_surface
                        else np.vstack((subject_points, carried_points))
                    )
                contact_center = _contact_camera_center(
                    contact_preview.contact_frame_position_xyz,
                    subject_points,
                    required_preview_points,
                )
                private.trace_diagnostics["contact_camera_framing"] = {
                    "center_base_xyz": [round(float(value), 6) for value in contact_center],
                    "subject_point_count": (
                        int(len(subject_points)) if subject_points is not None else 0
                    ),
                    "destination_region_id": source_ref,
                }
                private.contact_camera_pair = private.contact_camera_provider(
                    ContactCameraRequest(
                        center_base_xyz=contact_center,
                        frame_quaternion_xyzw=(
                            contact_preview.contact_frame_quaternion_xyzw
                        ),
                        width=CONTACT_FOCUS_WIDTH,
                        panel_height=CONTACT_PANEL_HEIGHT,
                        subject_points_base=subject_points,
                        required_points_base=required_preview_points,
                    )
                )
                private.contact_camera_action_id = action_id
            contact_cameras = private.contact_camera_pair
        observed_scene = _compile_observed_scene(
            workspace,
            camera,
            wrist_camera,
        )
        imagination_scene = render_scene_view(
            camera,
            wrist_camera,
            state.robot,
            contact_preview,
            dark=True,
            source_mask=source_mask,
            carried_volume_triangles_base=carried_volume_triangles,
        )
        contact_focus = render_contact_focus(
            camera,
            wrist_camera,
            state.robot,
            contact_preview,
            source_mask=source_mask,
            contact_cameras=contact_cameras,
            carried_volume_triangles_base=carried_volume_triangles,
        )

        # The persistent world view is deliberately sensor-clean.  All
        # grounding, self and imagination overlays belong to the dynamic
        # decision workspace below it.
        rasters: dict[str, np.ndarray] = {
            "agentview": rgb.copy(),
            "observed_scene": observed_scene,
            "imagination_scene": imagination_scene,
        }
        contact_focus_id = None
        if contact_focus is not None:
            contact_focus_id = "contact_focus"
            rasters[contact_focus_id] = contact_focus
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
        packet = ContextPacket(
            revision=state.observation_revision,
            world=WorldContextSpec(
                agentview_raster_id="agentview",
                observed_scene_raster_id="observed_scene",
                imagination_scene_raster_id="imagination_scene",
                contact_focus_raster_id=contact_focus_id,
                robot=state.robot,
                action=action_presentation,
                refinement_goal=(
                    state.refinement.instruction if state.refinement is not None else None
                ),
            ),
            catalog=EvidenceCatalogSpec(
                regions=tuple(region_specs),
                points=tuple(point_specs),
                seeds=tuple(seed_specs),
            ),
            decision=decision,
            rasters=rasters,
            projection=projection,
        )
        _validate_packet(packet)
        return packet


    def compile_imagination(self, workspace: ContextWorkspace) -> ContextPacket:
        """Compile the same trusted state into a focused SubAgent projection."""

        if workspace.state.refinement is None:
            raise ValueError("focused imagination context requires an active refinement")
        return self.compile(workspace, projection="imagination")

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
                (point for point in state.points.values() if point.within_region_id == region_id),
                None,
            )
            rotation = Rotation.from_quat(np.asarray(pose.quaternion_xyzw)).as_matrix()
            approach = tuple(float(value) for value in rotation[:, 2])
            delta = None
            if anchor is not None:
                delta = tuple(
                    float(value)
                    for value in (np.asarray(pose.position_xyz) - np.asarray(anchor.position_xyz))
                )
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
                if not preview_matches:
                    displayed_solve_ik = "mismatch"
            gripper_geometry = _seed_gripper_geometry(
                state.robot,
                pose,
                camera,
                rgb,
            )
            rasters[raster_id] = _candidate_crop(
                rgb,
                source.bbox_xyxy_px if source is not None else None,
                pose,
                camera,
                gripper_geometry=gripper_geometry,
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
                    family=artifact.family if artifact is not None else None,
                )
            )
        return specs


def _presentation_region_ref(
    state: ContextState,
    artifacts: Any | None,
) -> str | None:
    """Resolve the observed region supporting the current virtual target.

    Grasp actions already carry their source region.  Point actions carry a
    point ID, so placement previews must follow the point's parent region to
    keep the destination geometry in the Contact Camera frame.
    """

    source_ref = _observed_source_ref(artifacts)
    if source_ref in state.regions:
        return source_ref
    if source_ref in state.points:
        parent = state.points[source_ref].within_region_id
        return parent if parent in state.regions else None
    return None


def _contact_camera_center(
    target_xyz: tuple[float, float, float],
    subject_points: np.ndarray | None,
    required_points: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Frame both the virtual TCP and relevant observed geometry.

    Robust bounds ignore sparse RGB-D outliers.  The camera remains locked for
    the refinement session; only its initial centre is destination-aware.
    """

    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    point_groups: list[np.ndarray] = []
    if subject_points is not None:
        points = np.asarray(subject_points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            lower = np.quantile(points, 0.02, axis=0)
            upper = np.quantile(points, 0.98, axis=0)
            point_groups.extend((lower.reshape(1, 3), upper.reshape(1, 3)))
    if required_points is not None:
        required = np.asarray(required_points, dtype=np.float64).reshape(-1, 3)
        required = required[np.isfinite(required).all(axis=1)]
        if len(required):
            point_groups.extend(
                (
                    np.min(required, axis=0).reshape(1, 3),
                    np.max(required, axis=0).reshape(1, 3),
                )
            )
    if not point_groups:
        return tuple(float(value) for value in target)
    bounds = np.vstack(point_groups)
    lower = np.min(bounds, axis=0)
    upper = np.max(bounds, axis=0)
    lower = np.minimum(lower, target)
    upper = np.maximum(upper, target)
    center = 0.5 * (lower + upper)
    return tuple(float(value) for value in center)


def _preview_contact_points(
    robot: RobotState | None,
    preview: NearFieldPreview | None,
) -> np.ndarray | None:
    """Return private target-gripper vertices that must remain in frame."""

    if robot is None or preview is None or preview.gripper_opening is None:
        return None
    triangles = None
    if preview.joint_positions_rad is not None:
        fk = load_panda_urdf_fk()
        if fk is not None:
            try:
                triangles = fk.triangles(
                    np.asarray(preview.joint_positions_rad, dtype=np.float64),
                    float(preview.gripper_opening),
                )
            except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
                triangles = None
    if triangles is None:
        triangles = _direct_target_gripper_triangles(
            robot,
            preview.target_pose,
            preview.gripper_opening,
        )
    if triangles is None:
        return None
    points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    return np.ascontiguousarray(points) if len(points) else None


def _current_physical_contact_preview(
    workspace: ContextWorkspace,
) -> NearFieldPreview | None:
    """Anchor current-only Contact Views at the latest physical target."""

    if workspace.state.pending_action is not None:
        return None
    artifacts = workspace._private.last_physical_artifacts
    if workspace.state.last_physical_action is None or artifacts is None:
        return None
    focus = artifacts.focus_pose
    if focus is None:
        robot = workspace.state.robot
        focus = robot.tcp_pose if robot is not None else None
    if focus is None:
        return None
    return NearFieldPreview(
        target_pose=None,
        joint_positions_rad=None,
        gripper_opening=None,
        contact_frame_quaternion_xyzw=gravity_stable_contact_frame_quaternion(focus),
        contact_frame_position_xyz=focus.position_xyz,
    )


def _decision_spec(workspace: ContextWorkspace) -> DecisionWorkspaceSpec:
    state = workspace.state
    event = workspace._private.presentation_event
    if state.refinement is not None:
        return DecisionWorkspaceSpec(
            mode="editing",
            seed_ids=tuple(state.seeds)[:5],
        )

    if event is not None:
        if event.function_name == "done":
            return DecisionWorkspaceSpec(mode="terminal")
        result = event.result
        seed_ids = result.get("seed_ids")
        if isinstance(seed_ids, list):
            valid = tuple(str(value) for value in seed_ids if str(value) in state.seeds)[:5]
            return DecisionWorkspaceSpec(mode="seeds", seed_ids=valid)

    if state.pending_action is not None:
        return DecisionWorkspaceSpec(
            mode="proposal",
            seed_ids=tuple(state.seeds)[:5],
            action_id=state.pending_action.action_id,
        )

    region_id = event.result.get("region_id") if event is not None else None
    point_id = event.result.get("point_id") if event is not None else None
    regions, points = _grounding_references(state, region_id, point_id)
    primary = None
    if isinstance(point_id, str) and point_id in state.points:
        primary = f"point:{point_id}"
    elif isinstance(region_id, str) and region_id in state.regions:
        primary = f"region:{region_id}"

    if (
        state.last_physical_action is not None
        and workspace._private.last_physical_artifacts is not None
    ):
        return DecisionWorkspaceSpec(
            mode="contact",
            region_ids=regions,
            point_ids=points,
            primary_raster_id=primary,
        )

    if isinstance(region_id, str) or isinstance(point_id, str):
        return DecisionWorkspaceSpec(
            mode="grounding",
            region_ids=regions,
            point_ids=points,
            primary_raster_id=primary,
        )
    return DecisionWorkspaceSpec(mode="idle")


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


def _seed_gripper_geometry(
    robot: RobotState | None,
    pose: Pose,
    camera: dict[str, Any],
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return target triangles and silhouette bounds when projection has area.

    Side grasps collapse to an edge-on sliver from agentview; those cards keep
    the schematic contour glyph. Other poses use projected 3-D line art.
    """

    if robot is None:
        return None
    triangles = _direct_target_gripper_triangles(
        robot,
        pose,
        robot.gripper_opening,
    )
    if triangles is None:
        return None
    projected = project_world_to_pixel(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 3),
        camera["intrinsics"],
        camera["pose_mat"],
    ).reshape(-1, 3, 3)
    mask = rasterize_silhouette(
        projected[..., :2],
        projected[..., 2],
        rgb.shape[1],
        rgb.shape[0],
    )
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    height = int(ys.max() - ys.min()) + 1
    width = int(xs.max() - xs.min()) + 1
    if min(height, width) < 14 or int(mask.sum()) < 120:
        return None
    return triangles, mask


def _candidate_crop(
    rgb: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    pose: Pose,
    camera: dict[str, Any],
    *,
    gripper_geometry: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    gripper_triangles, gripper_mask = (
        gripper_geometry if gripper_geometry is not None else (None, None)
    )
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
    line_art = rgb.copy()
    if gripper_triangles is not None:
        projected = project_world_to_pixel(
            np.asarray(gripper_triangles, dtype=np.float64).reshape(-1, 3),
            camera["intrinsics"],
            camera["pose_mat"],
        ).reshape(-1, 3, 3)
        overlay_projected_mesh_outline(
            line_art,
            gripper_triangles,
            projected[..., :2],
            projected[..., 2],
            camera_position_base=np.asarray(camera["pose_mat"], dtype=np.float64)[
                :3, 3
            ],
        )
    raster = line_art[top:bottom, left:right].copy()
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


def _pose_origin_pixel(pose: Pose, camera: dict[str, Any]) -> tuple[float, float] | None:
    projected = _project_pose(pose, camera)
    return None if projected is None else projected[:2]


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
    if seed is None:
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
        ) + target_rotation @ np.asarray(tcp_to_hand_local_xyz, dtype=np.float64).reshape(3)
        position_error = float(np.linalg.norm(actual_hand[:3, 3] - expected_hand_position))
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


def _compile_observed_scene(
    workspace: ContextWorkspace,
    agentview: dict[str, Any],
    wrist_camera: dict[str, Any] | None,
) -> np.ndarray:
    """Compile the complementary observed view without exposing calibration.

    Offline fixtures intentionally retain the camera-aligned RGB-D fallback.
    A live LIBERO runner injects the opposite-camera provider; its pose is
    episode-stable and its RGB is refreshed once per physical revision.
    """

    private = workspace._private
    provider = private.opposite_scene_camera_provider
    if not callable(provider):
        return render_scene_view(
            agentview,
            wrist_camera,
            workspace.state.robot,
            dark=False,
        )
    revision = workspace.state.observation_revision
    if (
        private.opposite_scene_camera is None
        or private.opposite_scene_camera_revision != revision
    ):
        if private.opposite_scene_center_base_xyz is None:
            private.opposite_scene_center_base_xyz = _observed_workspace_center(agentview)
        pose_mat = np.asarray(agentview["pose_mat"], dtype=np.float64).reshape(4, 4)
        forward = pose_mat[:3, 2]
        private.opposite_scene_camera = provider(
            OppositeSceneCameraRequest(
                center_base_xyz=private.opposite_scene_center_base_xyz,
                agentview_forward_base_xyz=tuple(float(value) for value in forward),
                width=OBSERVED_SCENE_WIDTH,
                height=OBSERVED_SCENE_HEIGHT,
            )
        )
        private.opposite_scene_camera_revision = revision
    return _rgb(private.opposite_scene_camera).copy()


def _observed_workspace_center(camera: dict[str, Any]) -> tuple[float, float, float]:
    """Estimate one robust LIBERO workspace centre from the first RGB-D frame."""

    try:
        depth = np.asarray(camera["images"]["depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError):
        return (0.55, 0.0, 0.08)
    height, width = depth.shape
    stride = max(4, min(height, width) // 80)
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth[rows, cols]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.03) & (sampled_depth < 3.0)
    if np.count_nonzero(valid) < 32:
        return (0.55, 0.0, 0.08)
    pixels = np.column_stack(
        (
            cols[valid].astype(np.float64),
            rows[valid].astype(np.float64),
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        )
    )
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    points_camera = rays * sampled_depth[valid, None]
    homogeneous = np.column_stack(
        (points_camera, np.ones(len(points_camera), dtype=np.float64))
    )
    points_base = (base_from_camera @ homogeneous.T).T[:, :3]
    # Low surfaces contain the manipulable workspace while excluding the wall
    # and most robot links.  Quantile midpoints resist a single close object.
    usable = points_base[
        np.isfinite(points_base).all(axis=1)
        & (points_base[:, 2] >= -0.06)
        & (points_base[:, 2] <= 0.28)
    ]
    if len(usable) < 32:
        return (0.55, 0.0, 0.08)
    low, high = np.percentile(usable[:, :2], (12.0, 88.0), axis=0)
    center_xy = 0.5 * (low + high)
    center_z = float(np.clip(np.percentile(usable[:, 2], 65.0) + 0.06, 0.06, 0.18))
    center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
    if not np.isfinite(center).all():
        return (0.55, 0.0, 0.08)
    return tuple(float(value) for value in center)


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
