"""Deterministic compiler for Main and focused Imagination canvases."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from vaw.context_runtime import evidence as evidence_module
from vaw.context_runtime.attached_object import volume_triangles_base
from vaw.context_runtime.contact_camera import ContactCameraRequest
from vaw.context_runtime.follow_through import follow_through_summary
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
    CONTACT_PANEL_GAP,
    NearFieldPreview,
    PreviewGripperStyle,
    _direct_target_gripper_triangles,
    gravity_stable_contact_frame_quaternion,
    render_contact_auxiliary,
    render_contact_focus,
)
from vaw.context_runtime.plumb_line import (
    PlumbLine,
    compute_plumb_line,
    payload_bottom_from_triangles,
    surface_height_below,
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
from vaw.context_runtime.workspace import ContextWorkspace
from vaw.mmskill import MMSkill

CONTEXT_SCHEMA = "vaw-context-v57-mmskill-reference"
CONTEXT_WEB_SCHEMA_VERSION = 57
# FRONT and SIDE stay level and metrically legible.  A separate steep diagonal
# camera occupies the former blank upper-right slot, exposing lateral placement
# and clearing the Panda hand without changing Contact semantics.
CONTACT_AUXILIARY_ELEVATION_DEG = 68.0
CONTEXT_WIDTH = 2048
CONTEXT_HEIGHT = 1280
IMAGINATION_CONTEXT_WIDTH = 2048
IMAGINATION_CONTEXT_HEIGHT = 1280
# Required live-hand points farther than this from TCP only inflate the
# Contact distance; fingers and palm stay inside this radius.
_LIVE_GRIPPER_FRAMING_RADIUS_M = 0.08
# 当前闭合夹爪的 BASE-Z 标记采用固定的米制跨度。TCP 上方只保留一小段，
# 下方延伸更长，以便在放置时直接阅读垂直投影；它不是轨迹或落点预测。
_GRIPPER_Z_AXIS_ABOVE_M = 0.04
_GRIPPER_Z_AXIS_BELOW_M = 0.24

# Native raster sizes match the two-column policy slots. Main places NOW next
# to the steep auxiliary contact view, with level Contact Front/Side below.
MAIN_CONTACT_WIDTH = 1020
MAIN_CONTACT_HEIGHT = 631
IMAGINATION_CONTACT_WIDTH = 2042
IMAGINATION_CONTACT_HEIGHT = 636

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
    status: str = "verified"

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
        if self.status != "verified":
            out["status"] = self.status
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
    status: str = "verified"

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
        if self.status != "verified":
            out["status"] = self.status
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
    imagination_scene_raster_id: str
    contact_front_raster_id: str | None
    contact_side_raster_id: str | None
    contact_auxiliary_raster_id: str | None
    robot: RobotState | None
    action: dict[str, Any] | None
    refinement_goal: str | None
    # Tilt of the SIDE contact panel above the horizon. The panel label has to
    # announce it, otherwise the policy reads an oblique image as a level one.
    contact_side_elevation_deg: float = 0.0
    source_follow_through: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        result = {
            "agentviewRasterId": self.agentview_raster_id,
            "imaginationSceneRasterId": self.imagination_scene_raster_id,
            "contactFrontRasterId": self.contact_front_raster_id,
            "contactSideRasterId": self.contact_side_raster_id,
            "contactAuxiliaryRasterId": self.contact_auxiliary_raster_id,
            "contactSideElevationDeg": round(float(self.contact_side_elevation_deg), 1),
            "robot": self.robot.summary() if self.robot is not None else None,
            "action": self.action,
            "refinementGoal": self.refinement_goal,
        }
        if self.source_follow_through:
            result["sourceFollowThrough"] = self.source_follow_through
        return result


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
class SkillReferenceSpec:
    """当前已加载技能的一张历史参考；只含策略可见内容。"""

    reference_id: str
    state: str
    view: str
    when_to_use: str
    visual_cue: str
    raster_id: str

    def summary(self) -> dict[str, str]:
        return {
            "referenceId": self.reference_id,
            "state": self.state,
            "view": self.view,
            "whenToUse": self.when_to_use,
            "visualCue": self.visual_cue,
            "rasterId": self.raster_id,
        }


@dataclass(frozen=True)
class SkillReferenceBoardSpec:
    """技能正文加载后才出现的视觉参考集合。"""

    skill_id: str
    references: tuple[SkillReferenceSpec, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "skillId": self.skill_id,
            "references": [item.summary() for item in self.references],
        }


@dataclass(frozen=True)
class ContextPacket:
    revision: int
    world: WorldContextSpec
    catalog: EvidenceCatalogSpec
    decision: DecisionWorkspaceSpec
    rasters: dict[str, np.ndarray]
    skill_reference: SkillReferenceBoardSpec | None = None
    projection: ContextProjection = "main"
    schema: str = CONTEXT_SCHEMA
    imagination_attempts: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """JSON-safe packet metadata; RGB arrays are represented by ids."""

        width, height = (
            (IMAGINATION_CONTEXT_WIDTH, IMAGINATION_CONTEXT_HEIGHT)
            if self.projection == "imagination"
            else (CONTEXT_WIDTH, CONTEXT_HEIGHT)
        )
        return {
            "schema": self.schema,
            "projection": self.projection,
            "revision": self.revision,
            "viewport": {"width": width, "height": height},
            "world": self.world.summary(),
            "catalog": self.catalog.summary(),
            "decision": self.decision.summary(),
            "skillReference": (
                self.skill_reference.summary()
                if self.skill_reference is not None
                else None
            ),
            "rasterIds": list(self.rasters),
        }

    def manifest(self) -> dict[str, Any]:
        action = self.world.action
        prediction = action.get("prediction") if action is not None else None
        action_executable = bool(
            isinstance(prediction, dict)
            and prediction.get("solve_ik") == "returned"
        )
        result: dict[str, Any] = {
            "action_proposal": (
                {
                    "action_id": action.get("action_id"),
                    "intent": action.get("intent"),
                    "state": action.get("status"),
                    "executable": action_executable,
                }
                if action is not None
                else None
            ),
            "regions": [
                {
                    "id": item.region_id,
                    "query": item.query,
                    **(
                        {"status": item.status}
                        if item.status != "verified"
                        else {}
                    ),
                }
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
                    **(
                        {"status": item.status}
                        if item.status != "verified"
                        else {}
                    ),
                }
                for item in self.catalog.points
            ],
            "seed_ids": [item.seed_id for item in self.catalog.seeds],
        }
        if self.imagination_attempts:
            result["imagination_attempts"] = self.imagination_attempts
        if self.world.source_follow_through:
            result["source_follow_through"] = self.world.source_follow_through
        return result

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
        active_mmskill: MMSkill | None = None,
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
        if (
            contact_preview is not None
            and contact_preview.contact_frame_quaternion_xyzw is not None
        ):
            if private.contact_frame_quaternion_xyzw is None:
                private.contact_frame_quaternion_xyzw = (
                    contact_preview.contact_frame_quaternion_xyzw
                )
                private.contact_frame_source_revision = state.observation_revision
            contact_preview = replace(
                contact_preview,
                contact_frame_quaternion_xyzw=(
                    private.contact_frame_quaternion_xyzw
                ),
            )
            private.trace_diagnostics["contact_frame"] = {
                "quaternion_xyzw": [
                    round(float(value), 8)
                    for value in private.contact_frame_quaternion_xyzw
                ],
                "created_revision": private.contact_frame_source_revision,
                "current_revision": state.observation_revision,
                "locked_across_revision": (
                    private.contact_frame_source_revision
                    != state.observation_revision
                ),
            }
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
        framing_source_points = None
        if source_ref is not None:
            source_geometry = private.region_geometry.get(source_ref)
            if source_geometry is not None:
                framing_source_points = source_geometry.filtered_object_points_base
        # Contact framing follows the active action, while the amber overlay
        # follows evidence validity.  Keeping those responsibilities separate
        # prevents an expired ActionProposal from hiding a still-verified
        # detection.  When an action exists it still owns camera framing;
        # otherwise the verified evidence set defines the observable crop.
        evidence_mask, evidence_points = _verified_contact_evidence(state, private)
        verified_region_key = ",".join(
            sorted(
                region_id
                for region_id, region in state.regions.items()
                if region.status == "verified"
            )
        )
        contact_cameras = None
        contact_width, contact_panel_height = (
            (IMAGINATION_CONTACT_WIDTH, IMAGINATION_CONTACT_HEIGHT)
            if projection == "imagination"
            else (MAIN_CONTACT_WIDTH, MAIN_CONTACT_HEIGHT)
        )
        if (
            contact_preview is not None
            and contact_preview.contact_frame_position_xyz is not None
            and contact_preview.contact_frame_quaternion_xyzw is not None
            and callable(private.contact_camera_provider)
        ):
            action_id = (
                state.action_proposal.action_id if state.action_proposal is not None else None
            )
            # The action-reading Contact pair is always level.  A separate
            # auxiliary camera supplies the steep oblique evidence.
            side_elevation_deg = 0.0
            if private.contact_camera_side_elevation_deg != side_elevation_deg:
                # Entering or leaving carry changes which azimuth is occluded,
                # so the session sign lock has to be re-earned at the new tilt.
                private.contact_camera_signs = None
                private.contact_camera_side_elevation_deg = side_elevation_deg
            camera_cache_key = (
                f"{action_id or 'observed'}:{projection}:"
                f"{contact_width}x{contact_panel_height}:e{side_elevation_deg:.0f}:"
                f"aux={CONTACT_AUXILIARY_ELEVATION_DEG:.0f}:"
                f"regions={verified_region_key or 'none'}"
            )
            if (
                private.contact_camera_pair is None
                or private.contact_camera_action_id != camera_cache_key
            ):
                required_preview_points = _required_contact_points(
                    state.robot,
                    contact_preview,
                )
                # Before an ActionProposal exists, frame the current gripper
                # together with every verified detect surface so the persistent
                # amber overlay is actually visible rather than merely
                # projected outside a stale, gripper-only camera crop.
                subject_points = (
                    framing_source_points
                    if framing_source_points is not None
                    else evidence_points
                )
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
                    "verified_region_ids": (
                        verified_region_key.split(",") if verified_region_key else []
                    ),
                }
                private.contact_camera_pair = private.contact_camera_provider(
                    ContactCameraRequest(
                        center_base_xyz=contact_center,
                        frame_quaternion_xyzw=(
                            contact_preview.contact_frame_quaternion_xyzw
                        ),
                        width=contact_width,
                        panel_height=contact_panel_height,
                        subject_points_base=subject_points,
                        required_points_base=required_preview_points,
                        preferred_signs=private.contact_camera_signs,
                        side_elevation_deg=side_elevation_deg,
                        auxiliary_elevation_deg=CONTACT_AUXILIARY_ELEVATION_DEG,
                    )
                )
                if private.contact_camera_signs is None:
                    selection = private.contact_camera_pair.selection
                    private.contact_camera_signs = (
                        int(selection.front_sign),
                        int(selection.side_sign),
                    )
                private.contact_camera_action_id = camera_cache_key
            contact_cameras = private.contact_camera_pair
        # The global panel is direct visual context, not a second preview.
        # Keep it identical to the latest physical RGB; all action geometry
        # and coordinate cues belong to the dedicated Contact views.
        imagination_scene = rgb.copy()
        agentview = rgb.copy()
        # 旧 plumb cue 依赖 attachment metadata，同一真实抓持会因 Action
        # lineage 不同而时有时无。当前视觉协议改为只使用真实 TCP 的稳定
        # BASE-Z 标记；旧几何函数暂留给离线诊断，不再进入 policy Canvas。
        plumb_lines: list[PlumbLine] = []
        # Preview 已经同时画出 current 与 future gripper。此时隐藏现实夹爪
        # 的黄色闭合通道和 Z 标记，避免 VLM 把本体 cue 误认为候选动作。
        show_current_gripper_cues = target is None
        grasp_sweep_segment = (
            _observed_grasp_sweep_segment(state.robot)
            if show_current_gripper_cues
            else None
        )
        gripper_z_axis = (
            _observed_closed_gripper_z_axis(state.robot, private)
            if show_current_gripper_cues
            else None
        )
        contact_focus = render_contact_focus(
            camera,
            wrist_camera,
            state.robot,
            contact_preview,
            source_mask=evidence_mask,
            source_points_base=evidence_points,
            contact_cameras=contact_cameras,
            carried_volume_triangles_base=carried_volume_triangles,
            grasp_sweep_segment_base=grasp_sweep_segment,
            gripper_z_axis_base=gripper_z_axis,
            plumb_lines=plumb_lines,
            output_width=contact_width,
            panel_height=contact_panel_height,
        )
        contact_auxiliary = None
        if contact_cameras is not None and contact_cameras.auxiliary is not None:
            contact_auxiliary = render_contact_auxiliary(
                contact_cameras.auxiliary,
                state.robot,
                contact_preview,
                source_points_base=evidence_points,
                carried_volume_triangles_base=carried_volume_triangles,
                grasp_sweep_segment_base=grasp_sweep_segment,
                gripper_z_axis_base=gripper_z_axis,
                plumb_lines=plumb_lines,
                output_width=contact_width,
                panel_height=contact_panel_height,
            )

        # 接触 cue 和坐标提示只属于 Contact 视图；全局 RGB 保持干净。
        rasters: dict[str, np.ndarray] = {
            "agentview": agentview,
            "imagination_scene": imagination_scene,
        }
        contact_front_id = None
        contact_side_id = None
        contact_auxiliary_id = None
        if contact_focus is not None:
            contact_front_id = "contact_front"
            contact_side_id = "contact_side"
            side_top = contact_panel_height + CONTACT_PANEL_GAP
            rasters[contact_front_id] = np.ascontiguousarray(
                contact_focus[:contact_panel_height]
            )
            rasters[contact_side_id] = np.ascontiguousarray(
                contact_focus[side_top : side_top + contact_panel_height]
            )
        if contact_auxiliary is not None:
            contact_auxiliary_id = "contact_auxiliary"
            rasters[contact_auxiliary_id] = np.ascontiguousarray(contact_auxiliary)
        skill_reference = _compile_skill_references(active_mmskill, rasters)
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
                imagination_scene_raster_id="imagination_scene",
                contact_front_raster_id=contact_front_id,
                contact_side_raster_id=contact_side_id,
                contact_auxiliary_raster_id=contact_auxiliary_id,
                contact_side_elevation_deg=(
                    0.0
                    if contact_cameras is None
                    else float(contact_cameras.selection.side_elevation_deg)
                ),
                robot=state.robot,
                action=action_presentation,
                refinement_goal=(
                    state.imagination.instruction if state.imagination is not None else None
                ),
                source_follow_through=follow_through_summary(workspace),
            ),
            catalog=EvidenceCatalogSpec(
                regions=tuple(region_specs),
                points=tuple(point_specs),
                seeds=tuple(seed_specs),
            ),
            decision=decision,
            rasters=rasters,
            skill_reference=skill_reference,
            projection=projection,
            imagination_attempts=state.imagination_attempts.summary(),
        )
        _validate_packet(packet)
        return packet


    def compile_imagination(
        self,
        workspace: ContextWorkspace,
        *,
        active_mmskill: MMSkill | None = None,
    ) -> ContextPacket:
        """Compile the same trusted state into a focused SubAgent projection."""

        if workspace.state.imagination is None:
            raise ValueError("focused imagination context requires an active imagination session")
        return self.compile(
            workspace,
            projection="imagination",
            active_mmskill=active_mmskill,
        )

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
                    status=region.status,
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
                    status=point.status,
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


def _verified_contact_evidence(
    state: ContextState,
    private: PrivateEnvContext,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Collect only currently verified detect regions for Contact overlays.

    The agentview mask supports the RGB-D fallback renderer; metric surface
    points are reprojected into the dedicated Contact Cameras.  Occluded
    evidence remains available to the evidence lifecycle but is deliberately
    not visualized as current object surface.  Invalidated evidence has already
    been removed by ``revalidate_evidence``.
    """

    combined_mask: np.ndarray | None = None
    point_sets: list[np.ndarray] = []
    for region_id, region in state.regions.items():
        if region.status != "verified":
            continue

        mask = private.region_masks.get(region_id)
        if mask is not None:
            candidate = np.asarray(mask, dtype=bool)
            if candidate.ndim == 2:
                if combined_mask is None:
                    combined_mask = candidate.copy()
                elif combined_mask.shape == candidate.shape:
                    combined_mask |= candidate

        geometry = private.region_geometry.get(region_id)
        if geometry is None:
            continue
        points = np.asarray(
            geometry.filtered_object_points_base,
            dtype=np.float64,
        )
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) >= 3:
            point_sets.append(points)

    combined_points = (
        np.ascontiguousarray(np.vstack(point_sets)) if point_sets else None
    )
    return (
        np.ascontiguousarray(combined_mask) if combined_mask is not None else None,
        combined_points,
    )


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


def _stack_contact_points(*groups: np.ndarray | None) -> np.ndarray | None:
    chunks: list[np.ndarray] = []
    for group in groups:
        if group is None:
            continue
        points = np.asarray(group, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            chunks.append(points)
    if not chunks:
        return None
    return np.ascontiguousarray(np.vstack(chunks))


def _current_gripper_contact_points(robot: RobotState | None) -> np.ndarray | None:
    """Live finger/palm vertices that every Contact panel must keep in frame.

    Destination clouds (a basket opening, a table patch) are much denser than
    the hand.  If the current gripper is only mixed into that subject and then
    robust-quantiled, FRONT looks at the container face and clips the real
    hand.  Required points use min/max, so the gripper has to enter that set
    on its own.  This is framing only: it does not draw a Preview ghost.
    """

    if robot is None or robot.gripper_opening is None:
        return None
    if robot.joint_positions_rad is not None:
        fk = load_panda_urdf_fk()
        if fk is not None:
            try:
                triangles = fk.triangles(
                    np.asarray(robot.joint_positions_rad, dtype=np.float64),
                    float(robot.gripper_opening),
                )
            except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
                triangles = None
            else:
                points = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
                points = points[np.isfinite(points).all(axis=1)]
                if len(points):
                    origin = (
                        None
                        if robot.tcp_pose is None
                        else np.asarray(
                            robot.tcp_pose.position_xyz, dtype=np.float64
                        ).reshape(3)
                    )
                    return _clip_points_near_origin(points, origin)
    if robot.tcp_pose is None:
        return None
    origin = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64).reshape(3)
    extents = np.array(
        [
            [sx, sy, sz]
            for sx in (-0.04, 0.04)
            for sy in (-0.04, 0.04)
            for sz in (-0.02, 0.06)
        ],
        dtype=np.float64,
    )
    return np.ascontiguousarray(origin + extents)


def _clip_points_near_origin(
    points: np.ndarray,
    origin: np.ndarray | None,
) -> np.ndarray:
    """Keep a hand-scale cluster.  Never fall back to the forearm mesh."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if origin is not None:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        keep = np.linalg.norm(points - origin, axis=1) <= _LIVE_GRIPPER_FRAMING_RADIUS_M
        if np.any(keep):
            return np.ascontiguousarray(points[keep])
        nearest = points[int(np.argmin(np.linalg.norm(points - origin, axis=1)))]
    else:
        nearest = np.median(points, axis=0)
    keep = np.linalg.norm(points - nearest, axis=1) <= _LIVE_GRIPPER_FRAMING_RADIUS_M
    if not np.any(keep):
        return np.ascontiguousarray(points[int(np.argmin(
            np.linalg.norm(points - nearest, axis=1)
        ))].reshape(1, 3))
    return np.ascontiguousarray(points[keep])


def _required_contact_points(
    robot: RobotState | None,
    preview: NearFieldPreview | None,
) -> np.ndarray | None:
    """Keep the live gripper, and any planned Preview gripper, inside Contact."""

    return _stack_contact_points(
        _current_gripper_contact_points(robot),
        _preview_contact_points(robot, preview),
    )


def _preview_contact_points(
    robot: RobotState | None,
    preview: NearFieldPreview | None,
) -> np.ndarray | None:
    """Return private *planned* gripper vertices that must remain in frame."""

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
    if not len(points):
        return None
    origin = (
        None
        if preview.target_pose is None
        else np.asarray(preview.target_pose.position_xyz, dtype=np.float64).reshape(3)
    )
    return _clip_points_near_origin(points, origin)


def _current_physical_contact_preview(
    workspace: ContextWorkspace,
) -> NearFieldPreview | None:
    """Anchor current-only Contact Views at the latest physical target.

    ``gripper_opening`` stays unset on purpose: a filled Preview would draw a
    ghost on top of the live hand.  The live gripper enters framing through
    ``_current_gripper_contact_points`` instead.
    """

    if workspace.state.action_proposal is not None:
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
    if state.imagination is not None:
        return DecisionWorkspaceSpec(
            mode="editing",
            seed_ids=tuple(state.seeds)[:5],
        )

    if event is not None:
        if event.function_name == "finish_task":
            return DecisionWorkspaceSpec(mode="terminal")
        result = event.result
        seed_ids = result.get("seed_ids")
        if isinstance(seed_ids, list):
            valid = tuple(str(value) for value in seed_ids if str(value) in state.seeds)[:5]
            return DecisionWorkspaceSpec(mode="seeds", seed_ids=valid)

    region_id = event.result.get("region_id") if event is not None else None
    point_id = event.result.get("point_id") if event is not None else None
    regions, points = _grounding_references(state, region_id, point_id)
    primary = None
    if isinstance(point_id, str) and point_id in state.points:
        primary = f"point:{point_id}"
    elif isinstance(region_id, str) and region_id in state.regions:
        primary = f"region:{region_id}"

    if state.action_proposal is not None:
        return DecisionWorkspaceSpec(
            mode="proposal",
            seed_ids=tuple(state.seeds)[:5],
            action_id=state.action_proposal.action_id,
            region_ids=regions,
            point_ids=points,
            primary_raster_id=primary,
        )

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


def _compute_plumb_lines(
    state: ContextState,
    private: PrivateEnvContext,
    camera: dict[str, Any],
    target: Any,
    carried_target_triangles: np.ndarray | None,
    active_artifacts: Any,
) -> list[PlumbLine]:
    """Deterministic vertical drops for the carried payload and the preview.

    The current plumb hangs from the attached payload at the observed TCP;
    the preview plumb hangs from the payload (or bare TCP) at the unexecuted
    target pose.  Both intersect the currently observed depth surface.

    The qualitative XY direction arrow only exists when the active action was built from a
    semantic anchor point (``preview_pose``): a whole-region centroid (for a
    basket, roughly its body centre rather than its opening) reads as a
    biased correction and sends refinement chasing a skewed target, so no
    anchor means no arrow — the footprint and height remain.
    """

    robot = state.robot
    silhouette = evidence_module.robot_silhouette(camera, robot)

    target_center: tuple[float, float, float] | None = None
    context = (
        active_artifacts.planning_context if active_artifacts is not None else None
    )
    if context is not None and context.source_kind == "point":
        point = state.points.get(context.source_ref)
        if point is not None:
            target_center = tuple(float(value) for value in point.position_xyz)

    plumbs: list[PlumbLine] = []
    if (
        private.attachment_hypothesis is not None
        and robot is not None
        and robot.tcp_pose is not None
    ):
        current_triangles = volume_triangles_base(
            private.attachment_hypothesis,
            robot.tcp_pose,
        )
        anchor, corners = payload_bottom_from_triangles(current_triangles)
        surface = surface_height_below(camera, anchor, exclude_mask=silhouette)
        plumb = compute_plumb_line(
            "current",
            anchor,
            surface,
            footprint_base=corners,
            target_center_base_xyz=target_center,
        )
        if plumb is not None:
            plumbs.append(plumb)

    if target is not None:
        if carried_target_triangles is not None:
            anchor, corners = payload_bottom_from_triangles(carried_target_triangles)
        else:
            anchor = np.asarray(target.pose.position_xyz, dtype=np.float64)
            corners = None
        surface = surface_height_below(camera, anchor, exclude_mask=silhouette)
        plumb = compute_plumb_line(
            "preview",
            anchor,
            surface,
            footprint_base=corners,
            target_center_base_xyz=target_center,
        )
        if plumb is not None:
            plumbs.append(plumb)
    return plumbs


def _observed_grasp_sweep_segment(robot: RobotState | None) -> np.ndarray | None:
    """Project no intent: expose only the observed two-finger FK channel."""

    if (
        robot is None
        or robot.joint_positions_rad is None
        or robot.gripper_opening is None
    ):
        return None
    provider = load_panda_urdf_fk()
    if provider is None:
        return None
    try:
        segment = provider.grasp_sweep_segment(
            np.asarray(robot.joint_positions_rad, dtype=np.float64),
            float(robot.gripper_opening),
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    return np.ascontiguousarray(segment)


def _observed_closed_gripper_z_axis(
    robot: RobotState | None,
    private: PrivateEnvContext,
) -> np.ndarray | None:
    """返回 ``[下端, TCP, 上端]`` 的真实 BASE-Z 本体标记。

    ``grasp_follow_through`` 在一次成功的 ``close_gripper`` 后建立，并在
    ``open_gripper`` 时清除；它只用来确认最近的夹爪命令状态，不要求
    object proxy、attachment hypothesis 或任务语义成立。
    """

    if (
        private.grasp_follow_through is None
        or robot is None
        or robot.tcp_pose is None
    ):
        return None
    anchor = np.asarray(robot.tcp_pose.position_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(anchor).all():
        return None
    lower = anchor - np.array([0.0, 0.0, _GRIPPER_Z_AXIS_BELOW_M])
    upper = anchor + np.array([0.0, 0.0, _GRIPPER_Z_AXIS_ABOVE_M])
    return np.ascontiguousarray(np.stack((lower, anchor, upper), axis=0))


def _compile_skill_references(
    skill: MMSkill | None,
    rasters: dict[str, np.ndarray],
) -> SkillReferenceBoardSpec | None:
    """将已加载技能的历史图片加入 Packet，不触碰当前环境私有状态。"""

    if skill is None or not skill.references:
        return None
    specs: list[SkillReferenceSpec] = []
    for reference in skill.references:
        raster_id = f"mmskill:{skill.skill_id}:{reference.reference_id}"
        with Image.open(io.BytesIO(reference.image_bytes)) as image:
            rasters[raster_id] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        specs.append(
            SkillReferenceSpec(
                reference_id=reference.reference_id,
                state=reference.state,
                view=reference.view,
                when_to_use=reference.when_to_use,
                visual_cue=reference.visual_cue,
                raster_id=raster_id,
            )
        )
    return SkillReferenceBoardSpec(skill.skill_id, tuple(specs))


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
    "IMAGINATION_CONTEXT_HEIGHT",
    "IMAGINATION_CONTEXT_WIDTH",
    "SeedSpec",
    "ContextCompiler",
    "ContextPacket",
    "DecisionMode",
    "DecisionWorkspaceSpec",
    "EvidenceCatalogSpec",
    "PointSpec",
    "RegionSpec",
    "SkillReferenceBoardSpec",
    "SkillReferenceSpec",
    "WorldContextSpec",
    "encode_png_data_url",
]
