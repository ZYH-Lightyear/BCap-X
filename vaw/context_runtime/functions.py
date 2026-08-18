"""Implementations of the M1.4 agent-visible functions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from capx.utils.depth_utils import depth_to_pointcloud
from vaw.context_runtime.attached_object import (
    ObjectProxyCandidate,
    bind_proxy_to_tcp,
    fit_gravity_stable_proxy,
)
from vaw.context_runtime.errors import ContextFunctionError
from vaw.context_runtime.geometry import (
    GRASP_POSE_TO_CONTACT_M,
    graspnet_pose_to_panda_hand,
    local_surface_depth,
    pixel_to_base,
    shift_along_approach,
    tcp_position_from_hand_pose,
)
from vaw.context_runtime.model import (
    ActionPrediction,
    ActionSeed,
    ActionTarget,
    LastPhysicalAction,
    PendingAction,
    PointEvidence,
    Pose,
    RefinementSession,
    RegionEvidence,
)
from vaw.context_runtime.motion import MotionBackendError, MotionPlan
from vaw.context_runtime.private import (
    ActionArtifacts,
    LastPhysicalArtifacts,
    PlanningContext,
    RegionGeometryArtifact,
    SeedArtifacts,
    VisualEdit,
    build_edit_summary,
)
from vaw.context_runtime.semantic_grounding import (
    candidate_prompt,
    parse_candidates,
    parse_choice,
    render_candidate_review,
    review_prompt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaw.context_runtime.workspace import ContextWorkspace


@dataclass(frozen=True)
class _PreparedGrasp:
    target: Pose
    score: float
    raw_index: int
    approach_z: float
    family: str = "cgn"


_LIBERO_TOP_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)
_TOP_GRASP_OFFSET_M = 0.04
_MAX_TOP_GRASPS = 1
_MAX_PCA_GRASPS = 2
_MAX_CGN_GRASPS = 2
_SEMANTIC_QUERY_RETRIES = 2


_COMMIT_FAILURE_RECOVERY_HINT = (
    "region/seed/action_id 已随新观测作废，不能再 commit 该 id。"
    "若仍操作同一物体，请重新 detection_and_sam（优先使用 source_query）；"
    "若从当前真实 TCP 继续靠近，由 Main 调用 refine_action。"
)


class ContextFunctions:
    """Function semantics separated from workspace lifecycle/dispatch."""

    MAX_DELTA_M = 0.03
    MAX_ROTATION_DEG = 10.0

    def __init__(self, workspace: ContextWorkspace) -> None:
        self.ws = workspace
        self.main_handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "detection_and_sam": self.detection_and_sam,
            "locate_point": self.locate_point,
            "propose_grasps": self.propose_grasps,
            "propose_pose": self.propose_pose,
            "select": self.select,
            "delta_move": self.direct_delta_move,
            "commit": self.commit,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "reject_action": self.reject_action,
            "done": self.done,
        }
        self.imagination_handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "delta_move": self.delta_move,
            "rotate": self.rotate,
            "show_rotation_gizmo": self.show_rotation_gizmo,
            "finish_imagination": self.finish_imagination,
        }

    # ---------------------------------------------------------- perception --
    def detection_and_sam(
        self,
        query: str,
        within_region_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_query = " ".join(str(query).casefold().split())
        for region in reversed(tuple(self.ws.state.regions.values())):
            if (
                " ".join(region.query.casefold().split()) == normalized_query
                and region.within_region_id == within_region_id
            ):
                self.ws._private.trace_diagnostics["semantic_grounding"] = {
                    "mode": "revision_local_cache",
                    "region_id": region.region_id,
                }
                return {
                    "region_id": region.region_id,
                    "bbox_xyxy_px": [round(float(value), 3) for value in region.bbox_xyxy_px],
                }
        camera = self.ws._camera()
        rgb = _camera_image(camera, "rgb")
        crop_rgb, origin = self.ws._crop_rgb(rgb, within_region_id)
        semantic_crop = self.ws._semantic_crop(within_region_id)
        semantic_box = self._semantic_bbox(semantic_crop, str(query))
        local_box = _scale_box_between_images(
            semantic_box,
            source_shape=semantic_crop.shape[:2],
            target_shape=crop_rgb.shape[:2],
        )
        semantic_diagnostics = self.ws._private.trace_diagnostics.get("semantic_grounding")
        if isinstance(semantic_diagnostics, dict):
            semantic_diagnostics["semantic_rgb_shape"] = list(semantic_crop.shape)
            semantic_diagnostics["observation_rgb_shape"] = list(crop_rgb.shape)
            semantic_diagnostics["box_observation_px"] = [
                round(float(value), 3) for value in local_box
            ]
        results = self.ws._call_backend("segment_sam3_box_prompt", crop_rgb, list(local_box))
        if not isinstance(results, (list, tuple)):
            raise ContextFunctionError("segmentation did not return a result list")
        valid = [
            item for item in results if isinstance(item, dict) and item.get("mask") is not None
        ]
        if not valid:
            raise ContextFunctionError(f"segmentation returned no mask for '{query}'")
        best = max(valid, key=lambda item: float(item.get("score", 0.0)))
        local_mask = np.asarray(best["mask"], dtype=bool)
        if local_mask.shape != crop_rgb.shape[:2]:
            raise ContextFunctionError(
                f"segmentation mask shape {local_mask.shape} does not match image {crop_rgb.shape[:2]}"
            )
        x0, y0 = origin
        global_mask = np.zeros(rgb.shape[:2], dtype=bool)
        global_mask[y0 : y0 + local_mask.shape[0], x0 : x0 + local_mask.shape[1]] = local_mask
        global_box = tuple(
            value + offset for value, offset in zip(local_box, (x0, y0, x0, y0), strict=True)
        )
        region_id = self.ws.state.next_id("region")
        self.ws.state.regions[region_id] = RegionEvidence(
            region_id=region_id,
            query=str(query),
            within_region_id=within_region_id,
            bbox_xyxy_px=global_box,
            source_revision=self.ws.state.observation_revision,
        )
        self.ws._private.region_masks[region_id] = global_mask
        geometry = self._build_region_geometry(
            str(query),
            global_mask,
            float(best.get("score", 0.0)),
        )
        if geometry is not None:
            self.ws._private.region_geometry[region_id] = geometry
            self.ws._private.trace_diagnostics["object_volume_proxy"] = {
                "region_id": region_id,
                "available": geometry.volume_proxy is not None,
                "extent_xyz_m": (
                    list(geometry.volume_proxy.extent_xyz_m)
                    if geometry.volume_proxy is not None
                    else None
                ),
            }
        return {
            "region_id": region_id,
            "bbox_xyxy_px": [round(float(value), 3) for value in global_box],
        }

    def _semantic_bbox(
        self,
        rgb: np.ndarray,
        query: str,
    ) -> tuple[float, float, float, float]:
        if not callable(getattr(self.ws.api, "query_vlm", None)):
            box = self.ws._call_backend(
                "vlm_bbox_detection",
                rgb,
                _semantic_grounding_query(query),
            )
            self.ws._private.trace_diagnostics["semantic_grounding"] = {
                "mode": "single_box_fallback",
            }
            return _bounded_box(box, rgb.shape[1], rgb.shape[0])

        coord_space = _grounding_coord_space(self.ws.api)
        diagnostics: dict[str, Any] = {
            "mode": "candidate_review",
            "coord_space": coord_space,
            "candidate_attempts": [],
            "review_attempts": [],
        }
        self.ws._private.trace_diagnostics["semantic_grounding"] = diagnostics

        candidate_query = candidate_prompt(
            query,
            width=rgb.shape[1],
            height=rgb.shape[0],
            coord_space=coord_space,
        )
        candidates = None
        candidate_error: ValueError | None = None
        for attempt in range(1, _SEMANTIC_QUERY_RETRIES + 2):
            candidate_reply = self.ws._call_backend(
                "query_vlm",
                candidate_query,
                images=rgb,
                temperature=0.0,
                max_tokens=512,
            )
            raw_reply = str(candidate_reply)
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "reply": raw_reply,
                "reply_length": len(raw_reply),
            }
            diagnostics["candidate_attempts"].append(attempt_record)
            diagnostics["candidate_reply"] = raw_reply
            try:
                candidates = parse_candidates(
                    raw_reply,
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    coord_space=coord_space,
                )
            except ValueError as exc:
                candidate_error = exc
                attempt_record["parse_error"] = str(exc)
                continue
            break
        if candidates is None:
            message = str(candidate_error or "unknown parse error")
            raise ContextFunctionError(f"semantic candidate parsing failed: {message}")
        diagnostics["candidate_boxes"] = [
            [round(float(value), 3) for value in item.box_xyxy_px] for item in candidates
        ]
        if not candidates:
            raise ContextFunctionError(f"semantic grounding found no candidate for '{query}'")

        review_raster = render_candidate_review(rgb, candidates)
        selected = None
        review_parsed = False
        review_error: ValueError | None = None
        for attempt in range(1, _SEMANTIC_QUERY_RETRIES + 2):
            review_reply = self.ws._call_backend(
                "query_vlm",
                review_prompt(query),
                images=review_raster,
                temperature=0.0,
                max_tokens=160,
            )
            raw_reply = str(review_reply)
            attempt_record = {
                "attempt": attempt,
                "reply": raw_reply,
                "reply_length": len(raw_reply),
            }
            diagnostics["review_attempts"].append(attempt_record)
            diagnostics["review_reply"] = raw_reply
            try:
                selected = parse_choice(raw_reply, count=len(candidates))
            except ValueError as exc:
                review_error = exc
                attempt_record["parse_error"] = str(exc)
                continue
            review_parsed = True
            break
        if not review_parsed:
            message = str(review_error or "unknown parse error")
            raise ContextFunctionError(f"semantic candidate review failed: {message}")
        diagnostics["selected_candidate"] = None if selected is None else int(selected + 1)
        if selected is None:
            raise ContextFunctionError(
                f"semantic grounding is ambiguous for '{query}'; refine the query"
            )
        return candidates[selected].box_xyxy_px

    def locate_point(
        self,
        query: str,
        within_region_id: str | None = None,
    ) -> dict[str, Any]:
        camera = self.ws._camera()
        rgb = _camera_image(camera, "rgb")
        crop_rgb, origin = self.ws._crop_rgb(rgb, within_region_id)
        local_point = _vector(
            self.ws._call_backend("vlm_point_detection", crop_rgb, str(query)),
            2,
            "VLM point",
        )
        if not (
            0.0 <= local_point[0] < crop_rgb.shape[1] and 0.0 <= local_point[1] < crop_rgb.shape[0]
        ):
            raise ContextFunctionError(
                f"VLM point {local_point.tolist()} lies outside its search image"
            )
        pixel = (float(local_point[0] + origin[0]), float(local_point[1] + origin[1]))
        try:
            depth_m = local_surface_depth(_camera_image(camera, "depth"), pixel)
            position = pixel_to_base(
                pixel,
                depth_m,
                camera["intrinsics"],
                camera["pose_mat"],
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            raise ContextFunctionError(str(exc)) from exc
        point_id = self.ws.state.next_id("point")
        self.ws.state.points[point_id] = PointEvidence(
            point_id=point_id,
            query=str(query),
            within_region_id=within_region_id,
            pixel_xy=pixel,
            position_xyz=tuple(float(value) for value in position),
            source_revision=self.ws.state.observation_revision,
        )
        return {
            "point_id": point_id,
            "pixel_xy": [round(value, 3) for value in pixel],
            "position_xyz": [round(float(value), 6) for value in position],
        }

    # ------------------------------------------------------------ proposal --
    def propose_grasps(self, region_id: str) -> dict[str, Any]:
        region = self.ws._current_region(region_id)
        camera = self.ws._camera()
        mask = self.ws._private.region_masks.get(region.region_id)
        if mask is None:
            raise ContextFunctionError(f"region '{region_id}' has no private mask")
        geometry = self.ws._private.region_geometry.get(region.region_id)
        planned = self.ws._call_backend(
            "plan_grasp",
            _camera_image(camera, "depth"),
            camera["intrinsics"],
            mask.astype(np.int64),
        )
        try:
            base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
        except (KeyError, ValueError) as exc:
            raise ContextFunctionError(f"invalid camera pose: {exc}") from exc
        object_points = (
            geometry.filtered_object_points_base if geometry is not None else None
        )
        prepared, diagnostic = _prepare_grasps(
            planned,
            poses_are_base_frame=False,
            base_from_camera=base_from_camera,
            reference_quaternion_xyzw=(
                self.ws.state.robot.ee_pose.quaternion_xyzw
                if self.ws.state.robot is not None and self.ws.state.robot.ee_pose is not None
                else None
            ),
            object_points_base=object_points,
        )
        geometric, geometric_diagnostic = _geometric_grasps(object_points)
        pool = [*geometric, *prepared]
        ranked = _select_grasp_mixture(
            pool, limit=max(len(pool), self.ws.MAX_GRASP_CANDIDATES)
        )
        reachable: list[tuple[_PreparedGrasp, ActionPrediction]] = []
        dropped: list[dict[str, Any]] = []
        for item in ranked:
            if len(reachable) >= self.ws.MAX_GRASP_CANDIDATES:
                break
            prediction = self._predict(item.target)
            if not self._seed_is_reachable(item.target, prediction):
                dropped.append(
                    {
                        "family": item.family,
                        "solve_ik": prediction.solve_ik,
                        "reason": (
                            "ik_error"
                            if prediction.solve_ik != "returned"
                            else "ik_mismatch"
                        ),
                        "target_xyz": _rounded_vector(item.target.position_xyz),
                    }
                )
                continue
            reachable.append((item, prediction))
        self.ws._private.trace_diagnostics["grasp_candidates"] = {
            "selected_source": "mixture" if reachable else None,
            "object_points": _point_summary(object_points),
            "attempts": [
                {"source": "single_view", **diagnostic},
                {"source": "geometric", **geometric_diagnostic},
            ],
            "dropped_unreachable": dropped,
            "selected": [
                {
                    "family": item.family,
                    "approach_z": round(item.approach_z, 6),
                    "target_xyz": _rounded_vector(item.target.position_xyz),
                }
                for item, _prediction in reachable
            ],
        }
        if not reachable:
            raise ContextFunctionError(
                f"grasp planner returned no kinematically reachable candidates for '{region_id}'"
            )

        ids: list[str] = []
        for item, prediction in reachable:
            seed_id = self.ws.state.next_id("s")
            target = ActionTarget(pose=item.target)
            self.ws.state.seeds[seed_id] = ActionSeed(
                seed_id=seed_id,
                target=target,
                source_revision=self.ws.state.observation_revision,
            )
            self.ws._private.seed_artifacts[seed_id] = SeedArtifacts(
                planning_context=PlanningContext(
                    source_kind="grasp", source_ref=seed_id, region_id=region_id
                ),
                preview_plan=MotionPlan(self.ws.motion_backend_name, prediction),
                family=item.family,
            )
            ids.append(seed_id)
        return {"seed_ids": ids}

    def _build_region_geometry(
        self,
        query: str,
        agentview_mask: np.ndarray,
        agentview_score: float,
    ) -> RegionGeometryArtifact | None:
        try:
            agent_camera = self.ws._camera()
            agent_points = _masked_points_base(agent_camera, agentview_mask)
            object_points = agent_points
            scene_parts = [_scene_points_base(agent_camera)]
        except (ContextFunctionError, KeyError, ValueError, np.linalg.LinAlgError):
            return None

        wrist_mask: np.ndarray | None = None
        try:
            wrist_camera = self.ws._private.camera(self.ws.wrist_camera_name)
            scene_parts.append(_scene_points_base(wrist_camera))
            wrist = self._segment_wrist_mask(wrist_camera, query)
            wrist_mask = wrist[0] if wrist is not None else None
            if wrist_mask is not None:
                wrist_points = _masked_points_base(wrist_camera, wrist_mask)
                object_points = _merge_multiview_object_points(
                    agent_points,
                    float(agentview_score),
                    wrist_points,
                    float(wrist[1]),
                )
        except (RuntimeError, ContextFunctionError, KeyError, ValueError, np.linalg.LinAlgError):
            wrist_mask = None

        scene_points = _stack_points(scene_parts)
        filtered = self._filter_object_points(object_points)
        if not _has_points(filtered):
            filtered = object_points
        volume_proxy = fit_gravity_stable_proxy(
            query,
            self.ws.state.observation_revision,
            filtered,
        )
        return RegionGeometryArtifact(
            agentview_mask=np.asarray(agentview_mask, dtype=bool).copy(),
            wrist_mask=None if wrist_mask is None else np.asarray(wrist_mask, dtype=bool).copy(),
            object_points_base=object_points,
            scene_points_base=scene_points,
            filtered_object_points_base=filtered,
            volume_proxy=volume_proxy,
        )

    def _segment_wrist_mask(
        self, wrist_camera: dict[str, Any], query: str
    ) -> tuple[np.ndarray, float] | None:
        segment = getattr(self.ws.api, "segment_sam3_text_prompt", None)
        if not callable(segment):
            return None
        rgb = _camera_image(wrist_camera, "rgb")
        try:
            results = segment(rgb, query)
        except Exception:
            return None
        if not isinstance(results, (list, tuple)):
            return None
        valid = [
            item for item in results if isinstance(item, dict) and item.get("mask") is not None
        ]
        if not valid:
            return None
        best = max(valid, key=lambda item: float(item.get("score", 0.0)))
        mask = np.asarray(best["mask"], dtype=bool)
        if mask.shape != rgb.shape[:2]:
            return None
        return mask, float(best.get("score", 0.0))

    def _filter_object_points(self, points: np.ndarray) -> np.ndarray:
        if not _has_points(points):
            return points
        filter_noise = getattr(self.ws.api, "filter_noise", None)
        if not callable(filter_noise):
            return points
        try:
            filtered, _ = filter_noise(points)
        except Exception:
            return points
        filtered_array = _points_array(filtered)
        return filtered_array if _has_points(filtered_array) else points

    def propose_pose(
        self,
        point_id: str,
        offset_xyz: list[float],
        quaternion_xyzw: list[float] | None = None,
    ) -> dict[str, Any]:
        point = self.ws._current_point(point_id)
        offset = _vector(offset_xyz, 3, "offset_xyz")
        if quaternion_xyzw is None:
            robot = self.ws.state.robot
            if robot is None or robot.ee_pose is None:
                raise ContextFunctionError("current EE orientation is unavailable")
            quaternion = robot.ee_pose.quaternion_xyzw
        else:
            quaternion = tuple(
                float(value) for value in _vector(quaternion_xyzw, 4, "quaternion_xyzw")
            )
        target = Pose(
            tuple(float(value) for value in np.asarray(point.position_xyz) + offset),
            quaternion,
        )
        return self._create_pending_action(
            ActionTarget(pose=target),
            ActionArtifacts(
                planning_context=PlanningContext(
                    source_kind="point",
                    source_ref=point_id,
                    region_id=point.within_region_id,
                ),
                motion_plan=self.ws.motion.plan_pose(target),
            ),
            intent=f"move to {point.query}",
        )

    def select(self, seed_id: str) -> dict[str, Any]:
        seed = self.ws._current_seed(seed_id)
        artifacts = self.ws._private.seed_artifacts.get(seed_id)
        if artifacts is None:
            raise ContextFunctionError(f"seed '{seed_id}' has no private artifacts")
        plan = self._plan_target(seed.target, artifacts.planning_context)
        region = (
            self.ws.state.regions.get(artifacts.planning_context.region_id)
            if artifacts.planning_context is not None
            and artifacts.planning_context.region_id is not None
            else None
        )
        return self._create_pending_action(
            seed.target,
            ActionArtifacts(
                planning_context=artifacts.planning_context,
                motion_plan=plan,
            ),
            intent=(
                f"approach {region.query} for grasp"
                if region is not None
                else "execute selected spatial approach"
            ),
        )

    def delta_move(
        self,
        delta_xyz_m: list[float],
        frame: str,
    ) -> dict[str, Any]:
        delta = _vector(delta_xyz_m, 3, "delta_xyz_m")
        if not np.any(delta != 0.0):
            raise ContextFunctionError("delta_xyz_m must contain a non-zero offset")
        if np.any(np.abs(delta) > self.MAX_DELTA_M):
            raise ContextFunctionError(
                "each delta_xyz_m component must be within [-0.03, 0.03] meters"
            )
        normalized_frame = _frame(frame)
        reference, artifacts = self._adjustment_reference()
        rotation = _pose_rotation(reference)
        displacement = delta if normalized_frame == "base" else rotation.apply(delta)
        target = Pose(
            tuple(float(value) for value in np.asarray(reference.position_xyz) + displacement),
            tuple(float(value) for value in rotation.as_quat()),
        )
        visual_edit = VisualEdit(
            kind="delta_move",
            frame=normalized_frame,
            reference_pose=reference,
            delta_xyz_m=tuple(float(value) for value in delta),
        )
        planning_context = artifacts.planning_context
        plan = self._plan_adjusted_pose(target, planning_context)
        return self._store_pending_action(
            ActionTarget(pose=target),
            ActionArtifacts(
                planning_context=planning_context,
                motion_plan=plan,
                initial_target=artifacts.initial_target,
                previous_visual_edit=artifacts.previous_visual_edit,
                latest_visual_edit=visual_edit,
                rotation_gizmo_frame=artifacts.rotation_gizmo_frame,
                rotation_gizmo_axis=artifacts.rotation_gizmo_axis,
                turn_count=artifacts.turn_count,
            ),
        )

    def rotate(
        self,
        axis: str,
        angle_deg: float,
        frame: str,
    ) -> dict[str, Any]:
        normalized_axis = str(axis)
        if normalized_axis not in {"x", "y", "z"}:
            raise ContextFunctionError("axis must be one of 'x', 'y', or 'z'")
        try:
            angle = float(angle_deg)
        except (TypeError, ValueError) as exc:
            raise ContextFunctionError("angle_deg must be a finite number") from exc
        if not np.isfinite(angle):
            raise ContextFunctionError("angle_deg must be a finite number")
        if angle == 0.0:
            raise ContextFunctionError("angle_deg must be non-zero")
        if abs(angle) > self.MAX_ROTATION_DEG:
            raise ContextFunctionError("angle_deg must be within [-10, 10] degrees")
        normalized_frame = _frame(frame)
        reference, artifacts = self._adjustment_reference()
        reference_rotation = _pose_rotation(reference)
        axis_vector = np.eye(3, dtype=np.float64)["xyz".index(normalized_axis)]
        delta_rotation = Rotation.from_rotvec(axis_vector * np.deg2rad(angle))
        target_rotation = (
            delta_rotation * reference_rotation
            if normalized_frame == "base"
            else reference_rotation * delta_rotation
        )
        target = Pose(
            reference.position_xyz,
            tuple(float(value) for value in target_rotation.as_quat()),
        )
        visual_edit = VisualEdit(
            kind="rotate",
            frame=normalized_frame,
            reference_pose=reference,
            axis=normalized_axis,
            angle_deg=angle,
        )
        planning_context = artifacts.planning_context
        plan = self._plan_adjusted_pose(target, planning_context)
        return self._store_pending_action(
            ActionTarget(pose=target),
            ActionArtifacts(
                planning_context=planning_context,
                motion_plan=plan,
                initial_target=artifacts.initial_target,
                previous_visual_edit=artifacts.previous_visual_edit,
                latest_visual_edit=visual_edit,
                rotation_gizmo_frame=artifacts.rotation_gizmo_frame,
                rotation_gizmo_axis=artifacts.rotation_gizmo_axis,
                turn_count=artifacts.turn_count,
            ),
        )

    def _adjustment_reference(self) -> tuple[Pose, ActionArtifacts]:
        action = self.ws.state.pending_action
        if action is not None:
            reference_rotation = _pose_rotation(action.target.pose)
            reference = Pose(
                action.target.pose.position_xyz,
                tuple(float(value) for value in reference_rotation.as_quat()),
            )
            return reference, self.ws._private.action_artifacts or ActionArtifacts()

        robot = self.ws.state.robot
        if robot is None or robot.tcp_pose is None:
            raise ContextFunctionError("current TCP pose is unavailable")
        try:
            rotation = _pose_rotation(robot.tcp_pose)
        except ValueError as exc:
            raise ContextFunctionError(f"current TCP pose is invalid: {exc}") from exc
        reference = Pose(
            robot.tcp_pose.position_xyz,
            tuple(float(value) for value in rotation.as_quat()),
        )
        return (
            reference,
            ActionArtifacts(initial_target=ActionTarget(pose=reference)),
        )

    def _plan_adjusted_pose(
        self,
        target: Pose,
        planning_context: PlanningContext | None,
    ) -> MotionPlan:
        backend, route = self.ws.motion_backend_for_target(target)
        self.ws._private.trace_diagnostics["motion_route"] = route
        if planning_context is not None and planning_context.region_id is not None:
            region = self.ws.state.regions.get(planning_context.region_id)
            mask = self.ws._private.region_masks.get(planning_context.region_id)
            if region is None or mask is None:
                raise ContextFunctionError(
                    f"grasp source '{planning_context.source_ref}' is unavailable"
                )
            return backend.plan_grasp(
                target,
                object_name=region.query,
                object_mask=mask,
            )
        return backend.plan_pose(target)

    def _create_pending_action(
        self,
        target: ActionTarget,
        artifacts: ActionArtifacts,
        *,
        intent: str,
    ) -> dict[str, Any]:
        action_id = self.ws.state.next_id("a")
        artifacts = ActionArtifacts(
            planning_context=artifacts.planning_context,
            motion_plan=artifacts.motion_plan,
            initial_target=target,
            previous_visual_edit=None,
            latest_visual_edit=artifacts.latest_visual_edit,
            rotation_gizmo_frame=None,
            rotation_gizmo_axis=None,
            turn_count=0,
        )
        self.ws.state.pending_action = PendingAction(action_id, target, intent)
        self.ws._private.action_artifacts = artifacts
        self.ws._private.clear_contact_camera_lock()
        return _preview_result(action_id, target, artifacts)

    def _store_pending_action(
        self, target: ActionTarget, artifacts: ActionArtifacts
    ) -> dict[str, Any]:
        action = self.ws.state.pending_action
        if action is None:
            raise ContextFunctionError("there is no pending action to edit")
        before_target = (
            action.target.summary()
        )
        previous = self.ws._private.action_artifacts
        turn_count = previous.turn_count if previous is not None else 0
        initial_target = (
            previous.initial_target
            if previous is not None and previous.initial_target is not None
            else artifacts.initial_target or target
        )
        latest_edit = artifacts.latest_visual_edit
        previous_edit = artifacts.previous_visual_edit
        if previous is not None:
            if latest_edit is not None:
                previous_edit = previous.latest_visual_edit
            else:
                latest_edit = previous.latest_visual_edit
                previous_edit = previous.previous_visual_edit
        rotation_gizmo_frame = (
            artifacts.rotation_gizmo_frame
            if artifacts.rotation_gizmo_frame is not None
            else previous.rotation_gizmo_frame
            if previous is not None
            else None
        )
        rotation_gizmo_axis = (
            artifacts.rotation_gizmo_axis
            if artifacts.rotation_gizmo_axis is not None
            else previous.rotation_gizmo_axis
            if previous is not None
            else None
        )
        self.ws.state.pending_action = PendingAction(
            action.action_id,
            target,
            action.intent,
            ready_for_commit=False,
        )
        self.ws._private.action_artifacts = ActionArtifacts(
            planning_context=artifacts.planning_context,
            motion_plan=artifacts.motion_plan,
            initial_target=initial_target,
            previous_visual_edit=previous_edit,
            latest_visual_edit=latest_edit,
            rotation_gizmo_frame=rotation_gizmo_frame,
            rotation_gizmo_axis=rotation_gizmo_axis,
            turn_count=turn_count,
        )
        self.ws._private.trace_diagnostics["imagination_edit"] = {
            "target_before": before_target,
            "target_after": target.summary(),
            "refinement_instruction": (
                self.ws.state.refinement.instruction
                if self.ws.state.refinement is not None
                else None
            ),
            "turn_count": turn_count,
            "latest_visual_edit": (
                artifacts.latest_visual_edit.summary()
                if artifacts.latest_visual_edit is not None
                else None
            ),
            "planning_source": (
                {
                    "kind": artifacts.planning_context.source_kind,
                    "ref": artifacts.planning_context.source_ref,
                }
                if artifacts.planning_context is not None
                else None
            ),
        }
        return _preview_result(action.action_id, target, self.ws._private.action_artifacts)

    def _plan_target(
        self, target: ActionTarget, context: PlanningContext | None
    ) -> MotionPlan | None:
        if context is not None and context.region_id is not None:
            region = self.ws.state.regions.get(context.region_id)
            mask = self.ws._private.region_masks.get(context.region_id)
            if region is None or mask is None:
                raise ContextFunctionError(
                    f"grasp source '{context.source_ref}' is unavailable"
                )
            return self.ws.motion.plan_grasp(
                target.pose,
                object_name=region.query,
                object_mask=mask,
            )
        return self.ws.motion.plan_pose(target.pose)

    def _predict(self, target: Pose) -> ActionPrediction:
        return self.ws.motion.preview(target)

    def _seed_is_reachable(self, target: Pose, prediction: ActionPrediction) -> bool:
        """Keep only seeds whose IK joints actually realise the shown TCP."""

        if prediction.solve_ik != "returned" or prediction.joint_positions_rad is None:
            return False
        override = getattr(self.ws.api, "seed_ik_matches_target", None)
        if override is True:
            return True
        if callable(override):
            return bool(override(target, prediction))
        from vaw.context_runtime.gripper_mesh import load_panda_urdf_fk
        from vaw.context_runtime.packet import _candidate_fk_matches_target

        if load_panda_urdf_fk() is None:
            return True
        return _candidate_fk_matches_target(
            prediction.joint_positions_rad,
            target,
            self.ws._tcp_to_hand_local_xyz,
        )

    # ------------------------------------------------------------- physical --
    def direct_delta_move(
        self,
        delta_xyz_m: list[float],
        frame: str,
    ) -> dict[str, Any]:
        """Execute one small Main-owned TCP translation immediately."""

        delta = _vector(delta_xyz_m, 3, "delta_xyz_m")
        if not np.any(delta != 0.0):
            raise ContextFunctionError("delta_xyz_m must contain a non-zero offset")
        if np.any(np.abs(delta) > self.MAX_DELTA_M):
            raise ContextFunctionError(
                "each delta_xyz_m component must be within [-0.03, 0.03] meters"
            )
        normalized_frame = _frame(frame)
        robot = self.ws.state.robot
        if robot is None or robot.tcp_pose is None:
            raise ContextFunctionError("current TCP pose is unavailable")
        reference = robot.tcp_pose
        rotation = _pose_rotation(reference)
        displacement = delta if normalized_frame == "base" else rotation.apply(delta)
        displacement_base = tuple(float(value) for value in displacement)
        target = Pose(
            tuple(
                float(value)
                for value in np.asarray(reference.position_xyz, dtype=np.float64)
                + displacement
            ),
            tuple(float(value) for value in rotation.as_quat()),
        )
        # Planning is not a physical event.  A failure before ``execute`` must
        # not refresh the observation or enter TaskMemory.
        try:
            plan = self.ws.local_motion.plan_pose(target)
        except MotionBackendError as exc:
            raise ContextFunctionError(str(exc)) from exc
        if plan.prediction.solve_ik != "returned":
            raise ContextFunctionError(
                plan.prediction.detail or "direct delta_move has no executable plan"
            )

        previous_physical = self.ws._private.last_physical_artifacts
        execution_error: ContextFunctionError | None = None
        try:
            self.ws.execute_motion_plan(plan, target)
        except MotionBackendError as exc:
            execution_error = ContextFunctionError(str(exc))
        finally:
            self.ws.refresh_observation()

        self.ws.state.last_physical_action = LastPhysicalAction(
            intent=f"direct delta_move in {normalized_frame} frame",
            executed_stages="arm",
            outcome="completed" if execution_error is None else "arm_failed",
            requested_arm_delta_base_m=displacement_base,
            error_detail=str(execution_error) if execution_error is not None else None,
            evidence_invalidated=True,
            recovery_hint=(
                _COMMIT_FAILURE_RECOVERY_HINT if execution_error is not None else None
            ),
        )
        self.ws._private.last_physical_artifacts = LastPhysicalArtifacts(
            focus_pose=target,
            subject_query=(
                previous_physical.subject_query
                if previous_physical is not None
                else None
            ),
            subject_points_base=(
                previous_physical.subject_points_base
                if previous_physical is not None
                else None
            ),
        )
        self.ws._private.trace_diagnostics["direct_delta_move"] = {
            "motion_backend": plan.backend,
            "frame": normalized_frame,
            "delta_xyz_m": [float(value) for value in delta],
            "delta_base_m": list(displacement_base),
        }
        if execution_error is not None:
            raise execution_error
        achieved = self.ws.state.robot.tcp_pose if self.ws.state.robot is not None else None
        result: dict[str, Any] = {}
        if achieved is not None:
            result["position_error_m"] = round(
                float(
                    np.linalg.norm(
                        np.asarray(achieved.position_xyz, dtype=np.float64)
                        - np.asarray(target.position_xyz, dtype=np.float64)
                    )
                ),
                6,
            )
        return result

    def commit(self, action_id: str) -> dict[str, Any]:
        action = self.ws.state.pending_action
        if action is None or action.action_id != action_id:
            raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
        if not action.ready_for_commit:
            raise ContextFunctionError(
                f"action '{action_id}' is a coarse proposal; call refine_action before commit"
            )
        start_tcp = self.ws.state.robot.tcp_pose if self.ws.state.robot is not None else None
        requested_arm_delta = (
            tuple(
                float(value)
                for value in (
                    np.asarray(action.target.pose.position_xyz, dtype=np.float64)
                    - np.asarray(start_tcp.position_xyz, dtype=np.float64)
                )
            )
            if start_tcp is not None
            else None
        )
        focus_pose = action.target.pose
        artifacts = self.ws._private.action_artifacts
        # Missing private artifacts or a missing plan are pre-dispatch
        # failures.  Keep the current observation/revision intact and expose
        # the problem only as the current FunctionEvent.
        if artifacts is None:
            raise ContextFunctionError(
                f"active action '{action_id}' has no private artifacts"
            )
        if artifacts.motion_plan is None:
            raise ContextFunctionError(
                f"active action '{action_id}' has no cached motion plan"
            )

        execution_error: ContextFunctionError | None = None
        failed_stage: str | None = "arm"
        executed_stages: list[str] = ["arm"]
        previous_physical = self.ws._private.last_physical_artifacts
        subject_query = previous_physical.subject_query if previous_physical is not None else None
        subject_points = (
            previous_physical.subject_points_base
            if previous_physical is not None
            else None
        )
        planning_context = artifacts.planning_context if artifacts is not None else None
        object_proxy = None
        if (
            planning_context is not None
            and planning_context.source_kind == "grasp"
            and planning_context.region_id is not None
        ):
            source = self.ws.state.regions.get(planning_context.region_id)
            if source is not None:
                subject_query = source.query
            geometry = self.ws._private.region_geometry.get(planning_context.region_id)
            if geometry is not None:
                object_proxy = geometry.volume_proxy
                subject_points = geometry.filtered_object_points_base
        elif (
            planning_context is not None
            and planning_context.source_kind == "point"
            and planning_context.source_ref is not None
        ):
            # Preserve the destination region for post-commit Contact views.
            # The point itself is only a coarse anchor and has no extent; its
            # parent region provides the opening/boundary geometry needed to
            # frame placement corrections and release.
            point = self.ws.state.points.get(planning_context.source_ref)
            region = (
                self.ws.state.regions.get(point.within_region_id)
                if point is not None and point.within_region_id is not None
                else None
            )
            if region is not None:
                subject_query = region.query
                geometry = self.ws._private.region_geometry.get(region.region_id)
                if geometry is not None:
                    subject_points = geometry.filtered_object_points_base
                    object_proxy = geometry.volume_proxy
        try:
            self.ws.execute_motion_plan(artifacts.motion_plan, action.target.pose)
            failed_stage = None
        except MotionBackendError as exc:
            execution_error = ContextFunctionError(str(exc))
        finally:
            self.ws.refresh_observation()

        stage_summary = "arm"
        if execution_error is None:
            outcome = "completed"
        elif failed_stage == "arm":
            outcome = "arm_failed"
        else:
            outcome = "gripper_failed"
        self.ws.state.last_physical_action = LastPhysicalAction(
            intent=action.intent,
            executed_stages=stage_summary,
            outcome=outcome,
            requested_arm_delta_base_m=requested_arm_delta,
            failed_action_id=action_id if execution_error is not None else None,
            error_detail=str(execution_error) if execution_error is not None else None,
            source_query=subject_query,
            evidence_invalidated=True,
            recovery_hint=_COMMIT_FAILURE_RECOVERY_HINT if execution_error is not None else None,
        )
        self.ws._private.last_physical_artifacts = LastPhysicalArtifacts(
            focus_pose=focus_pose,
            subject_query=subject_query,
            subject_points_base=subject_points,
        )
        can_promote_proxy = (
            execution_error is None
            and object_proxy is not None
            and self.ws._private.attachment_hypothesis is None
        )
        self.ws._private.object_proxy_candidate = (
            ObjectProxyCandidate(action_id=action_id, proxy=object_proxy)
            if can_promote_proxy
            else None
        )
        if (
            execution_error is None
            and planning_context is not None
            and planning_context.source_kind == "grasp"
        ):
            self.ws._private.attachment_hypothesis = None

        self.ws._private.trace_diagnostics["waypoint_commit"] = {
            "action_id": action_id,
            "motion_backend": artifacts.motion_plan.backend,
            "executed_stages": list(executed_stages),
            "failed_stage": failed_stage,
            "object_proxy_promoted": (
                self.ws._private.object_proxy_candidate is not None
                and self.ws._private.object_proxy_candidate.action_id == action_id
            ),
        }

        achieved = self.ws.state.robot.ee_pose if self.ws.state.robot is not None else None
        position_error = None
        if achieved is not None:
            achieved_tcp_position = tcp_position_from_hand_pose(
                achieved.position_xyz,
                achieved.quaternion_xyzw,
                self.ws._tcp_to_hand_local_xyz,
            )
            position_error = float(
                np.linalg.norm(achieved_tcp_position - np.asarray(action.target.pose.position_xyz))
            )
        if execution_error is not None:
            raise execution_error
        result: dict[str, Any] = {}
        if position_error is not None:
            result["position_error_m"] = round(position_error, 6)
        return result

    def open_gripper(self) -> dict[str, Any]:
        return self._execute_gripper("open")

    def close_gripper(self) -> dict[str, Any]:
        return self._execute_gripper("closed")

    def show_rotation_gizmo(self, frame: str, axis: str) -> dict[str, Any]:
        """Request a non-occluding +/- guide for one rotation axis."""

        normalized_frame = _frame(frame)
        normalized_axis = str(axis).lower()
        if normalized_axis not in {"x", "y", "z"}:
            raise ContextFunctionError("axis must be one of 'x', 'y', or 'z'")
        action = self.ws.state.pending_action
        if action is None:
            raise ContextFunctionError("there is no pending action")
        artifacts = self.ws._private.action_artifacts or ActionArtifacts()
        self.ws._private.action_artifacts = replace(
            artifacts,
            rotation_gizmo_frame=normalized_frame,
            rotation_gizmo_axis=normalized_axis,
        )
        self.ws._private.trace_diagnostics["rotation_gizmo"] = {
            "frame": normalized_frame,
            "axis": normalized_axis,
            "target": action.target.summary(),
        }
        return {
            "preview": "updated",
            "rotation_guide": {
                "frame": normalized_frame,
                "axis": normalized_axis,
            },
        }

    def _execute_gripper(self, target: str) -> dict[str, Any]:
        """Execute a simple Main-owned gripper command immediately."""

        normalized_target = _gripper_target(target)
        backend_function = (
            "open_gripper" if normalized_target == "open" else "close_gripper"
        )
        focus_pose = self.ws.state.robot.tcp_pose if self.ws.state.robot is not None else None
        previous_physical = self.ws._private.last_physical_artifacts
        candidate = self.ws._private.object_proxy_candidate
        execution_error: ContextFunctionError | None = None
        try:
            self.ws._call_backend(backend_function)
        except ContextFunctionError as exc:
            execution_error = exc
        finally:
            self.ws.refresh_observation()
        self.ws.state.last_physical_action = LastPhysicalAction(
            intent=f"direct {backend_function}",
            executed_stages="gripper",
            outcome="completed" if execution_error is None else "gripper_failed",
            target_gripper=normalized_target,
            error_detail=str(execution_error) if execution_error is not None else None,
            evidence_invalidated=True,
        )
        self.ws._private.last_physical_artifacts = LastPhysicalArtifacts(
            focus_pose=focus_pose,
            subject_query=(
                previous_physical.subject_query
                if previous_physical is not None
                else None
            ),
            subject_points_base=(
                previous_physical.subject_points_base
                if previous_physical is not None
                else None
            ),
        )
        if execution_error is None and normalized_target == "open":
            self.ws._private.object_proxy_candidate = None
            self.ws._private.attachment_hypothesis = None
        elif execution_error is None and normalized_target == "closed" and candidate is not None:
            observed_tcp = (
                self.ws.state.robot.tcp_pose if self.ws.state.robot is not None else None
            )
            if observed_tcp is not None:
                self.ws._private.attachment_hypothesis = bind_proxy_to_tcp(
                    candidate,
                    observed_tcp,
                )
                self.ws._private.object_proxy_candidate = None
        self.ws._private.trace_diagnostics["attachment_preview"] = {
            "command": normalized_target,
            "candidate_available_before": candidate is not None,
            "hypothesis_available_after": (
                self.ws._private.attachment_hypothesis is not None
            ),
        }
        if execution_error is not None:
            raise execution_error
        opening = self.ws.state.robot.gripper_opening if self.ws.state.robot is not None else None
        return (
            {"gripper_opening": round(float(opening), 6)}
            if opening is not None
            else {}
        )

    def finish_imagination(self, status: str) -> dict[str, Any]:
        if status not in {"ready", "failed"}:
            raise ContextFunctionError("status must be 'ready' or 'failed'")
        if status == "failed":
            return self._fail_imagination("agent_failed")
        return self._finish_refinement("agent_ready")

    def reject_action(self, action_id: str) -> dict[str, Any]:
        """Decline a reviewed virtual target without changing the world."""

        action = self.ws.state.pending_action
        if action is None or action.action_id != action_id:
            raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
        self.ws.discard_pending_action()
        return {
            "declined_action_id": action_id,
            "declined_how": "explicit",
            "declined_intent": action.intent,
        }

    def limit_imagination(self) -> dict[str, Any]:
        """Fail a refinement session that did not converge within its budget."""

        return self._fail_imagination("turn_limit")

    def _fail_imagination(self, termination_reason: str) -> dict[str, Any]:
        session = self.ws.state.refinement
        if session is None:
            raise ContextFunctionError("there is no active refinement session")
        source_ref = _planning_source_ref(self.ws._private.action_artifacts)
        failed_action_id = session.action_id
        # A failed refinement is not a deliverable action.  Keeping the coarse
        # target live caused Main to re-dispatch the same impossible local task
        # indefinitely.  Seeds/evidence remain valid so Main can choose a new
        # source, but the failed action itself is retired.
        self.ws.discard_pending_action()
        self.ws._private.trace_diagnostics["imagination_handoff"] = {
            "status": "failed",
            "termination_reason": termination_reason,
            "source_ref": source_ref,
            "failed_action_id": failed_action_id,
        }
        return {"status": "failed", "source_ref": source_ref}

    def _finish_refinement(self, termination_reason: str) -> dict[str, Any]:
        session = self.ws.state.refinement
        action = self.ws.state.pending_action
        if session is None or action is None:
            raise ContextFunctionError("there is no active refinement session")
        artifacts = self.ws._private.action_artifacts or ActionArtifacts()
        plan = artifacts.motion_plan
        if plan is None or plan.prediction.solve_ik != "returned":
            return self._fail_imagination(
                f"{termination_reason}_without_executable_plan"
            )
        self.ws.state.pending_action = replace(action, ready_for_commit=True)
        self.ws.state.refinement = None
        self.ws._private.trace_diagnostics["imagination_handoff"] = {
            "status": "ready",
            "termination_reason": termination_reason,
            "action_id": action.action_id,
            "source_ref": _planning_source_ref(artifacts),
            "target": action.target.summary(),
            "intent": action.intent,
        }
        return {"status": "ready", "action_id": action.action_id}

    def done(self, success: bool) -> dict[str, Any]:
        self.ws.finished = True
        self.ws.claimed_success = bool(success)
        return {}


def _gripper_target(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in {"open", "closed"}:
        raise ValueError(f"invalid private gripper target {value!r}")
    return value


def _planning_source_ref(artifacts: ActionArtifacts | None) -> str | None:
    if artifacts is None or artifacts.planning_context is None:
        return None
    return artifacts.planning_context.source_ref


def _preview_result(
    action_id: str,
    target: ActionTarget,
    artifacts: ActionArtifacts,
) -> dict[str, Any]:
    result: dict[str, Any] = {"action_id": action_id, "preview": "updated"}
    plan = artifacts.motion_plan
    if plan is not None:
        prediction = plan.prediction
        result.update(
            {
                "solve_ik": prediction.solve_ik,
                "trajectory_checked": prediction.trajectory_checked,
                "collision_checked": prediction.collision_checked,
            }
        )
    return result


def _camera_image(camera: dict[str, Any], name: str) -> np.ndarray:
    try:
        image = np.asarray(camera["images"][name])
    except (KeyError, TypeError) as exc:
        raise ContextFunctionError(f"camera has no {name} image") from exc
    return image


def _camera_depth(camera: dict[str, Any]) -> np.ndarray:
    depth = _camera_image(camera, "depth")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ContextFunctionError(f"camera depth must be 2D, got {depth.shape}")
    return np.asarray(depth, dtype=np.float64)


def _camera_pose(camera: dict[str, Any]) -> np.ndarray:
    try:
        return np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextFunctionError(f"invalid camera pose: {exc}") from exc


def _camera_intrinsics(camera: dict[str, Any]) -> np.ndarray:
    try:
        return np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextFunctionError(f"invalid camera intrinsics: {exc}") from exc


def _scene_points_base(camera: dict[str, Any]) -> np.ndarray:
    points_camera = depth_to_pointcloud(
        _camera_depth(camera),
        _camera_intrinsics(camera),
        subsample_factor=1,
        filter_invalid=True,
    )
    return _transform_points(_points_array(points_camera), _camera_pose(camera))


def _masked_points_base(camera: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    depth = _camera_depth(camera)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != depth.shape:
        raise ContextFunctionError(
            f"mask shape {mask_array.shape} does not match depth {depth.shape}"
        )
    points_camera = depth_to_pointcloud(
        depth,
        _camera_intrinsics(camera),
        subsample_factor=1,
        filter_invalid=False,
    )
    points = _points_array(points_camera)
    flat_mask = mask_array.reshape(-1)
    if flat_mask.shape[0] != points.shape[0]:
        raise ContextFunctionError(
            f"mask has {flat_mask.shape[0]} pixels but point cloud has {points.shape[0]} points"
        )
    valid = (
        flat_mask
        & np.isfinite(points).all(axis=1)
        & (points[:, 2] >= 0.015)
        & (points[:, 2] <= 20.0)
    )
    return _transform_points(points[valid], _camera_pose(camera))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = _points_array(points)
    if points.shape[0] == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    return (transform @ homogeneous.T).T[:, :3]


def _stack_points(parts: list[np.ndarray]) -> np.ndarray:
    arrays = [_points_array(part) for part in parts if _has_points(part)]
    if not arrays:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(arrays, axis=0)


def _merge_multiview_object_points(
    agent_points: np.ndarray,
    agent_score: float,
    wrist_points: np.ndarray,
    wrist_score: float,
) -> np.ndarray:
    agent = _points_array(agent_points)
    wrist = _points_array(wrist_points)
    if agent.shape[0] == 0:
        return wrist
    if wrist.shape[0] == 0:
        return agent
    # Match CaP-X Full API's conservative merge: combine views only when the
    # two segmented clouds agree spatially, otherwise trust the higher-score
    # segmentation instead of mixing likely-wrong object points.
    agent_probe = _deterministic_point_sample(agent, 3000)
    wrist_probe = _deterministic_point_sample(wrist, 3000)
    distances = np.linalg.norm(
        agent_probe[:, np.newaxis, :] - wrist_probe[np.newaxis, :, :], axis=2
    )
    if float(np.min(distances)) < 0.01:
        return np.concatenate([agent, wrist], axis=0)
    return wrist if wrist_score > agent_score else agent


def _deterministic_point_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if points.shape[0] <= limit:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=limit, dtype=np.int64)
    return points[indices]


def _points_array(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return array.reshape(-1, 3)


def _has_points(points: Any) -> bool:
    try:
        return _points_array(points).shape[0] > 0
    except (TypeError, ValueError):
        return False


def _source_extent(
    object_points: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    if object_points.shape[0] == 0:
        return None, None
    lower, upper = np.quantile(object_points, [0.02, 0.98], axis=0)
    source_center = (lower + upper) / 2.0
    source_radius_m = max(0.04, float(np.linalg.norm(upper - lower)))
    return source_center, source_radius_m


def _approach_vector(item: _PreparedGrasp) -> np.ndarray:
    return Rotation.from_quat(np.asarray(item.target.quaternion_xyzw)).as_matrix()[:, 2]


def _near_duplicate_approach(
    item: _PreparedGrasp,
    chosen: list[_PreparedGrasp],
    *,
    threshold: float = 0.97,
) -> bool:
    if not chosen:
        return False
    approach = _approach_vector(item)
    return any(
        float(np.dot(approach, _approach_vector(existing))) >= threshold
        for existing in chosen
    )


def _prepared_from_contact(
    position_xyz: np.ndarray,
    quaternion_xyzw: tuple[float, float, float, float],
    *,
    family: str,
    score: float,
    raw_index: int,
    source_center: np.ndarray | None,
    source_radius_m: float | None,
) -> _PreparedGrasp | None:
    rotation = Rotation.from_quat(np.asarray(quaternion_xyzw)).as_matrix()
    if not np.isfinite(position_xyz).all() or not np.isfinite(quaternion_xyzw).all():
        return None
    approach_z = float(rotation[2, 2])
    if approach_z > 0.0:
        return None
    contact = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    if (
        source_center is not None
        and source_radius_m is not None
        and float(np.linalg.norm(contact - source_center)) > source_radius_m
    ):
        return None
    return _PreparedGrasp(
        target=Pose(
            tuple(float(value) for value in contact),
            quaternion_xyzw,
        ),
        score=score,
        raw_index=raw_index,
        approach_z=approach_z,
        family=family,
    )


def _side_quaternion_xyzw(approach_axis: np.ndarray) -> tuple[float, float, float, float] | None:
    z_axis = np.asarray(approach_axis, dtype=np.float64)
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-8:
        return None
    z_axis = z_axis / norm
    y_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-8:
        return None
    y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    quaternion = Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()
    return tuple(float(value) for value in quaternion)


def _geometric_grasps(
    object_points_base: np.ndarray | None,
) -> tuple[list[_PreparedGrasp], dict[str, Any]]:
    object_points = (
        _points_array(object_points_base)
        if object_points_base is not None and _has_points(object_points_base)
        else np.empty((0, 3), dtype=np.float64)
    )
    source_center, source_radius_m = _source_extent(object_points)
    prepared: list[_PreparedGrasp] = []
    rejected: list[dict[str, Any]] = []
    if object_points.shape[0] < 20:
        return prepared, {"raw_count": 0, "accepted_count": 0, "rejected": rejected}

    center = np.median(object_points, axis=0)
    top_z = float(np.percentile(object_points[:, 2], 95))
    top = _prepared_from_contact(
        np.array([center[0], center[1], top_z - _TOP_GRASP_OFFSET_M]),
        _LIBERO_TOP_QUAT_XYZW,
        family="top",
        score=0.6,
        raw_index=0,
        source_center=source_center,
        source_radius_m=source_radius_m,
    )
    if top is None:
        rejected.append({"index": 0, "family": "top", "reasons": ["filtered"]})
    else:
        prepared.append(top)

    covariance = np.cov((object_points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    body_axis = eigenvectors[:, int(np.argmax(eigenvalues))].copy()
    body_axis[2] = 0.0
    if float(np.linalg.norm(body_axis)) < 1e-8:
        body_axis = np.array([1.0, 0.0, 0.0])
    body_axis = body_axis / np.linalg.norm(body_axis)
    side_axis = np.array([-body_axis[1], body_axis[0], 0.0])
    grasp_z = float(
        np.clip(center[2], np.percentile(object_points[:, 2], 30), np.percentile(object_points[:, 2], 70))
    )
    grasp = np.array([center[0], center[1], grasp_z])
    for offset, sign in enumerate((1.0, -1.0), start=1):
        quaternion = _side_quaternion_xyzw(sign * side_axis)
        if quaternion is None:
            rejected.append({"index": offset, "family": "pca", "reasons": ["invalid_rotation"]})
            continue
        item = _prepared_from_contact(
            grasp,
            quaternion,
            family="pca",
            score=0.55,
            raw_index=offset,
            source_center=source_center,
            source_radius_m=source_radius_m,
        )
        if item is None:
            rejected.append({"index": offset, "family": "pca", "reasons": ["filtered"]})
        else:
            prepared.append(item)
    return prepared, {
        "raw_count": 3,
        "accepted_count": len(prepared),
        "rejected": rejected,
    }


def _select_grasp_mixture(
    prepared: list[_PreparedGrasp],
    *,
    limit: int,
) -> list[_PreparedGrasp]:
    chosen: list[_PreparedGrasp] = []
    quotas = {"top": _MAX_TOP_GRASPS, "pca": _MAX_PCA_GRASPS, "cgn": _MAX_CGN_GRASPS}
    for family, quota in quotas.items():
        pool = [item for item in prepared if item.family == family]
        pool.sort(key=lambda item: item.score, reverse=True)
        taken = 0
        for item in pool:
            if taken >= quota or len(chosen) >= limit:
                break
            if family == "cgn" and _near_duplicate_approach(item, chosen):
                continue
            chosen.append(item)
            taken += 1
    leftover = [item for item in prepared if item not in chosen]
    leftover.sort(key=lambda item: _min_approach_dot(item, chosen))
    for item in leftover:
        if len(chosen) >= limit:
            break
        if _near_duplicate_approach(item, chosen, threshold=0.995):
            continue
        chosen.append(item)
    return chosen[:limit]


def _min_approach_dot(item: _PreparedGrasp, chosen: list[_PreparedGrasp]) -> float:
    if not chosen:
        return 1.0
    approach = _approach_vector(item)
    return min(float(np.dot(approach, _approach_vector(existing))) for existing in chosen)


def _prepare_grasps(
    planned: Any,
    *,
    poses_are_base_frame: bool,
    base_from_camera: np.ndarray | None,
    reference_quaternion_xyzw: tuple[float, float, float, float] | None = None,
    object_points_base: np.ndarray | None,
) -> tuple[list[_PreparedGrasp], dict[str, Any]]:
    """Convert planner output without inventing a different approach pose.

    Contact-GraspNet's local +Z is the hand-to-contact direction used by the
    Franka TCP offset.  A positive base-Z component would therefore place the
    hand below a tabletop contact.  Such candidates are rejected so the caller
    can fall back to the independent single-view planner; they are never
    repaired by rotating the gripper 180 degrees around the contact point.

    A candidate must also remain spatially consistent with the segmented
    source surface.  The admissible radius is derived from that surface's own
    robust 3-D extent rather than from a task or object-class threshold.  This
    catches occasional planner/frame outliers without ranking otherwise valid
    candidates or hard-coding a pick strategy.
    """

    if not isinstance(planned, tuple) or len(planned) != 2:
        raise ContextFunctionError("grasp planner did not return poses and scores")
    poses, scores = planned
    try:
        scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        poses_array = np.asarray(poses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContextFunctionError(f"invalid grasp planner output: {exc}") from exc
    if scores_array.size == 0:
        return [], {"raw_count": 0, "accepted_count": 0, "rejected": []}
    if poses_array.shape != (scores_array.size, 4, 4):
        raise ContextFunctionError(
            f"grasp poses {poses_array.shape} do not match {scores_array.size} scores"
        )
    if not poses_are_base_frame and base_from_camera is None:
        raise ContextFunctionError("camera-frame grasps require a camera pose")

    object_points = (
        _points_array(object_points_base)
        if object_points_base is not None and _has_points(object_points_base)
        else np.empty((0, 3), dtype=np.float64)
    )
    source_center, source_radius_m = _source_extent(object_points)
    prepared: list[_PreparedGrasp] = []
    rejected: list[dict[str, Any]] = []
    for index, (pose, score) in enumerate(zip(poses_array, scores_array, strict=True)):
        reasons: list[str] = []
        if not np.isfinite(score):
            reasons.append("non_finite_score")
        if not np.isfinite(pose).all():
            reasons.append("non_finite_pose")
        elif not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            reasons.append("invalid_homogeneous_row")
        else:
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3) or not np.isclose(
                np.linalg.det(rotation), 1.0, atol=5e-3
            ):
                reasons.append("invalid_rotation")
        if reasons:
            rejected.append({"index": index, "reasons": reasons})
            continue

        base_from_graspnet = pose if poses_are_base_frame else np.asarray(base_from_camera) @ pose
        base_from_hand = graspnet_pose_to_panda_hand(
            base_from_graspnet,
            reference_quaternion_xyzw=reference_quaternion_xyzw,
        )
        quaternion_xyzw = Rotation.from_matrix(base_from_hand[:3, :3]).as_quat()
        approach_z = float(base_from_hand[2, 2])
        contact = shift_along_approach(
            base_from_hand[:3, 3],
            np.roll(quaternion_xyzw, 1),
            GRASP_POSE_TO_CONTACT_M,
        )
        nearest_object_m = None
        if object_points.shape[0]:
            nearest_object_m = float(
                np.min(np.linalg.norm(object_points - contact[None, :], axis=1))
            )
        source_center_distance_m = (
            float(np.linalg.norm(contact - source_center)) if source_center is not None else None
        )
        if approach_z > 0.0:
            rejected.append(
                {
                    "index": index,
                    "reasons": ["approach_points_upward"],
                    "approach_z": round(approach_z, 6),
                    "target_xyz": _rounded_vector(contact),
                    "nearest_object_m": _rounded_scalar(nearest_object_m),
                }
            )
            continue
        if (
            source_center_distance_m is not None
            and source_radius_m is not None
            and source_center_distance_m > source_radius_m
        ):
            rejected.append(
                {
                    "index": index,
                    "reasons": ["outside_source_geometry"],
                    "approach_z": round(approach_z, 6),
                    "target_xyz": _rounded_vector(contact),
                    "nearest_object_m": _rounded_scalar(nearest_object_m),
                    "source_center_distance_m": _rounded_scalar(source_center_distance_m),
                    "source_radius_m": _rounded_scalar(source_radius_m),
                }
            )
            continue
        prepared.append(
            _PreparedGrasp(
                target=Pose(
                    tuple(float(value) for value in contact),
                    tuple(float(value) for value in quaternion_xyzw),
                ),
                score=float(score),
                raw_index=index,
                approach_z=approach_z,
                family="cgn",
            )
        )

    accepted = [
        {
            "index": item.raw_index,
            "approach_z": round(item.approach_z, 6),
            "target_xyz": _rounded_vector(item.target.position_xyz),
            "nearest_object_m": _rounded_scalar(
                float(
                    np.min(
                        np.linalg.norm(
                            object_points - np.asarray(item.target.position_xyz)[None, :],
                            axis=1,
                        )
                    )
                )
                if object_points.shape[0]
                else None
            ),
            "source_center_distance_m": _rounded_scalar(
                float(np.linalg.norm(np.asarray(item.target.position_xyz) - source_center))
                if source_center is not None
                else None
            ),
            "source_radius_m": _rounded_scalar(source_radius_m),
        }
        for item in prepared
    ]
    return prepared, {
        "raw_count": int(scores_array.size),
        "accepted_count": len(prepared),
        "accepted": accepted,
        "rejected": rejected,
    }


def _point_summary(points: Any) -> dict[str, Any]:
    if not _has_points(points):
        return {"count": 0}
    array = _points_array(points)
    return {
        "count": int(array.shape[0]),
        "min_xyz": _rounded_vector(np.min(array, axis=0)),
        "max_xyz": _rounded_vector(np.max(array, axis=0)),
        "centroid_xyz": _rounded_vector(np.mean(array, axis=0)),
    }


def _rounded_vector(values: Any) -> list[float]:
    return [round(float(value), 6) for value in np.asarray(values).reshape(-1)]


def _rounded_scalar(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _bounded_box(box: Any, width: int, height: int) -> tuple[float, float, float, float]:
    values = _vector(box, 4, "VLM bbox")
    x1, x2 = sorted((float(values[0]), float(values[2])))
    y1, y2 = sorted((float(values[1]), float(values[3])))
    x1, x2 = float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))
    y1, y2 = float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ContextFunctionError(f"VLM bbox has no area: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def _semantic_grounding_query(query: Any) -> str:
    """Make a forced single-box detector discriminate the requested semantics.

    The reduced CaP-X detector must return one box.  Without an explicit
    discrimination instruction, a VLM can select a larger or nearer generic
    instance (for example, one food can instead of another).  Keep this
    detector-only instruction private: the public evidence continues to store
    the caller's concise query.
    """

    target = " ".join(str(query).split())
    return (
        f"the exact semantic target described as {target}; use visible semantic "
        "attributes and relations to distinguish it from all other objects or "
        "regions, and do not select a generic visual match merely because it is "
        "closer or larger"
    )


def _grounding_coord_space(api: Any) -> str:
    """Reuse the configured CaP-X grounding convention at the VAW boundary."""

    resolver = getattr(api, "_vlm_grounding_coord_space", None)
    if callable(resolver):
        try:
            value = str(resolver(None, None))
        except (TypeError, ValueError):
            value = "pixel"
        if value in {"pixel", "norm1000"}:
            return value
    return "pixel"


def _scale_box_between_images(
    box: tuple[float, float, float, float],
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map one xyxy box between aligned rasters without changing its semantics."""

    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ContextFunctionError("cannot map a box between empty images")
    scale_x = target_w / source_w
    scale_y = target_h / source_h
    return _bounded_box(
        (
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y,
        ),
        target_w,
        target_h,
    )


def _vector(values: Any, length: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ContextFunctionError(f"{label} must contain {length} numbers") from exc
    if array.size != length or not np.isfinite(array).all():
        raise ContextFunctionError(f"{label} must contain {length} finite numbers")
    return array


def _frame(value: Any) -> str:
    frame = str(value)
    if frame not in {"base", "tool"}:
        raise ContextFunctionError("frame must be either 'base' or 'tool'")
    return frame


def _pose_rotation(pose: Pose) -> Rotation:
    quaternion = np.asarray(pose.quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ContextFunctionError("pose quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ContextFunctionError("pose quaternion must be non-zero")
    return Rotation.from_quat(quaternion / norm)


__all__ = ["ContextFunctions"]
