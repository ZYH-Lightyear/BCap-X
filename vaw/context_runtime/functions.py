"""Implementations of the M1.4 agent-visible functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from capx.utils.depth_utils import depth_to_pointcloud
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
    ActionReview,
    ActionSeed,
    ActionTarget,
    ImaginationHandoff,
    ImaginationState,
    LastPhysicalAction,
    PointEvidence,
    Pose,
    RegionEvidence,
)
from vaw.context_runtime.motion import MotionBackendError, MotionPlan
from vaw.context_runtime.private import (
    ActionReviewArtifacts,
    ImaginationArtifacts,
    LastPhysicalArtifacts,
    PlanningContext,
    RegionGeometryArtifact,
    SeedArtifacts,
    VisualEdit,
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


class ContextFunctions:
    """Function semantics separated from workspace lifecycle/dispatch."""

    MAX_DELTA_M = 0.03
    MAX_ROTATION_DEG = 90.0

    def __init__(self, workspace: ContextWorkspace) -> None:
        self.ws = workspace
        self.handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "detection_and_sam": self.detection_and_sam,
            "locate_point": self.locate_point,
            "propose_grasps": self.propose_grasps,
            "propose_pose": self.propose_pose,
            "select": self.select,
            "delta_move": self.delta_move,
            "rotate": self.rotate,
            "commit": self.commit,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "finish_imagination": self.finish_imagination,
            "done": self.done,
        }

    # ---------------------------------------------------------- perception --
    def detection_and_sam(
        self,
        query: str,
        within_region_id: str | None = None,
    ) -> dict[str, Any]:
        camera = self.ws._camera()
        rgb = _camera_image(camera, "rgb")
        crop_rgb, origin = self.ws._crop_rgb(rgb, within_region_id)
        local_box = self._semantic_bbox(crop_rgb, str(query))
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
        candidate_reply = self.ws._call_backend(
            "query_vlm",
            candidate_prompt(
                query,
                width=rgb.shape[1],
                height=rgb.shape[0],
                coord_space=coord_space,
            ),
            images=rgb,
            temperature=0.0,
            max_tokens=512,
        )
        try:
            candidates = parse_candidates(
                str(candidate_reply),
                width=rgb.shape[1],
                height=rgb.shape[0],
                coord_space=coord_space,
            )
        except ValueError as exc:
            raise ContextFunctionError(f"semantic candidate parsing failed: {exc}") from exc
        diagnostics: dict[str, Any] = {
            "mode": "candidate_review",
            "coord_space": coord_space,
            "candidate_reply": str(candidate_reply),
            "candidate_boxes": [
                [round(float(value), 3) for value in item.box_xyxy_px]
                for item in candidates
            ],
        }
        self.ws._private.trace_diagnostics["semantic_grounding"] = diagnostics
        if not candidates:
            raise ContextFunctionError(f"semantic grounding found no candidate for '{query}'")

        review_raster = render_candidate_review(rgb, candidates)
        review_reply = self.ws._call_backend(
            "query_vlm",
            review_prompt(query),
            images=review_raster,
            temperature=0.0,
            max_tokens=160,
        )
        diagnostics["review_reply"] = str(review_reply)
        try:
            selected = parse_choice(str(review_reply), count=len(candidates))
        except ValueError as exc:
            raise ContextFunctionError(f"semantic candidate review failed: {exc}") from exc
        diagnostics["selected_candidate"] = (
            None if selected is None else int(selected + 1)
        )
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
            base_from_camera = np.asarray(
                camera["pose_mat"], dtype=np.float64
            ).reshape(4, 4)
        except (KeyError, ValueError) as exc:
            raise ContextFunctionError(f"invalid camera pose: {exc}") from exc
        prepared, diagnostic = _prepare_grasps(
            planned,
            poses_are_base_frame=False,
            base_from_camera=base_from_camera,
            reference_quaternion_xyzw=(
                self.ws.state.robot.ee_pose.quaternion_xyzw
                if self.ws.state.robot is not None
                and self.ws.state.robot.ee_pose is not None
                else None
            ),
            object_points_base=(
                geometry.filtered_object_points_base
                if geometry is not None
                else None
            ),
        )
        self.ws._private.trace_diagnostics["grasp_candidates"] = {
            "selected_source": "single_view" if prepared else None,
            "object_points": _point_summary(
                geometry.filtered_object_points_base if geometry is not None else None
            ),
            "attempts": [{"source": "single_view", **diagnostic}],
        }
        if not prepared:
            raise ContextFunctionError(
                f"grasp planner returned no geometrically valid candidates for '{region_id}'"
            )

        ids: list[str] = []
        ordered = sorted(prepared, key=lambda item: item.score, reverse=True)
        for item in ordered[: self.ws.MAX_GRASP_CANDIDATES]:
            seed_id = self.ws.state.next_id("s")
            target = ActionTarget(pose=item.target)
            prediction = self._predict(item.target)
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
        return RegionGeometryArtifact(
            agentview_mask=np.asarray(agentview_mask, dtype=bool).copy(),
            wrist_mask=None if wrist_mask is None else np.asarray(wrist_mask, dtype=bool).copy(),
            object_points_base=object_points,
            scene_points_base=scene_points,
            filtered_object_points_base=filtered,
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
        return self._start_imagination(
            ActionTarget(pose=target),
            ImaginationArtifacts(
                planning_context=PlanningContext(
                    source_kind="point", source_ref=point_id
                ),
                preview_plan=self.ws.motion.plan_pose(target),
            ),
        )

    def select(self, seed_id: str) -> dict[str, Any]:
        seed = self.ws._current_seed(seed_id)
        artifacts = self.ws._private.seed_artifacts.get(seed_id)
        if artifacts is None:
            raise ContextFunctionError(f"seed '{seed_id}' has no private artifacts")
        plan = self._plan_target(seed.target, artifacts.planning_context)
        return self._start_imagination(
            seed.target,
            ImaginationArtifacts(
                planning_context=artifacts.planning_context,
                preview_plan=plan,
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
        current = self.ws.state.imagination
        gripper = (
            current.target.gripper
            if current is not None
            else (
                artifacts.initial_target.gripper
                if artifacts.initial_target is not None
                else None
            )
        )
        return self._store_imagination(
            ActionTarget(
                pose=target,
                gripper=gripper,
            ),
            ImaginationArtifacts(
                planning_context=planning_context,
                preview_plan=plan,
                initial_target=artifacts.initial_target,
                previous_visual_edit=artifacts.previous_visual_edit,
                latest_visual_edit=visual_edit,
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
            raise ContextFunctionError("angle_deg must be within [-90, 90] degrees")
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
        current = self.ws.state.imagination
        gripper = (
            current.target.gripper
            if current is not None
            else (
                artifacts.initial_target.gripper
                if artifacts.initial_target is not None
                else None
            )
        )
        return self._store_imagination(
            ActionTarget(
                pose=target,
                gripper=gripper,
            ),
            ImaginationArtifacts(
                planning_context=planning_context,
                preview_plan=plan,
                initial_target=artifacts.initial_target,
                previous_visual_edit=artifacts.previous_visual_edit,
                latest_visual_edit=visual_edit,
                turn_count=artifacts.turn_count,
            ),
        )

    def _adjustment_reference(self) -> tuple[Pose, ImaginationArtifacts]:
        imagination = self.ws.state.imagination
        if imagination is not None and imagination.target.pose is not None:
            reference_rotation = _pose_rotation(imagination.target.pose)
            reference = Pose(
                imagination.target.pose.position_xyz,
                tuple(float(value) for value in reference_rotation.as_quat()),
            )
            return reference, self.ws._private.imagination_artifacts or ImaginationArtifacts()

        review = self.ws.state.action_review
        if review is not None and review.target.pose is not None:
            reference_rotation = _pose_rotation(review.target.pose)
            reference = Pose(
                review.target.pose.position_xyz,
                tuple(float(value) for value in reference_rotation.as_quat()),
            )
            review_artifacts = self.ws._private.review_artifacts.get(review.action_id)
            return (
                reference,
                ImaginationArtifacts(
                    planning_context=(
                        review_artifacts.planning_context
                        if review_artifacts is not None
                        else None
                    ),
                    preview_plan=(
                        review_artifacts.motion_plan
                        if review_artifacts is not None
                        else None
                    ),
                    initial_target=review.target,
                ),
            )

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
            ImaginationArtifacts(initial_target=ActionTarget(pose=reference)),
        )

    def _plan_adjusted_pose(
        self,
        target: Pose,
        planning_context: PlanningContext | None,
    ) -> MotionPlan:
        if planning_context is not None and planning_context.region_id is not None:
            region = self.ws.state.regions.get(planning_context.region_id)
            mask = self.ws._private.region_masks.get(planning_context.region_id)
            if region is None or mask is None:
                raise ContextFunctionError(
                    f"grasp source '{planning_context.source_ref}' is unavailable"
                )
            return self.ws.motion.plan_grasp(
                target,
                object_name=region.query,
                object_mask=mask,
            )
        return self.ws.motion.plan_pose(target)

    def _start_imagination(
        self, target: ActionTarget, artifacts: ImaginationArtifacts
    ) -> dict[str, Any]:
        self.ws.state.action_review = None
        self.ws._private.review_artifacts.clear()
        self.ws._private.imagination_artifacts = None
        artifacts = ImaginationArtifacts(
            planning_context=artifacts.planning_context,
            preview_plan=artifacts.preview_plan,
            initial_target=target,
            previous_visual_edit=None,
            latest_visual_edit=artifacts.latest_visual_edit,
            turn_count=0,
        )
        return self._store_imagination(target, artifacts)

    def _store_imagination(
        self, target: ActionTarget, artifacts: ImaginationArtifacts
    ) -> dict[str, Any]:
        before_target = (
            self.ws.state.imagination.target.summary()
            if self.ws.state.imagination is not None
            else None
        )
        previous = self.ws._private.imagination_artifacts
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
        self.ws.state.action_review = None
        self.ws._private.review_artifacts.clear()
        self.ws.state.imagination = ImaginationState(
            target=target,
            refinement_goal=self.ws.refinement_goal,
        )
        self.ws._private.imagination_artifacts = ImaginationArtifacts(
            planning_context=artifacts.planning_context,
            preview_plan=artifacts.preview_plan,
            initial_target=initial_target,
            previous_visual_edit=previous_edit,
            latest_visual_edit=latest_edit,
            turn_count=turn_count,
        )
        self.ws._private.trace_diagnostics["imagination_edit"] = {
            "target_before": before_target,
            "target_after": target.summary(),
            "refinement_goal": self.ws.refinement_goal,
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
        return _preview_result(target, self.ws._private.imagination_artifacts)

    def _plan_target(
        self, target: ActionTarget, context: PlanningContext | None
    ) -> MotionPlan | None:
        if target.pose is None:
            return None
        return self._plan_adjusted_pose(target.pose, context)

    def _predict(self, target: Pose) -> ActionPrediction:
        return self.ws.motion.preview(target)

    # ------------------------------------------------------------- physical --
    def commit(self, action_id: str) -> dict[str, Any]:
        action = self.ws.state.action_review
        if action is None or action.action_id != action_id:
            raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
        execution_error: ContextFunctionError | None = None
        failed_stage: str | None = (
            "arm" if action.target.pose is not None else "gripper"
        )
        executed_stages: list[str] = []
        focus_pose = (
            action.target.pose
            or (self.ws.state.robot.tcp_pose if self.ws.state.robot is not None else None)
        )
        try:
            artifacts = self.ws._private.review_artifacts.get(action_id)
            if artifacts is None:
                raise ContextFunctionError(
                    f"active action '{action_id}' has no private artifacts"
                )
            if action.target.pose is not None:
                failed_stage = "arm"
                executed_stages.append("arm")
                if artifacts.motion_plan is None:
                    raise ContextFunctionError(
                        f"active action '{action_id}' has no cached motion plan"
                    )
                self.ws.motion.execute(artifacts.motion_plan, action.target.pose)
            if action.target.gripper is not None:
                failed_stage = "gripper"
                executed_stages.append("gripper")
                backend_function = (
                    "open_gripper"
                    if action.target.gripper == "open"
                    else "close_gripper"
                )
                self.ws._call_backend(backend_function)
            failed_stage = None
        except MotionBackendError as exc:
            execution_error = ContextFunctionError(str(exc))
        except ContextFunctionError as exc:
            execution_error = exc
        finally:
            self.ws.refresh_observation()

        if not executed_stages and failed_stage is not None:
            executed_stages.append(failed_stage)
        if executed_stages == ["arm", "gripper"]:
            stage_summary = "arm+gripper"
        elif executed_stages == ["arm"]:
            stage_summary = "arm"
        else:
            stage_summary = "gripper"
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
        )
        self.ws._private.last_physical_artifacts = LastPhysicalArtifacts(
            focus_pose=focus_pose
        )

        self.ws._private.trace_diagnostics["waypoint_commit"] = {
            "action_id": action_id,
            "executed_stages": list(executed_stages),
            "failed_stage": failed_stage,
            "target_gripper_state": action.target.gripper,
        }

        achieved = self.ws.state.robot.ee_pose if self.ws.state.robot is not None else None
        position_error = None
        if achieved is not None and action.target.pose is not None:
            achieved_tcp_position = tcp_position_from_hand_pose(
                achieved.position_xyz,
                achieved.quaternion_xyzw,
                self.ws._tcp_to_hand_local_xyz,
            )
            position_error = float(
                np.linalg.norm(achieved_tcp_position - np.asarray(action.target.pose.position_xyz))
            )
        opening = self.ws.state.robot.gripper_opening if self.ws.state.robot is not None else None
        if execution_error is not None:
            raise execution_error
        result: dict[str, Any] = {}
        if position_error is not None:
            result["position_error_m"] = round(position_error, 6)
        if action.target.gripper is not None and opening is not None:
            result["gripper_opening"] = round(float(opening), 6)
        return result

    def open_gripper(self) -> dict[str, Any]:
        return self._preview_gripper("open")

    def close_gripper(self) -> dict[str, Any]:
        return self._preview_gripper("closed")

    def _preview_gripper(self, target: str) -> dict[str, Any]:
        imagination = self.ws.state.imagination
        if imagination is not None:
            artifacts = self.ws._private.imagination_artifacts or ImaginationArtifacts()
            pose = imagination.target.pose
        elif self.ws.state.action_review is not None:
            review = self.ws.state.action_review
            review_artifacts = self.ws._private.review_artifacts.get(review.action_id)
            artifacts = ImaginationArtifacts(
                planning_context=(
                    review_artifacts.planning_context
                    if review_artifacts is not None
                    else None
                ),
                preview_plan=(
                    review_artifacts.motion_plan
                    if review_artifacts is not None
                    else None
                ),
                initial_target=review.target,
            )
            pose = review.target.pose
        else:
            artifacts = ImaginationArtifacts()
            pose = None
        edit = VisualEdit(kind="gripper", gripper_target=target)
        return self._store_imagination(
            ActionTarget(pose=pose, gripper=_gripper_target(target)),
            ImaginationArtifacts(
                planning_context=artifacts.planning_context,
                preview_plan=artifacts.preview_plan,
                initial_target=artifacts.initial_target,
                previous_visual_edit=artifacts.previous_visual_edit,
                latest_visual_edit=edit,
                turn_count=artifacts.turn_count,
            ),
        )

    def finish_imagination(self, status: str) -> dict[str, Any]:
        if status not in {"ready", "failed"}:
            raise ContextFunctionError("status must be 'ready' or 'failed'")
        imagination = self.ws.state.imagination
        if imagination is None:
            raise ContextFunctionError("there is no active imagination session")
        if status == "failed":
            self.ws.state.imagination = None
            self.ws._private.imagination_artifacts = None
            self.ws.state.last_handoff = ImaginationHandoff(status="failed")
            self.ws._private.trace_diagnostics["imagination_handoff"] = {
                "status": "failed",
                "termination_reason": "agent_failed",
                "target": imagination.target.summary(),
            }
            return {"status": "failed"}
        return self._handoff_review("agent_ready")

    def limit_imagination(self) -> dict[str, Any]:
        """Return the final preview to Main without marking it as approved."""

        imagination = self.ws.state.imagination
        if imagination is None:
            raise ContextFunctionError("there is no active imagination session")
        return self._handoff_review("turn_limit")

    def _handoff_review(self, termination_reason: str) -> dict[str, Any]:
        imagination = self.ws.state.imagination
        if imagination is None:
            raise ContextFunctionError("there is no active imagination session")
        artifacts = self.ws._private.imagination_artifacts or ImaginationArtifacts()
        plan = artifacts.preview_plan
        if imagination.target.pose is not None and (
            plan is None or plan.prediction.solve_ik != "returned"
        ):
            self.ws.state.imagination = None
            self.ws._private.imagination_artifacts = None
            self.ws.state.last_handoff = ImaginationHandoff("failed")
            self.ws._private.trace_diagnostics["imagination_handoff"] = {
                "status": "failed",
                "termination_reason": f"{termination_reason}_without_executable_plan",
                "target": imagination.target.summary(),
            }
            return {"status": "failed"}

        action_id = self.ws.state.next_id("a")
        review = ActionReview(
            action_id,
            imagination.target,
            imagination.refinement_goal,
        )
        self.ws.state.action_review = review
        self.ws._private.review_artifacts = {
            action_id: ActionReviewArtifacts(
                motion_plan=plan,
                planning_context=artifacts.planning_context,
                termination_reason=termination_reason,
            )
        }
        self.ws.state.imagination = None
        self.ws._private.imagination_artifacts = None
        self.ws.state.last_handoff = ImaginationHandoff(
            "review_required", action_id
        )
        self.ws._private.trace_diagnostics["imagination_handoff"] = {
            "status": "review_required",
            "termination_reason": termination_reason,
            "action_id": action_id,
            "target": review.target.summary(),
            "intent": review.intent,
        }
        return self.ws.state.last_handoff.summary()

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


def _preview_result(
    target: ActionTarget,
    artifacts: ImaginationArtifacts,
) -> dict[str, Any]:
    result: dict[str, Any] = {"preview": "updated"}
    plan = artifacts.preview_plan
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
    prepared: list[_PreparedGrasp] = []
    rejected: list[dict[str, Any]] = []
    for index, (pose, score) in enumerate(
        zip(poses_array, scores_array, strict=True)
    ):
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

        base_from_graspnet = (
            pose if poses_are_base_frame else np.asarray(base_from_camera) @ pose
        )
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
        prepared.append(
            _PreparedGrasp(
                target=Pose(
                    tuple(float(value) for value in contact),
                    tuple(float(value) for value in quaternion_xyzw),
                ),
                score=float(score),
                raw_index=index,
                approach_z=approach_z,
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
                            object_points
                            - np.asarray(item.target.position_xyz)[None, :],
                            axis=1,
                        )
                    )
                )
                if object_points.shape[0]
                else None
            ),
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
