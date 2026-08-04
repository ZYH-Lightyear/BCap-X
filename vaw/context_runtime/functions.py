"""Implementations of the M1.4 agent-visible functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime.errors import ContextFunctionError
from vaw.context_runtime.geometry import (
    graspnet_pose_to_panda_hand,
    local_surface_depth,
    pixel_to_base,
    tcp_position_from_hand_pose,
)
from vaw.context_runtime.model import (
    ActionAdjustment,
    ActionCandidate,
    ActionPrediction,
    ActionProposal,
    ExecutionReceipt,
    PointEvidence,
    Pose,
    RegionEvidence,
    SpatialTargetSummary,
)
from vaw.context_runtime.motion import MotionBackendError, MotionPlan
from vaw.geometry import GRASP_POSE_TO_CONTACT_M, shift_along_approach

if TYPE_CHECKING:
    from collections.abc import Callable

    from vaw.context_runtime.workspace import ContextWorkspace


class ContextFunctions:
    """Function semantics separated from workspace lifecycle/dispatch."""

    MAX_DELTA_M = 0.03
    MAX_ROTATION_DEG = 90.0

    def __init__(self, workspace: ContextWorkspace) -> None:
        self.ws = workspace
        self.handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "inspect": self.inspect,
            "locate_point": self.locate_point,
            "propose_grasps": self.propose_grasps,
            "propose_pose": self.propose_pose,
            "select": self.select,
            "delta_move": self.delta_move,
            "rotate": self.rotate,
            "commit": self.commit,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "done": self.done,
        }

    # ---------------------------------------------------------- perception --
    def inspect(
        self,
        query: str,
        within_region_id: str | None = None,
    ) -> dict[str, Any]:
        camera = self.ws._camera()
        rgb = _camera_image(camera, "rgb")
        crop_rgb, origin = self.ws._crop_rgb(rgb, within_region_id)
        local_box = _bounded_box(
            self.ws._call_backend("vlm_bbox_detection", crop_rgb, str(query)),
            crop_rgb.shape[1],
            crop_rgb.shape[0],
        )
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
        return {
            "region_id": region_id,
            "bbox_xyxy_px": [round(float(value), 3) for value in global_box],
        }

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
        planned = self.ws._call_backend(
            "plan_grasp",
            _camera_image(camera, "depth"),
            camera["intrinsics"],
            mask.astype(np.int64),
        )
        if not isinstance(planned, tuple) or len(planned) != 2:
            raise ContextFunctionError("grasp planner did not return poses and scores")
        poses_camera, scores = planned
        scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        poses_array = np.asarray(poses_camera, dtype=np.float64)
        if scores_array.size == 0:
            raise ContextFunctionError(f"grasp planner returned no candidates for '{region_id}'")
        if poses_array.shape != (scores_array.size, 4, 4):
            raise ContextFunctionError(
                f"grasp poses {poses_array.shape} do not match {scores_array.size} scores"
            )
        order = np.argsort(scores_array, kind="stable")[::-1][: self.ws.MAX_GRASP_CANDIDATES]
        try:
            base_from_camera = np.asarray(camera["pose_mat"], dtype=np.float64).reshape(4, 4)
        except (KeyError, ValueError) as exc:
            raise ContextFunctionError(f"invalid camera pose: {exc}") from exc

        ids: list[str] = []
        for index in order:
            base_from_graspnet = base_from_camera @ poses_array[index]
            base_from_hand = graspnet_pose_to_panda_hand(base_from_graspnet)
            quaternion_xyzw = Rotation.from_matrix(base_from_hand[:3, :3]).as_quat()
            contact = shift_along_approach(
                base_from_hand[:3, 3],
                np.roll(quaternion_xyzw, 1),
                GRASP_POSE_TO_CONTACT_M,
            )
            target = Pose(
                tuple(float(value) for value in contact),
                tuple(float(value) for value in quaternion_xyzw),
            )
            candidate_id = self.ws.state.next_id("g")
            self.ws.state.candidates[candidate_id] = ActionCandidate(
                candidate_id=candidate_id,
                kind="grasp",
                source_ref=region_id,
                target_pose=target,
                source_revision=self.ws.state.observation_revision,
                prediction=self._predict(target),
            )
            ids.append(candidate_id)
        return {"candidate_ids": ids}

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
        plan = self.ws.motion.plan_pose(target)
        return self._create_action("pose", point_id, target, motion_plan=plan)

    def select(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.ws._current_candidate(candidate_id)
        region = (
            self.ws._current_region(candidate.source_ref)
            if candidate.source_ref is not None
            else None
        )
        mask = (
            self.ws._private.region_masks.get(region.region_id)
            if region is not None
            else None
        )
        if region is None or mask is None:
            raise ContextFunctionError(
                f"candidate '{candidate_id}' has no current source region mask"
            )
        plan = self.ws.motion.plan_grasp(
            candidate.target_pose,
            object_name=region.query,
            object_mask=mask,
            preview=candidate.prediction,
        )
        return self._create_action(
            "grasp",
            candidate_id,
            candidate.target_pose,
            motion_plan=plan,
        )

    def delta_move(
        self,
        delta_xyz_m: list[float],
        frame: str = "base",
        action_id: str | None = None,
    ) -> dict[str, Any]:
        delta = _vector(delta_xyz_m, 3, "delta_xyz_m")
        if not np.any(delta != 0.0):
            raise ContextFunctionError("delta_xyz_m must contain a non-zero offset")
        if np.any(np.abs(delta) > self.MAX_DELTA_M):
            raise ContextFunctionError(
                "each delta_xyz_m component must be within [-0.03, 0.03] meters"
            )
        normalized_frame = _frame(frame)
        reference, source_ref, parent = self._adjustment_reference(action_id)
        rotation = _pose_rotation(reference)
        displacement = delta if normalized_frame == "base" else rotation.apply(delta)
        target = Pose(
            tuple(float(value) for value in np.asarray(reference.position_xyz) + displacement),
            tuple(float(value) for value in rotation.as_quat()),
        )
        adjustment = ActionAdjustment(
            kind="delta_move",
            frame=normalized_frame,
            reference_pose=reference,
            parent_action_id=parent.action_id if parent is not None else None,
            delta_xyz_m=tuple(float(value) for value in delta),
        )
        plan = self._plan_adjusted_pose(target, source_ref)
        return self._create_action(
            "delta_move",
            source_ref,
            target,
            motion_plan=plan,
            adjustment=adjustment,
        )

    def rotate(
        self,
        axis: str,
        angle_deg: float,
        frame: str = "tool",
        action_id: str | None = None,
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
        reference, source_ref, parent = self._adjustment_reference(action_id)
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
        adjustment = ActionAdjustment(
            kind="rotate",
            frame=normalized_frame,
            reference_pose=reference,
            parent_action_id=parent.action_id if parent is not None else None,
            axis=normalized_axis,
            angle_deg=angle,
        )
        plan = self._plan_adjusted_pose(target, source_ref)
        return self._create_action(
            "rotate",
            source_ref,
            target,
            motion_plan=plan,
            adjustment=adjustment,
        )

    def _adjustment_reference(
        self, action_id: str | None
    ) -> tuple[Pose, str | None, ActionProposal | None]:
        if action_id is not None:
            action = self.ws.state.active_action
            if (
                action is None
                or action.action_id != action_id
                or action.source_revision != self.ws.state.observation_revision
            ):
                raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
            reference_rotation = _pose_rotation(action.target_pose)
            reference = Pose(
                action.target_pose.position_xyz,
                tuple(float(value) for value in reference_rotation.as_quat()),
            )
            return reference, action.source_ref, action

        robot = self.ws.state.robot
        if robot is None or robot.tcp_pose is None:
            raise ContextFunctionError("current TCP pose is unavailable")
        try:
            rotation = _pose_rotation(robot.tcp_pose)
        except ValueError as exc:
            raise ContextFunctionError(f"current TCP pose is invalid: {exc}") from exc
        return (
            Pose(
                robot.tcp_pose.position_xyz,
                tuple(float(value) for value in rotation.as_quat()),
            ),
            None,
            None,
        )

    def _plan_adjusted_pose(self, target: Pose, source_ref: str | None) -> MotionPlan:
        candidate = self.ws.state.candidates.get(source_ref) if source_ref is not None else None
        if candidate is not None and candidate.kind == "grasp":
            region = (
                self.ws.state.regions.get(candidate.source_ref)
                if candidate.source_ref is not None
                else None
            )
            mask = (
                self.ws._private.region_masks.get(region.region_id)
                if region is not None
                else None
            )
            if region is None or mask is None:
                raise ContextFunctionError(
                    f"grasp source for action candidate '{candidate.candidate_id}' is unavailable"
                )
            return self.ws.motion.plan_grasp(
                target,
                object_name=region.query,
                object_mask=mask,
            )
        return self.ws.motion.plan_pose(target)

    def _create_action(
        self,
        kind: str,
        source_ref: str | None,
        target: Pose,
        *,
        motion_plan: MotionPlan | None = None,
        adjustment: ActionAdjustment | None = None,
    ) -> dict[str, Any]:
        motion_plan = motion_plan or self.ws.motion.plan_pose(target)
        prediction = motion_plan.prediction
        previous = self.ws.state.active_action
        if previous is not None:
            self.ws._private.motion_plans.pop(previous.action_id, None)
        action = ActionProposal(
            action_id=self.ws.state.next_id("a"),
            kind=kind,
            source_ref=source_ref,
            source_revision=self.ws.state.observation_revision,
            target_pose=target,
            prediction=prediction,
            adjustment=adjustment,
        )
        self.ws.state.active_action = action
        self.ws._private.motion_plans[action.action_id] = motion_plan
        return {
            "action_id": action.action_id,
            "solve_ik": prediction.solve_ik,
            "trajectory_checked": prediction.trajectory_checked,
            "collision_checked": prediction.collision_checked,
        }

    def _predict(self, target: Pose) -> ActionPrediction:
        return self.ws.motion.preview(target)

    # ------------------------------------------------------------- physical --
    def commit(self, action_id: str) -> dict[str, Any]:
        action = self.ws.state.active_action
        if action is None or action.action_id != action_id:
            raise ContextFunctionError(f"unknown or expired action_id '{action_id}'")
        before = self.ws.state.observation_revision
        execution_error: ContextFunctionError | None = None
        try:
            plan = self.ws._private.motion_plans.get(action_id)
            if plan is None:
                raise ContextFunctionError(
                    f"active action '{action_id}' has no cached motion plan"
                )
            self.ws.motion.execute(plan, action.target_pose)
        except MotionBackendError as exc:
            execution_error = ContextFunctionError(str(exc))
        except ContextFunctionError as exc:
            execution_error = exc
        finally:
            self.ws.refresh_observation()

        achieved = self.ws.state.robot.ee_pose if self.ws.state.robot is not None else None
        position_error = None
        if achieved is not None:
            achieved_tcp_position = tcp_position_from_hand_pose(
                achieved.position_xyz,
                achieved.quaternion_xyzw,
                self.ws._tcp_to_hand_local_xyz,
            )
            position_error = float(
                np.linalg.norm(achieved_tcp_position - np.asarray(action.target_pose.position_xyz))
            )
        discrepancy = (
            {"execution_error": str(execution_error)} if execution_error is not None else None
        )
        receipt = ExecutionReceipt(
            receipt_id=self.ws.state.next_id("receipt"),
            function_name="commit",
            action_id=action_id,
            revision_before=before,
            revision_after=self.ws.state.observation_revision,
            position_error_m=position_error,
            discrepancy=discrepancy,
        )
        self.ws.state.last_spatial_target = SpatialTargetSummary(
            action_id=action_id,
            target_pose=action.target_pose,
            revision_after=self.ws.state.observation_revision,
        )
        self.ws.state.last_receipt = receipt
        if execution_error is not None:
            raise execution_error
        result: dict[str, Any] = {}
        if position_error is not None:
            result["position_error_m"] = round(position_error, 6)
        return result

    def open_gripper(self) -> dict[str, Any]:
        return self._execute_gripper("open")

    def close_gripper(self) -> dict[str, Any]:
        return self._execute_gripper("close")

    def _execute_gripper(self, action: str) -> dict[str, Any]:
        before = self.ws.state.observation_revision
        execution_error: ContextFunctionError | None = None
        try:
            self.ws._call_backend(f"{action}_gripper")
        except ContextFunctionError as exc:
            execution_error = exc
        finally:
            self.ws.refresh_observation()
        opening = self.ws.state.robot.gripper_opening if self.ws.state.robot is not None else None
        discrepancy = (
            {"execution_error": str(execution_error)} if execution_error is not None else None
        )
        receipt = ExecutionReceipt(
            receipt_id=self.ws.state.next_id("receipt"),
            function_name=f"{action}_gripper",
            revision_before=before,
            revision_after=self.ws.state.observation_revision,
            gripper_opening=opening,
            discrepancy=discrepancy,
        )
        self.ws.state.last_receipt = receipt
        if execution_error is not None:
            raise execution_error
        result: dict[str, Any] = {}
        if opening is not None:
            result["gripper_opening"] = round(float(opening), 6)
        return result

    def done(self, success: bool) -> dict[str, Any]:
        self.ws.finished = True
        self.ws.claimed_success = bool(success)
        return {}


def _camera_image(camera: dict[str, Any], name: str) -> np.ndarray:
    try:
        image = np.asarray(camera["images"][name])
    except (KeyError, TypeError) as exc:
        raise ContextFunctionError(f"camera has no {name} image") from exc
    return image


def _bounded_box(box: Any, width: int, height: int) -> tuple[float, float, float, float]:
    values = _vector(box, 4, "VLM bbox")
    x1, x2 = sorted((float(values[0]), float(values[2])))
    y1, y2 = sorted((float(values[1]), float(values[3])))
    x1, x2 = float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))
    y1, y2 = float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ContextFunctionError(f"VLM bbox has no area: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


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
