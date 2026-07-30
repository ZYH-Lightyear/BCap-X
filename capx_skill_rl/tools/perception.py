"""Observation-grounded perception tools."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from capx_skill_rl.context import EnvContext, SensorFrame


def vlm_bbox_detection(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    bbox = _finite_vector(
        context.backend.vlm_bbox_detection(context.frame.rgb, arguments["query"]),
        length=4,
        name="VLM bbox",
    )
    _validate_bbox_in_frame(bbox, context.frame)
    return {"bbox": bbox}


def vlm_point_detection(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    point = _finite_vector(
        context.backend.vlm_point_detection(context.frame.rgb, arguments["query"]),
        length=2,
        name="VLM point",
    )
    _validate_point_in_frame(point, context.frame)
    return {"point": point}


def sam3(context: EnvContext, arguments: dict[str, Any]) -> dict[str, Any]:
    if "text" in arguments:
        results = context.backend.sam3_text(context.frame.rgb, arguments["text"])
    elif "bbox" in arguments:
        _validate_bbox_in_frame(arguments["bbox"], context.frame)
        results = context.backend.sam3_box(context.frame.rgb, arguments["bbox"])
    else:
        _validate_point_in_frame(arguments["point"], context.frame)
        results = context.backend.sam3_point(context.frame.rgb, arguments["point"])

    candidates: list[tuple[float, np.ndarray]] = []
    for result in results:
        mask = result.get("mask")
        if mask is None:
            continue
        mask_array = np.asarray(mask, dtype=bool).squeeze()
        if mask_array.shape != context.frame.depth.shape or not mask_array.any():
            continue
        candidates.append((float(result.get("score", 0.0)), mask_array))
    if not candidates:
        raise ValueError("SAM3 returned no non-empty mask for the current image")

    _, best_mask = max(candidates, key=lambda item: item[0])
    mask_id = context.artifacts.add_mask(best_mask, context.frame.revision)
    return {"mask_id": mask_id}


def get_obb(context: EnvContext, arguments: dict[str, Any]) -> dict[str, Any]:
    mask = context.artifacts.get_mask(
        arguments["mask_id"],
        context.frame.revision,
    )
    points_base = _masked_points_in_base(context.frame, mask)
    obb = context.backend.get_obb(points_base)
    center = _finite_vector(obb.get("center"), length=3, name="OBB center")
    extent = _finite_vector(obb.get("extent"), length=3, name="OBB extent")
    rotation = np.asarray(obb.get("R"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("OBB rotation must be a finite 3x3 matrix")
    quaternion = Rotation.from_matrix(rotation).as_quat().tolist()
    return {
        "center": center,
        "extent": extent,
        "quaternion": [float(value) for value in quaternion],
    }


def plan_grasp(context: EnvContext, arguments: dict[str, Any]) -> dict[str, Any]:
    mask = context.artifacts.get_mask(
        arguments["mask_id"],
        context.frame.revision,
    )
    poses_camera, scores = context.backend.plan_grasp(
        context.frame.depth,
        context.frame.intrinsics,
        mask,
    )
    poses = np.asarray(poses_camera, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"grasp poses must have shape (K, 4, 4), got {poses.shape}")
    if len(poses) == 0 or len(poses) != len(score_array):
        raise ValueError("grasp planner returned no aligned pose/score candidates")
    valid = np.isfinite(score_array)
    if not valid.any():
        raise ValueError("grasp planner returned no finite score")
    best_index = int(np.argmax(np.where(valid, score_array, -np.inf)))
    pose_base = context.frame.base_from_camera @ poses[best_index]
    if not np.isfinite(pose_base).all():
        raise ValueError("selected grasp pose contains non-finite values")
    position = pose_base[:3, 3].tolist()
    quaternion = Rotation.from_matrix(pose_base[:3, :3]).as_quat().tolist()
    return {
        "position": [float(value) for value in position],
        "quaternion": [float(value) for value in quaternion],
    }


def _masked_points_in_base(frame: SensorFrame, mask: np.ndarray) -> np.ndarray:
    height, width = frame.depth.shape
    ys, xs = np.indices((height, width), dtype=np.float64)
    z = np.asarray(frame.depth, dtype=np.float64)
    fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
    cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("camera intrinsics contain a zero focal length")
    points_camera = np.stack(
        [
            (xs - cx) * z / fx,
            (ys - cy) * z / fy,
            z,
        ],
        axis=-1,
    )[mask]
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0)
    points_camera = points_camera[valid]
    if len(points_camera) < 4:
        raise ValueError("mask contains fewer than four valid depth points")
    homogeneous = np.concatenate(
        [points_camera, np.ones((len(points_camera), 1), dtype=np.float64)],
        axis=1,
    )
    points_base = (frame.base_from_camera @ homogeneous.T).T[:, :3]
    return points_base


def _finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) != length or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    return [float(item) for item in array]


def _validate_bbox_in_frame(bbox: list[float], frame: SensorFrame) -> None:
    x1, y1, x2, y2 = bbox
    height, width = frame.rgb.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"bbox must lie within the current {width}x{height} RGB image"
        )


def _validate_point_in_frame(point: list[float], frame: SensorFrame) -> None:
    x, y = point
    height, width = frame.rgb.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"point must lie within the current {width}x{height} RGB image"
        )


__all__ = [
    "get_obb",
    "plan_grasp",
    "sam3",
    "vlm_bbox_detection",
    "vlm_point_detection",
]
