"""Reusable object grounding helper for RoboMEx skills.

The functions here are sidecar helpers for agents. They do not move the robot.
Callers pass environment APIs explicitly so the helper can run inside the existing
Code-as-Policy sandbox without importing environment globals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw


def ground_object(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    target_name: str,
    artifacts_dir: str,
    evidence: dict[str, Any],
    vlm_bbox_detection: Callable[..., Any],
    segment_sam3_box_prompt: Callable[..., list[dict[str, Any]]],
    segment_sam3_text_prompt: Callable[..., list[dict[str, Any]]] | None,
    mask_to_world_points: Callable[..., np.ndarray],
    filter_noise: Callable[..., Any],
    box_filename: str = "segment_vlm_box.png",
    mask_filename: str = "segment_sam3_mask.png",
) -> dict[str, Any]:
    """Ground ``target_name`` and publish canonical grounding evidence.

    Canonical keys:
        ``EVIDENCE["object_grounding"]``
        ``EVIDENCE["grounding.mask"]``
        ``EVIDENCE["grounding.points"]``
    """

    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    px = _as_box(vlm_bbox_detection(rgb, target_name))
    box_path = str(art / box_filename)
    _save_box_overlay(rgb, px, box_path)

    results = segment_sam3_box_prompt(rgb, px)
    if not results and segment_sam3_text_prompt is not None:
        results = segment_sam3_text_prompt(rgb, text_prompt=target_name)
    if not results:
        raise AssertionError(f"no mask for target {target_name!r}")
    mask = max(results, key=lambda r: float(r.get("score", 0.0)))["mask"]

    mask_path = str(art / mask_filename)
    _save_mask_overlay(rgb, mask, px, mask_path)

    points = mask_to_world_points(mask, depth, intrinsics, camera_pose)
    filtered = filter_noise(points)
    points = filtered[0] if isinstance(filtered, tuple) else filtered
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise AssertionError(f"empty point cloud for target {target_name!r}")

    evidence["grounding.mask"] = mask
    evidence["grounding.points"] = points
    grounding = {
        "target": target_name,
        "bbox": [float(v) for v in px],
        "mask_key": "grounding.mask",
        "points_key": "grounding.points",
        "center_xyz": [float(v) for v in points.mean(axis=0)],
        "artifacts": {
            "bbox_overlay": box_path,
            "mask_overlay": mask_path,
        },
        "num_points": int(len(points)),
    }
    evidence["object_grounding"] = grounding
    return grounding


def ground_object_from_observation(
    obs: dict[str, Any],
    *,
    target_name: str,
    artifacts_dir: str,
    evidence: dict[str, Any],
    apis: dict[str, Any],
    camera: str | None = None,
) -> dict[str, Any]:
    """Ground an object from a live RoboMEx observation.

    This convenience helper keeps SubAgent calls short and avoids guessing parameter names. Pass
    ``apis=globals()`` from the Code-as-Policy sandbox so the helper can use the live
    VLM, SAM3, depth, and filtering functions already injected there.
    """

    cam = _select_camera(obs, camera)
    images = cam["images"]
    return ground_object(
        rgb=images["rgb"],
        depth=images["depth"],
        intrinsics=cam["intrinsics"],
        camera_pose=cam["pose_mat"],
        target_name=target_name,
        artifacts_dir=artifacts_dir,
        evidence=evidence,
        vlm_bbox_detection=_require_api(apis, "vlm_bbox_detection"),
        segment_sam3_box_prompt=_require_api(apis, "segment_sam3_box_prompt"),
        segment_sam3_text_prompt=apis.get("segment_sam3_text_prompt"),
        mask_to_world_points=_require_api(apis, "mask_to_world_points"),
        filter_noise=_require_api(apis, "filter_noise"),
    )


def _select_camera(obs: dict[str, Any], camera: str | None) -> dict[str, Any]:
    if camera:
        return obs[camera]
    for cam in obs.values():
        if (
            isinstance(cam, dict)
            and isinstance(cam.get("images"), dict)
            and "rgb" in cam["images"]
            and "depth" in cam["images"]
            and "intrinsics" in cam
            and "pose_mat" in cam
        ):
            return cam
    raise KeyError("no observation camera has rgb/depth/intrinsics/pose_mat")


def _require_api(apis: dict[str, Any], name: str) -> Any:
    fn = apis.get(name)
    if fn is None:
        raise KeyError(f"required RoboMEx sandbox API is missing: {name}")
    return fn


def _as_box(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"bbox must contain 4 values, got {value!r}")
    return [float(v) for v in arr]


def _save_box_overlay(rgb: np.ndarray, box: list[float], path: str) -> None:
    img = Image.fromarray(np.asarray(rgb).copy())
    ImageDraw.Draw(img).rectangle(box, outline=(255, 0, 0), width=3)
    img.save(path)


def _save_mask_overlay(rgb: np.ndarray, mask: np.ndarray, box: list[float], path: str) -> None:
    import cv2

    vis = np.asarray(rgb).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    img = Image.fromarray(vis)
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(255, 0, 0), width=2)
    img.save(path)
