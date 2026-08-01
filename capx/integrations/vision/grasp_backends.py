"""Unified grasp proposal API over GG-CNN / GraspGen / GraspGenX HTTP clients.

Usage::

    from capx.integrations.vision.grasp_backends import (
        init_propose_grasp_pose,
        health_check,
    )

    # Default backend is GraspGen (:8121); override with CAPX_GRASP_BACKEND
    # or init_propose_grasp_pose("ggcnn" | "graspgenx").
    propose = init_propose_grasp_pose()
    grasp_pose_dict = propose(pc_segment=pc, num_grasps=80, topk_num_grasps=8)

All backends share the same ``propose_grasp_pose`` signature. Call sites should
pass only parameters they intentionally set; unused geometry slots stay ``None``
and backend-irrelevant kwargs are ignored by the adapter.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

BackendName = Literal["ggcnn", "graspgen", "graspgenx", "graspnet"]
DEFAULT_BACKEND: BackendName = "graspgen"

_BACKEND_ALIASES = {
    "ggcnn": "ggcnn",
    "gg-cnn": "ggcnn",
    "graspgen": "graspgen",
    "grasp_gen": "graspgen",
    "graspgenx": "graspgenx",
    "grasp_gen_x": "graspgenx",
    "graspgen-x": "graspgenx",
    "graspnet": "graspnet",
    "contact_graspnet": "graspnet",
    "contact-graspnet": "graspnet",
}


def _resolve_backend(backend: str | None = None) -> BackendName:
    raw = (
        backend
        if backend is not None
        else os.environ.get("CAPX_GRASP_BACKEND", DEFAULT_BACKEND)
    )
    key = (raw or DEFAULT_BACKEND).strip().lower()
    resolved = _BACKEND_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown grasp backend {raw!r}; expected one of "
            f"{sorted(set(_BACKEND_ALIASES.values()))}."
        )
    return resolved  # type: ignore[return-value]


def health_check(backend: str | None = None, timeout: float = 3.0) -> bool:
    """Return True if the selected backend service responds healthy.

    When ``backend`` is ``None``, uses ``CAPX_GRASP_BACKEND`` or GraspGen.
    """
    name = _resolve_backend(backend)
    if name == "ggcnn":
        from capx.integrations.vision import ggcnn as mod

        return mod.health_check(timeout=timeout)
    if name == "graspgen":
        from capx.integrations.vision import graspgen as mod

        return mod.health_check(timeout=timeout)
    if name == "graspgenx":
        from capx.integrations.vision import graspgenx as mod

        return mod.health_check(timeout=timeout)
    from capx.integrations.vision import graspnet as mod

    return mod.health_check(timeout=timeout)


def _empty_poses() -> np.ndarray:
    return np.zeros((0, 4, 4), dtype=np.float32)


def _empty_scores() -> np.ndarray:
    return np.zeros((0,), dtype=np.float32)


def _normalize_result(
    *,
    poses: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    grasps: list | None = None,
    contact_pts: np.ndarray | None = None,
    maps: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    poses_arr = (
        np.asarray(poses, dtype=np.float32)
        if poses is not None and len(poses)
        else _empty_poses()
    )
    if poses_arr.ndim == 2 and poses_arr.shape == (4, 4):
        poses_arr = poses_arr[None, ...]
    scores_arr = (
        np.asarray(scores, dtype=np.float32)
        if scores is not None
        else _empty_scores()
    )
    return {
        "poses": poses_arr,
        "scores": scores_arr,
        "grasps": grasps,
        "contact_pts": contact_pts,
        "maps": maps,
    }


def init_propose_grasp_pose(
    backend: str | None = None,
    **init_kwargs: Any,
) -> Callable[..., dict[str, Any]]:
    """Return a ``propose_grasp_pose(*inputs, **kwargs) -> grasp_pose_dict`` callable.

    Args:
        backend: ``"ggcnn"`` / ``"graspgen"`` / ``"graspgenx"`` / ``"graspnet"``.
            When ``None``, uses env ``CAPX_GRASP_BACKEND`` (default ``"graspgen"``).

    Positional / keyword geometry inputs (same for all backends)::

        depth, cam_K, pc_full, pc_segment, segmap

    Keyword hyperparameters (same names; unused ones may be omitted / ``None``)::

        segmap_id, gripper_name, num_grasps, topk_num_grasps,
        grasp_threshold, quality_threshold, min_grasps, max_tries,
        remove_outliers, forward_passes, output_size, inpaint, min_distance,
        width_scale_m, return_maps, timeout

    Returns a dict with keys ``poses``, ``scores``, ``grasps``, ``contact_pts``,
    ``maps`` (absent modalities are ``None`` or empty arrays).
    """
    name = _resolve_backend(backend)
    print(f"[grasp_backends] propose_grasp_pose backend={name}", flush=True)

    if name == "ggcnn":
        from capx.integrations.vision.ggcnn import init_ggcnn

        plan = init_ggcnn(**init_kwargs)

        def propose_grasp_pose(
            depth: np.ndarray | None = None,
            cam_K: np.ndarray | None = None,
            pc_full: np.ndarray | None = None,
            pc_segment: np.ndarray | None = None,
            segmap: np.ndarray | None = None,
            *,
            segmap_id: int = 1,
            gripper_name: str | None = None,
            num_grasps: int = 5,
            topk_num_grasps: int | None = None,
            grasp_threshold: float | None = None,
            quality_threshold: float | None = None,
            min_grasps: int | None = None,
            max_tries: int | None = None,
            remove_outliers: bool | None = None,
            forward_passes: int | None = None,
            output_size: int = 300,
            inpaint: bool = True,
            min_distance: int = 20,
            width_scale_m: float = 0.0,
            return_maps: bool = False,
            timeout: float = 60.0,
        ) -> dict[str, Any]:
            del (
                pc_full,
                pc_segment,
                gripper_name,
                topk_num_grasps,
                grasp_threshold,
                min_grasps,
                max_tries,
                remove_outliers,
                forward_passes,
            )
            if depth is None:
                raise ValueError("ggcnn propose_grasp_pose requires depth")
            thr = 0.2 if quality_threshold is None else float(quality_threshold)
            raw = plan(
                depth=depth,
                cam_K=cam_K,
                segmap=segmap,
                segmap_id=segmap_id,
                n_grasps=num_grasps,
                output_size=output_size,
                inpaint=inpaint,
                min_distance=min_distance,
                threshold_abs=thr,
                width_scale_m=width_scale_m,
                return_maps=return_maps,
                timeout=timeout,
            )
            maps = None
            if return_maps:
                maps = {
                    k: raw[k] for k in ("q", "ang", "width") if k in raw
                } or None
            return _normalize_result(
                poses=raw.get("poses"),
                scores=raw.get("scores"),
                grasps=raw.get("grasps"),
                maps=maps,
            )

        return propose_grasp_pose

    if name == "graspgen":
        from capx.integrations.vision.graspgen import (
            init_graspgen,
            init_graspgen_point_clouds,
        )

        if init_kwargs:
            raise TypeError(
                f"Unexpected init kwargs for graspgen: {sorted(init_kwargs)}"
            )
        infer = init_graspgen()
        plan_pcs = init_graspgen_point_clouds()

        def propose_grasp_pose(
            depth: np.ndarray | None = None,
            cam_K: np.ndarray | None = None,
            pc_full: np.ndarray | None = None,
            pc_segment: np.ndarray | None = None,
            segmap: np.ndarray | None = None,
            *,
            segmap_id: int = 1,
            gripper_name: str | None = None,
            num_grasps: int = 200,
            topk_num_grasps: int = 100,
            grasp_threshold: float | None = None,
            quality_threshold: float | None = None,
            min_grasps: int = 40,
            max_tries: int = 6,
            remove_outliers: bool = True,
            forward_passes: int | None = None,
            output_size: int | None = None,
            inpaint: bool | None = None,
            min_distance: int | None = None,
            width_scale_m: float | None = None,
            return_maps: bool = False,
            timeout: float = 120.0,
        ) -> dict[str, Any]:
            del (
                depth,
                cam_K,
                segmap,
                gripper_name,
                quality_threshold,
                forward_passes,
                output_size,
                inpaint,
                min_distance,
                width_scale_m,
                return_maps,
                timeout,
            )
            if pc_segment is None:
                raise ValueError("graspgen propose_grasp_pose requires pc_segment")
            thr = -1.0 if grasp_threshold is None else float(grasp_threshold)
            if pc_full is not None:
                poses, scores, contact = plan_pcs(
                    pc_full,
                    pc_segment,
                    segmap_id=segmap_id,
                    grasp_threshold=thr,
                    num_grasps=num_grasps,
                    topk_num_grasps=topk_num_grasps,
                    min_grasps=min_grasps,
                    max_tries=max_tries,
                    remove_outliers=remove_outliers,
                )
                return _normalize_result(
                    poses=poses, scores=scores, contact_pts=contact
                )
            poses, scores = infer(
                pc_segment,
                grasp_threshold=thr,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )
            return _normalize_result(poses=poses, scores=scores)

        return propose_grasp_pose

    if name == "graspgenx":
        from capx.integrations.vision.graspgenx import (
            init_graspgenx,
            init_graspgenx_point_clouds,
        )

        default_gripper = init_kwargs.pop("default_gripper", "franka_panda")
        if init_kwargs:
            raise TypeError(
                f"Unexpected init kwargs for graspgenx: {sorted(init_kwargs)}"
            )
        infer = init_graspgenx(default_gripper=default_gripper)
        plan_pcs = init_graspgenx_point_clouds(default_gripper=default_gripper)

        def propose_grasp_pose(
            depth: np.ndarray | None = None,
            cam_K: np.ndarray | None = None,
            pc_full: np.ndarray | None = None,
            pc_segment: np.ndarray | None = None,
            segmap: np.ndarray | None = None,
            *,
            segmap_id: int = 1,
            gripper_name: str | None = None,
            num_grasps: int = 200,
            topk_num_grasps: int = 100,
            grasp_threshold: float | None = None,
            quality_threshold: float | None = None,
            min_grasps: int = 40,
            max_tries: int = 6,
            remove_outliers: bool = True,
            forward_passes: int | None = None,
            output_size: int | None = None,
            inpaint: bool | None = None,
            min_distance: int | None = None,
            width_scale_m: float | None = None,
            return_maps: bool = False,
            timeout: float = 180.0,
        ) -> dict[str, Any]:
            del (
                depth,
                cam_K,
                segmap,
                quality_threshold,
                forward_passes,
                output_size,
                inpaint,
                min_distance,
                width_scale_m,
                return_maps,
                timeout,
            )
            if pc_segment is None:
                raise ValueError("graspgenx propose_grasp_pose requires pc_segment")
            thr = -1.0 if grasp_threshold is None else float(grasp_threshold)
            if pc_full is not None:
                poses, scores, contact = plan_pcs(
                    pc_full,
                    pc_segment,
                    segmap_id=segmap_id,
                    gripper_name=gripper_name,
                    grasp_threshold=thr,
                    num_grasps=num_grasps,
                    topk_num_grasps=topk_num_grasps,
                    min_grasps=min_grasps,
                    max_tries=max_tries,
                    remove_outliers=remove_outliers,
                )
                return _normalize_result(
                    poses=poses, scores=scores, contact_pts=contact
                )
            poses, scores = infer(
                pc_segment,
                gripper_name=gripper_name,
                grasp_threshold=thr,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )
            return _normalize_result(poses=poses, scores=scores)

        return propose_grasp_pose

    # graspnet (Contact-GraspNet :8115)
    from capx.integrations.vision.graspnet import (
        init_contact_graspnet,
        init_contact_graspnet_point_clouds,
    )

    plan = init_contact_graspnet(**init_kwargs)
    plan_pcs = init_contact_graspnet_point_clouds()

    def propose_grasp_pose(
        depth: np.ndarray | None = None,
        cam_K: np.ndarray | None = None,
        pc_full: np.ndarray | None = None,
        pc_segment: np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        *,
        segmap_id: int = 1,
        gripper_name: str | None = None,
        num_grasps: int | None = None,
        topk_num_grasps: int | None = None,
        grasp_threshold: float | None = None,
        quality_threshold: float | None = None,
        min_grasps: int | None = None,
        max_tries: int = 10,
        remove_outliers: bool | None = None,
        forward_passes: int = 2,
        output_size: int | None = None,
        inpaint: bool | None = None,
        min_distance: int | None = None,
        width_scale_m: float | None = None,
        return_maps: bool = False,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        del (
            gripper_name,
            num_grasps,
            topk_num_grasps,
            grasp_threshold,
            quality_threshold,
            min_grasps,
            remove_outliers,
            output_size,
            inpaint,
            min_distance,
            width_scale_m,
            return_maps,
            timeout,
        )
        if pc_full is not None and pc_segment is not None:
            poses, scores, contact = plan_pcs(
                pc_full,
                pc_segment,
                segmap_id=segmap_id,
                forward_passes=forward_passes,
                max_retries=max_tries,
            )
            return _normalize_result(
                poses=poses, scores=scores, contact_pts=contact
            )
        if depth is None or cam_K is None or segmap is None:
            raise ValueError(
                "graspnet propose_grasp_pose requires "
                "(depth, cam_K, segmap) or (pc_full, pc_segment)"
            )
        poses, scores, contact = plan(
            depth=np.asarray(depth, dtype=np.float32),
            cam_K=np.asarray(cam_K, dtype=np.float32),
            segmap=np.asarray(segmap),
            segmap_id=segmap_id,
            forward_passes=forward_passes,
            max_retries=max_tries,
        )
        return _normalize_result(poses=poses, scores=scores, contact_pts=contact)

    return propose_grasp_pose


__all__ = ["DEFAULT_BACKEND", "health_check", "init_propose_grasp_pose"]
