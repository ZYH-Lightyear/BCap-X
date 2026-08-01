"""Unified grasp proposal API over GG-CNN / GraspGen / GraspGenX HTTP clients.

Usage::

    from capx.integrations.vision.grasp_backends import (
        init_propose_grasp_pose,
        health_check,
    )

    propose = init_propose_grasp_pose("ggcnn")
    grasp_pose_dict = propose(
        depth=depth,
        cam_K=K,
        segmap=mask,
        num_grasps=5,
        quality_threshold=0.05,
    )

All backends share the same ``propose_grasp_pose`` signature. Call sites should
pass only parameters they intentionally set; unused geometry slots stay ``None``
and backend-irrelevant kwargs are ignored by the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

BackendName = Literal["ggcnn", "graspgen", "graspgenx"]

_BACKEND_ALIASES = {
    "ggcnn": "ggcnn",
    "gg-cnn": "ggcnn",
    "graspgen": "graspgen",
    "grasp_gen": "graspgen",
    "graspgenx": "graspgenx",
    "grasp_gen_x": "graspgenx",
    "graspgen-x": "graspgenx",
}


def _resolve_backend(backend: str) -> BackendName:
    key = (backend or "").strip().lower()
    resolved = _BACKEND_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown grasp backend {backend!r}; expected one of "
            f"{sorted(set(_BACKEND_ALIASES.values()))}."
        )
    return resolved  # type: ignore[return-value]


def health_check(backend: str, timeout: float = 3.0) -> bool:
    """Return True if the selected backend service responds healthy."""
    name = _resolve_backend(backend)
    if name == "ggcnn":
        from capx.integrations.vision import ggcnn as mod

        return mod.health_check(timeout=timeout)
    if name == "graspgen":
        from capx.integrations.vision import graspgen as mod

        return mod.health_check(timeout=timeout)
    from capx.integrations.vision import graspgenx as mod

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
    backend: str,
    **init_kwargs: Any,
) -> Callable[..., dict[str, Any]]:
    """Return a ``propose_grasp_pose(*inputs, **kwargs) -> grasp_pose_dict`` callable.

    Positional / keyword geometry inputs (same for all backends)::

        depth, cam_K, pc_full, pc_segment, segmap

    Keyword hyperparameters (same names; unused ones may be omitted / ``None``)::

        segmap_id, gripper_name, num_grasps, topk_num_grasps,
        grasp_threshold, quality_threshold, min_grasps, max_tries,
        remove_outliers, output_size, inpaint, min_distance, width_scale_m,
        return_maps, timeout

    Returns a dict with keys ``poses``, ``scores``, ``grasps``, ``contact_pts``,
    ``maps`` (absent modalities are ``None`` or empty arrays).
    """
    name = _resolve_backend(backend)

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

    # graspgenx
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


__all__ = ["health_check", "init_propose_grasp_pose"]
