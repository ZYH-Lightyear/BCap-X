"""AnyGrasp HTTP client (service default: http://127.0.0.1:8120)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import requests

from capx.utils.serve_utils import http_post

SERVICE_URL = os.environ.get("ANYGRASP_SERVICE_URL", "http://127.0.0.1:8120")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    with io.BytesIO(data) as f:
        return np.load(f)


def init_anygrasp(device: str = "cuda", checkpoint_path: str | None = None) -> Any:
    """Return a callable that plans grasps via the AnyGrasp local HTTP service.

    ``device`` / ``checkpoint_path`` are ignored in client mode (kept for API parity
    with Contact-GraspNet). Override the URL with ``ANYGRASP_SERVICE_URL``.
    """

    def plan(
        depth: np.ndarray,
        cam_K: np.ndarray,
        segmap: np.ndarray | None = None,
        segmap_id: int = 1,
        z_range: list[float] | None = None,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        approach_steering: list[float] | None = None,
        approach_thresh: float = 3.141592653589793,
        top_k: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan grasps from depth + intrinsics (+ optional segmap).

        Returns:
            grasps: (N, 4, 4) camera-frame poses (tip-adjusted)
            scores: (N,)
            widths: (N,) gripper widths in meters
        """
        if z_range is None:
            z_range = [0.0, 2.0]
        payload: dict[str, Any] = {
            "depth_base64": _numpy_to_base64(np.asarray(depth)),
            "cam_K_base64": _numpy_to_base64(np.asarray(cam_K)),
            "segmap_id": segmap_id,
            "z_range": z_range,
            "dense_grasp": dense_grasp,
            "collision_detection": collision_detection,
            "approach_steering": approach_steering,
            "approach_thresh": approach_thresh,
            "top_k": top_k,
        }
        if segmap is not None:
            payload["segmap_base64"] = _numpy_to_base64(np.asarray(segmap))

        try:
            resp = http_post(f"{SERVICE_URL}/plan", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with AnyGrasp service at {SERVICE_URL}: {e}"
            ) from e

        grasps = _base64_to_numpy(data["grasps_base64"])
        scores = _base64_to_numpy(data["scores_base64"])
        widths = _base64_to_numpy(data["widths_base64"])
        return grasps, scores, widths

    return plan


def init_anygrasp_points() -> Any:
    """Return a callable that plans grasps from a camera-frame point cloud."""

    def plan_points(
        points: np.ndarray,
        region_mask: np.ndarray | None = None,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        approach_steering: list[float] | None = None,
        approach_thresh: float = 3.141592653589793,
        top_k: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan grasps from XYZ points (N,3) float32 in camera frame.

        Returns:
            grasps: (N, 4, 4), scores: (N,), widths: (N,)
        """
        payload: dict[str, Any] = {
            "points_base64": _numpy_to_base64(np.asarray(points, dtype=np.float32)),
            "dense_grasp": dense_grasp,
            "collision_detection": collision_detection,
            "approach_steering": approach_steering,
            "approach_thresh": approach_thresh,
            "top_k": top_k,
        }
        if region_mask is not None:
            payload["region_mask_base64"] = _numpy_to_base64(
                np.asarray(region_mask, dtype=bool)
            )

        try:
            resp = http_post(f"{SERVICE_URL}/plan_points", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with AnyGrasp service at {SERVICE_URL}: {e}"
            ) from e

        grasps = _base64_to_numpy(data["grasps_base64"])
        scores = _base64_to_numpy(data["scores_base64"])
        widths = _base64_to_numpy(data["widths_base64"])
        return grasps, scores, widths

    return plan_points
