"""GG-CNN client (HTTP) for generative antipodal grasp synthesis from depth.

Service URL defaults to ``http://127.0.0.1:8119`` and can be overridden with
``GGCNN_SERVICE_URL``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import numpy as np
import requests

from capx.utils.serve_utils import http_get, http_post

logger = logging.getLogger(__name__)

SERVICE_URL = os.environ.get("GGCNN_SERVICE_URL", "http://127.0.0.1:8119")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    with io.BytesIO(data) as f:
        return np.load(f)


def health_check(timeout: float = 2.0) -> bool:
    """Return True if the GG-CNN service responds to ``/health``."""
    try:
        resp = http_get(f"{SERVICE_URL}/health", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def init_ggcnn(device: str = "cuda", checkpoint_path: str | None = None) -> Any:
    """Initialize a GG-CNN client callable.

    Arguments are kept for API symmetry with ``init_contact_graspnet``; the
    model itself lives in the FastAPI process.
    """
    _ = (device, checkpoint_path)

    def plan(
        depth: np.ndarray,
        cam_K: np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        segmap_id: int = 1,
        n_grasps: int = 5,
        output_size: int = 300,
        inpaint: bool = True,
        min_distance: int = 20,
        threshold_abs: float = 0.2,
        width_scale_m: float = 0.0,
        return_maps: bool = False,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Call the GG-CNN ``/plan`` endpoint.

        Args:
            depth: HxW depth image (meters preferred).
            cam_K: optional 3x3 intrinsics; when set, each grasp may include a
                4x4 camera-frame pose.
            segmap: optional HxW integer mask; only ``segmap_id`` pixels keep Q.
            n_grasps: max peaks to return.
            return_maps: if True, also decode Q / angle / width heatmaps.

        Returns:
            dict with keys:
              - ``grasps``: list of dicts (row, col, angle, width_px, quality, ...)
              - ``poses``: (N, 4, 4) float32 array (empty if no poses)
              - ``scores``: (N,) qualities
              - optionally ``q``, ``ang``, ``width`` heatmaps
        """
        payload: dict[str, Any] = {
            "depth_base64": _numpy_to_base64(np.asarray(depth, dtype=np.float32)),
            "segmap_id": int(segmap_id),
            "n_grasps": int(n_grasps),
            "output_size": int(output_size),
            "inpaint": bool(inpaint),
            "min_distance": int(min_distance),
            "threshold_abs": float(threshold_abs),
            "width_scale_m": float(width_scale_m),
        }
        if cam_K is not None:
            payload["cam_K_base64"] = _numpy_to_base64(np.asarray(cam_K, dtype=np.float32))
        if segmap is not None:
            payload["segmap_base64"] = _numpy_to_base64(np.asarray(segmap))

        try:
            resp = http_post(f"{SERVICE_URL}/plan", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with GG-CNN service at {SERVICE_URL}: {e}"
            ) from e

        grasps = data.get("grasps", [])
        poses = []
        scores = []
        for g in grasps:
            scores.append(float(g.get("quality", 0.0)))
            pose = g.get("pose")
            if pose is not None:
                poses.append(np.asarray(pose, dtype=np.float32))

        result: dict[str, Any] = {
            "grasps": grasps,
            "scores": np.asarray(scores, dtype=np.float32),
            "poses": np.stack(poses, axis=0) if poses else np.zeros((0, 4, 4), dtype=np.float32),
        }
        if return_maps:
            if data.get("q_base64"):
                result["q"] = _base64_to_numpy(data["q_base64"])
            if data.get("ang_base64"):
                result["ang"] = _base64_to_numpy(data["ang_base64"])
            if data.get("width_base64"):
                result["width"] = _base64_to_numpy(data["width_base64"])
        return result

    return plan
