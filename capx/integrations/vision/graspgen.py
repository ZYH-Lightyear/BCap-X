"""HTTP client for the GraspGen local FastAPI server (default :8121)."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import requests

SERVICE_URL = os.environ.get("GRASPGEN_SERVICE_URL", "http://127.0.0.1:8121")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str)
    with io.BytesIO(data) as f:
        return np.load(f)


def init_graspgen() -> Any:
    """Return a callable ``infer(pc, **kwargs) -> (grasps, scores)``."""

    def infer(
        pc: np.ndarray,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        payload = {
            "pc_base64": _numpy_to_base64(np.asarray(pc, dtype=np.float32)),
            "grasp_threshold": grasp_threshold,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk_num_grasps,
            "min_grasps": min_grasps,
            "max_tries": max_tries,
            "remove_outliers": remove_outliers,
        }
        try:
            resp = requests.post(f"{SERVICE_URL}/infer", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with GraspGen service at {SERVICE_URL}: {e}"
            ) from e
        return _base64_to_numpy(data["grasps_base64"]), _base64_to_numpy(
            data["scores_base64"]
        )

    return infer


def init_graspgen_point_clouds() -> Any:
    """Contact-GraspNet-shaped client: ``(pc_full, pc_segment) -> grasps, scores, contact``."""

    def plan_point_clouds(
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 100,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        payload = {
            "pc_full_base64": _numpy_to_base64(np.asarray(pc_full, dtype=np.float32)),
            "pc_segment_base64": _numpy_to_base64(
                np.asarray(pc_segment, dtype=np.float32)
            ),
            "segmap_id": segmap_id,
            "grasp_threshold": grasp_threshold,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk_num_grasps,
            "min_grasps": min_grasps,
            "max_tries": max_tries,
            "remove_outliers": remove_outliers,
        }
        try:
            resp = requests.post(
                f"{SERVICE_URL}/plan_point_clouds", json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with GraspGen service at {SERVICE_URL}: {e}"
            ) from e
        return (
            _base64_to_numpy(data["grasps_base64"]),
            _base64_to_numpy(data["scores_base64"]),
            _base64_to_numpy(data["contact_pts_base64"]),
        )

    return plan_point_clouds


def health_check(timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=timeout)
        return r.ok and r.json().get("status") == "ok"
    except requests.RequestException:
        return False
