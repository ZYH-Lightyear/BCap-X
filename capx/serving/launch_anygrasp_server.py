"""AnyGrasp grasp-detection HTTP service (default port 8120).

Mirrors Contact-GraspNet's local-port pattern: FastAPI + base64 numpy payloads.
Requires a licensed AnyGrasp SDK under ``capx/third_party/anygrasp_sdk`` plus a
checkpoint. Use ``--mock`` to exercise the HTTP contract without a license.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Service Configuration ---
app = FastAPI(title="AnyGrasp Service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
_DETECTOR: Any | None = None
_MOCK: bool = False
_DEVICE: str = "cuda"
_GPU_SEMAPHORE = asyncio.Semaphore(1)

_SDK_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "anygrasp_sdk"
_DETECT_DIR = _SDK_ROOT / "grasp_detection"
_DEFAULT_CHECKPOINT = _DETECT_DIR / "log" / "checkpoint_detection.tar"
_DEFAULT_LICENSE = _DETECT_DIR / "license"


async def _run_on_gpu(fn, *args, **kwargs):
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}") from e


def _ensure_gsnet_so() -> None:
    """Copy the matching CPython extension into grasp_detection/ as gsnet.so."""
    py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = sorted((_DETECT_DIR / "gsnet_versions").glob(f"gsnet.{py_tag}*-x86_64-linux-gnu.so"))
    if not candidates:
        raise FileNotFoundError(
            f"No gsnet .so for Python {sys.version_info.major}.{sys.version_info.minor} "
            f"under {_DETECT_DIR / 'gsnet_versions'}"
        )
    target = _DETECT_DIR / "gsnet.so"
    src = candidates[-1]
    if not target.exists() or target.resolve() != src.resolve():
        target.write_bytes(src.read_bytes())
        logger.info("Installed %s -> %s", src.name, target)


def _depth_to_points(
    depth: np.ndarray,
    cam_K: np.ndarray,
    segmap: np.ndarray | None = None,
    segmap_id: int | None = None,
    z_range: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project depth to camera-frame XYZ; optional region mask aligned with points."""
    if z_range is None:
        z_range = [0.0, 2.0]
    h, w = depth.shape[:2]
    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    valid = (z > z_range[0]) & (z < z_range[1])
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    points = np.stack([x, y, z], axis=-1)[valid].astype(np.float32)
    region = None
    if segmap is not None and segmap_id is not None:
        region = (segmap == segmap_id)[valid]
    return points, region


def _grasp_group_to_arrays(gg: Any, top_k: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert graspnetAPI GraspGroup -> (poses Nx4x4, scores, widths, translations)."""
    if gg is None or len(gg) == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    gg = gg.nms().sort_by_score()
    if top_k > 0:
        gg = gg[:top_k]
    n = len(gg)
    poses = np.zeros((n, 4, 4), dtype=np.float32)
    scores = np.zeros((n,), dtype=np.float32)
    widths = np.zeros((n,), dtype=np.float32)
    translations = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        g = gg[i]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.asarray(g.rotation_matrix, dtype=np.float32)
        T[:3, 3] = np.asarray(g.translation, dtype=np.float32)
        # Tip offset along grasp approach (X in graspnet frame)
        depth = float(getattr(g, "depth", 0.0))
        tip = T[:3, 3] + depth * T[:3, 0]
        T[:3, 3] = tip
        poses[i] = T
        scores[i] = float(g.score)
        widths[i] = float(g.width)
        translations[i] = tip
    return poses, scores, widths, translations


def _mock_grasps(points: np.ndarray, region: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic top-down grasp at the (masked) cloud centroid for HTTP smoke tests."""
    pts = points[region] if region is not None and region.any() else points
    if pts.shape[0] == 0:
        pts = points
    if pts.shape[0] == 0:
        return (
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    center = pts.mean(axis=0)
    top_z = float(pts[:, 2].max())
    T = np.eye(4, dtype=np.float32)
    # Approach along +Z (camera), fingers along Y — rough top-down in cam frame
    T[:3, :3] = np.array(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        dtype=np.float32,
    )
    T[:3, 3] = np.array([center[0], center[1], top_z], dtype=np.float32)
    return (
        T[None, ...],
        np.array([1.0], dtype=np.float32),
        np.array([0.08], dtype=np.float32),
        T[:3, 3][None, ...],
    )


# --- API Models ---


class PlanRequest(BaseModel):
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str | None = None
    segmap_id: int = 1
    z_range: list[float] | None = None
    dense_grasp: bool = False
    collision_detection: bool = True
    approach_steering: list[float] | None = None
    approach_thresh: float = 3.141592653589793
    top_k: int = 50


class PlanPointsRequest(BaseModel):
    points_base64: str
    region_mask_base64: str | None = None
    dense_grasp: bool = False
    collision_detection: bool = True
    approach_steering: list[float] | None = None
    approach_thresh: float = 3.141592653589793
    top_k: int = 50


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    widths_base64: str
    contact_pts_base64: str
    mock: bool = False


class HealthResponse(BaseModel):
    status: str
    mock: bool
    device: str
    sdk_root: str
    feature_id: str | None = None
    checkpoint: str | None = None


def _do_plan_points(
    points: np.ndarray,
    region: np.ndarray | None,
    dense_grasp: bool,
    collision_detection: bool,
    approach_steering: list[float] | None,
    approach_thresh: float,
    top_k: int,
) -> PlanResponse:
    if _MOCK or _DETECTOR is None:
        poses, scores, widths, tips = _mock_grasps(points, region)
        return PlanResponse(
            grasps_base64=_numpy_to_base64(poses),
            scores_base64=_numpy_to_base64(scores),
            widths_base64=_numpy_to_base64(widths),
            contact_pts_base64=_numpy_to_base64(tips),
            mock=True,
        )

    optional: dict[str, Any] = {
        "dense_grasp": dense_grasp,
        "collision_detection": collision_detection,
        "region_steering": region,
        "approach_steering": approach_steering,
        "approach_thresh": approach_thresh,
    }
    gg = _DETECTOR.get_grasp(points, optional)
    poses, scores, widths, tips = _grasp_group_to_arrays(gg, top_k=top_k)
    return PlanResponse(
        grasps_base64=_numpy_to_base64(poses),
        scores_base64=_numpy_to_base64(scores),
        widths_base64=_numpy_to_base64(widths),
        contact_pts_base64=_numpy_to_base64(tips),
        mock=False,
    )


def _do_plan(req: PlanRequest) -> PlanResponse:
    depth = _base64_to_numpy(req.depth_base64)
    cam_K = _base64_to_numpy(req.cam_K_base64)
    segmap = _base64_to_numpy(req.segmap_base64) if req.segmap_base64 else None
    points, region = _depth_to_points(
        depth, cam_K, segmap=segmap, segmap_id=req.segmap_id, z_range=req.z_range
    )
    if points.shape[0] == 0:
        empty = np.zeros((0, 4, 4), dtype=np.float32)
        z = np.zeros((0,), dtype=np.float32)
        return PlanResponse(
            grasps_base64=_numpy_to_base64(empty),
            scores_base64=_numpy_to_base64(z),
            widths_base64=_numpy_to_base64(z),
            contact_pts_base64=_numpy_to_base64(np.zeros((0, 3), dtype=np.float32)),
            mock=_MOCK or _DETECTOR is None,
        )
    return _do_plan_points(
        points,
        region,
        req.dense_grasp,
        req.collision_detection,
        req.approach_steering,
        req.approach_thresh,
        req.top_k,
    )


def _do_plan_cloud(req: PlanPointsRequest) -> PlanResponse:
    points = _base64_to_numpy(req.points_base64).astype(np.float32)
    region = None
    if req.region_mask_base64:
        region = _base64_to_numpy(req.region_mask_base64).astype(bool)
    return _do_plan_points(
        points,
        region,
        req.dense_grasp,
        req.collision_detection,
        req.approach_steering,
        req.approach_thresh,
        req.top_k,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    feature_id = None
    try:
        if str(_DETECT_DIR) not in sys.path:
            sys.path.insert(0, str(_DETECT_DIR))
        from gsnet import get_feature_id  # type: ignore

        feature_id = str(get_feature_id())
    except Exception:
        pass
    ready = _MOCK or _DETECTOR is not None
    return HealthResponse(
        status="ready" if ready else "unavailable",
        mock=_MOCK,
        device=_DEVICE,
        sdk_root=str(_SDK_ROOT),
        feature_id=feature_id,
        checkpoint=os.environ.get("ANYGRASP_CHECKPOINT"),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest):
    if not _MOCK and _DETECTOR is None:
        raise HTTPException(status_code=503, detail="AnyGrasp detector not initialized")
    try:
        return await _run_on_gpu(_do_plan, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AnyGrasp /plan failed")
        raise HTTPException(status_code=500, detail=f"AnyGrasp planning failed: {e}") from e


@app.post("/plan_points", response_model=PlanResponse)
async def plan_points_endpoint(req: PlanPointsRequest):
    if not _MOCK and _DETECTOR is None:
        raise HTTPException(status_code=503, detail="AnyGrasp detector not initialized")
    try:
        return await _run_on_gpu(_do_plan_cloud, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AnyGrasp /plan_points failed")
        raise HTTPException(status_code=500, detail=f"AnyGrasp planning failed: {e}") from e


def _init_detector(
    checkpoint_path: Path,
    license_dir: Path,
    max_gripper_width: float,
    gripper_height: float,
) -> Any:
    _ensure_gsnet_so()
    if str(_DETECT_DIR) not in sys.path:
        sys.path.insert(0, str(_DETECT_DIR))

    if license_dir.is_dir():
        try:
            from gsnet import check_license  # type: ignore

            check_license(str(license_dir))
            logger.info("AnyGrasp license OK: %s", license_dir)
        except Exception as e:
            logger.warning("License check failed (%s); continuing to create_detector", e)

    from gsnet import create_detector  # type: ignore

    cfg = Namespace(
        checkpoint_path=str(checkpoint_path),
        max_gripper_width=max(0.0, min(0.1, max_gripper_width)),
        gripper_height=gripper_height,
    )
    detector = create_detector(cfg)
    if detector is None:
        raise RuntimeError(
            "create_detector returned None — usually missing/invalid license or checkpoint. "
            "Apply for a license (see license_registration/README.md) or start with --mock."
        )
    return detector


def main(
    device: str = "cuda",
    port: int = 8120,
    host: str = "127.0.0.1",
    checkpoint_path: str | None = None,
    license_dir: str | None = None,
    max_gripper_width: float = 0.1,
    gripper_height: float = 0.03,
    mock: bool = False,
):
    """Launch AnyGrasp FastAPI service.

    Args:
        device: Device label reported in /health (SDK uses CUDA when available).
        port: Listen port (default 8120).
        host: Bind address.
        checkpoint_path: Path to checkpoint_detection.tar from the license package.
        license_dir: Directory containing licenseCfg.json + key/signature/lic.
        max_gripper_width: Max gripper opening in meters (<=0.1).
        gripper_height: Finger height in meters for collision checks.
        mock: If True, skip SDK load and return geometric mock grasps.
    """
    global _DETECTOR, _MOCK, _DEVICE
    _DEVICE = device
    _MOCK = mock

    ckpt = Path(checkpoint_path or os.environ.get("ANYGRASP_CHECKPOINT", _DEFAULT_CHECKPOINT))
    lic = Path(license_dir or os.environ.get("ANYGRASP_LICENSE_DIR", _DEFAULT_LICENSE))
    os.environ["ANYGRASP_CHECKPOINT"] = str(ckpt)

    if mock:
        logger.warning("Starting AnyGrasp in MOCK mode (no licensed SDK inference)")
        _DETECTOR = None
    elif not ckpt.is_file():
        logger.warning(
            "Checkpoint not found at %s — falling back to MOCK mode. "
            "Place checkpoint_detection.tar there or pass --checkpoint-path after license approval.",
            ckpt,
        )
        _MOCK = True
        _DETECTOR = None
    else:
        logger.info("Loading AnyGrasp from %s (checkpoint=%s)", _DETECT_DIR, ckpt)
        try:
            _DETECTOR = _init_detector(ckpt, lic, max_gripper_width, gripper_height)
            logger.info("AnyGrasp detector ready")
        except Exception as e:
            logger.warning("AnyGrasp init failed (%s); falling back to MOCK mode", e)
            _MOCK = True
            _DETECTOR = None

    logger.info("AnyGrasp service listening on %s:%s (mock=%s)", host, port, _MOCK)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
