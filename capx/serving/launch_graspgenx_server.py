"""GraspGenX HTTP inference server (CapX local-port pattern).

Cross-embodiment grasp generation — one model, gripper conditioned by swept
volume. Mirrors Contact-GraspNet / GraspGen FastAPI style on port **8123**.

Launch (from CapX root, with the GraspGenX uv venv)::

    capx/third_party/GraspGenX/.venv/bin/python -m capx.serving.launch_graspgenx_server \\
        --host 127.0.0.1 --port 8123 \\
        --default-gripper franka_panda
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="GraspGenX", version="1.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CFG: Any | None = None
_SHARED_MODEL: Any | None = None
_ASSETS_DIR: str = ""
_DEFAULT_GRIPPER: str = "franka_panda"
_SAMPLERS: dict[str, Any] = {}
_SAMPLERS_LOCK = threading.Lock()
_METADATA: dict[str, Any] = {}
_GPU_SEMAPHORE = asyncio.Semaphore(1)


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
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}")


def _get_sampler(gripper_name: str):
    from graspgenx.grasp_server import GraspGenXSampler

    with _SAMPLERS_LOCK:
        sampler = _SAMPLERS.get(gripper_name)
        if sampler is None:
            logger.info("Loading GraspGenX sampler for gripper=%s", gripper_name)
            sampler = GraspGenXSampler(
                _CFG,
                gripper_name=gripper_name,
                assets_dir=_ASSETS_DIR,
                model=_SHARED_MODEL,
            )
            _SAMPLERS[gripper_name] = sampler
        return sampler


class InferRequest(BaseModel):
    pc_base64: str
    gripper_name: str | None = None
    grasp_threshold: float = -1.0
    num_grasps: int = 200
    topk_num_grasps: int = 100
    min_grasps: int = 40
    max_tries: int = 6
    remove_outliers: bool = True


class InferResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    num_grasps: int
    infer_ms: float
    gripper_name: str


class HealthResponse(BaseModel):
    status: str
    default_gripper: str | None = None
    loaded_grippers: list[str] | None = None
    checkpoint_root: str | None = None


class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    gripper_name: str | None = None
    grasp_threshold: float = -1.0
    num_grasps: int = 200
    topk_num_grasps: int = 100
    min_grasps: int = 40
    max_tries: int = 6
    remove_outliers: bool = True


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    contact_pts_base64: str


def _do_infer(req: InferRequest) -> InferResponse:
    from graspgenx.grasp_server import GraspGenXSampler

    pc = _base64_to_numpy(req.pc_base64).astype(np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise HTTPException(
            status_code=400, detail=f"point cloud must be (N, 3), got {pc.shape}"
        )

    gripper_name = req.gripper_name or _DEFAULT_GRIPPER
    sampler = _get_sampler(gripper_name)

    t0 = time.monotonic()
    grasps, scores = GraspGenXSampler.run_inference(
        pc,
        sampler,
        grasp_threshold=req.grasp_threshold,
        num_grasps=req.num_grasps,
        topk_num_grasps=req.topk_num_grasps,
        min_grasps=req.min_grasps,
        max_tries=req.max_tries,
        remove_outliers=req.remove_outliers,
    )
    infer_ms = (time.monotonic() - t0) * 1000.0

    if hasattr(grasps, "cpu"):
        grasps_np = grasps.cpu().numpy().astype(np.float32)
        scores_np = scores.cpu().numpy().astype(np.float32)
    else:
        grasps_np = np.asarray(grasps, dtype=np.float32)
        scores_np = np.asarray(scores, dtype=np.float32)

    if grasps_np.size == 0:
        grasps_np = np.empty((0, 4, 4), dtype=np.float32)
        scores_np = np.empty((0,), dtype=np.float32)

    return InferResponse(
        grasps_base64=_numpy_to_base64(grasps_np),
        scores_base64=_numpy_to_base64(scores_np),
        num_grasps=int(grasps_np.shape[0]),
        infer_ms=float(infer_ms),
        gripper_name=gripper_name,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _SHARED_MODEL is None:
        return HealthResponse(status="loading")
    return HealthResponse(
        status="ok",
        default_gripper=_DEFAULT_GRIPPER,
        loaded_grippers=sorted(_SAMPLERS.keys()),
        checkpoint_root=_METADATA.get("checkpoint_root"),
    )


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    if _SHARED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        **_METADATA,
        "default_gripper": _DEFAULT_GRIPPER,
        "loaded_grippers": sorted(_SAMPLERS.keys()),
    }


@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(req: InferRequest):
    if _SHARED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_on_gpu(_do_infer, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GraspGenX inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"GraspGenX inference failed: {e}")


@app.post("/plan_point_clouds", response_model=PlanResponse)
async def plan_point_clouds_endpoint(req: PlanPointCloudsRequest):
    """Plan grasps for a segmented object PC (uses pc_segment; pc_full ignored)."""
    if _SHARED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    infer_req = InferRequest(
        pc_base64=req.pc_segment_base64,
        gripper_name=req.gripper_name,
        grasp_threshold=req.grasp_threshold,
        num_grasps=req.num_grasps,
        topk_num_grasps=req.topk_num_grasps,
        min_grasps=req.min_grasps,
        max_tries=req.max_tries,
        remove_outliers=req.remove_outliers,
    )
    try:
        result = await _run_on_gpu(_do_infer, infer_req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GraspGenX plan_point_clouds failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    grasps = _base64_to_numpy(result.grasps_base64)
    if grasps.size == 0:
        contact = np.empty((0, 3), dtype=np.float32)
    else:
        contact = grasps[:, :3, 3].astype(np.float32)

    return PlanResponse(
        grasps_base64=result.grasps_base64,
        scores_base64=result.scores_base64,
        contact_pts_base64=_numpy_to_base64(contact),
    )


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "GraspGenX"


def main(
    checkpoint_root: str = "",
    assets_dir: str = "",
    default_gripper: str = "franka_panda",
    device: str = "cuda",
    port: int = 8123,
    host: str = "127.0.0.1",
) -> None:
    global _CFG, _SHARED_MODEL, _ASSETS_DIR, _DEFAULT_GRIPPER, _METADATA

    vendor = _vendor_root()
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    if device.startswith("cuda"):
        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        )

    # Import triggers auto-clone of checkpoints + gripper_descriptions into ext/
    import graspgenx  # noqa: F401
    from graspgenx import get_checkpoints_version_dir
    from graspgenx.grasp_server import load_grasp_gen_model
    from graspgenx.utils.checkpoint_io import load_model_cfg

    if not checkpoint_root:
        checkpoint_root = os.environ.get("GRASPGENX_CHECKPOINT_DIR", "")
        if checkpoint_root:
            # Env may point at the HF repo root (with release/) or the version dir.
            cand = Path(checkpoint_root)
            if (cand / "release" / "gen").is_dir():
                checkpoint_root = str(cand / "release")
            elif not (cand / "gen").is_dir():
                checkpoint_root = str(get_checkpoints_version_dir())
        else:
            checkpoint_root = str(get_checkpoints_version_dir())

    ckpt = Path(checkpoint_root).expanduser().resolve()
    if not (ckpt / "gen").is_dir() or not (ckpt / "dis").is_dir():
        raise FileNotFoundError(
            f"Checkpoint root must contain gen/ and dis/: {ckpt}. "
            "First import of graspgenx auto-clones "
            "https://huggingface.co/adithyamurali/GraspGenXModel into "
            "capx/third_party/GraspGenX/ext/graspgenx_checkpoints."
        )

    if not assets_dir:
        assets_dir = str(vendor / "assets")
    _ASSETS_DIR = assets_dir
    _DEFAULT_GRIPPER = default_gripper

    logger.info("Loading GraspGenX cfg from %s", ckpt)
    _CFG = load_model_cfg(str(ckpt / "gen"), str(ckpt / "dis"))
    logger.info("Loading shared GraspGenX model weights…")
    _SHARED_MODEL = load_grasp_gen_model(_CFG)

    _METADATA = {
        "checkpoint_root": str(ckpt),
        "assets_dir": _ASSETS_DIR,
        "default_gripper": _DEFAULT_GRIPPER,
        "gen_checkpoint": str(_CFG.eval.gen_checkpoint),
        "dis_checkpoint": str(_CFG.eval.dis_checkpoint),
    }

    # Eagerly warm the default gripper sampler
    _get_sampler(_DEFAULT_GRIPPER)
    logger.info("GraspGenX ready on %s:%d (gripper=%s)", host, port, _DEFAULT_GRIPPER)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
