"""GraspGen HTTP inference server (CapX local-port pattern).

Mirrors Contact-GraspNet's FastAPI style on a dedicated port (default 8119).
Requires the GraspGen package + checkpoints (see third_party/GraspGen).

Launch (from CapX root, with GraspGen venv or PYTHONPATH set)::

    /path/to/GraspGen/.venv/bin/python -m capx.serving.launch_graspgen_server \\
        --port 8121 --host 127.0.0.1 \\
        --gripper-config /path/to/GraspGenModels/checkpoints/graspgen_franka_panda.yml
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Service Configuration ---
app = FastAPI(title="GraspGen", version="1.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SAMPLER: Any | None = None
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


class InferRequest(BaseModel):
    pc_base64: str
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


class HealthResponse(BaseModel):
    status: str
    gripper_name: str | None = None
    model_name: str | None = None


def _do_infer(req: InferRequest) -> InferResponse:
    from grasp_gen.grasp_server import GraspGenSampler

    pc = _base64_to_numpy(req.pc_base64).astype(np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise HTTPException(
            status_code=400, detail=f"point cloud must be (N, 3), got {pc.shape}"
        )

    import time

    t0 = time.monotonic()
    grasps, scores = GraspGenSampler.run_inference(
        pc,
        _SAMPLER,
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
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _SAMPLER is None:
        return HealthResponse(status="loading")
    return HealthResponse(
        status="ok",
        gripper_name=_METADATA.get("gripper_name"),
        model_name=_METADATA.get("model_name"),
    )


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    if _SAMPLER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return _METADATA


@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(req: InferRequest):
    if _SAMPLER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_on_gpu(_do_infer, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GraspGen inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"GraspGen inference failed: {e}")


# CapX Contact-GraspNet-compatible alias for point-cloud planning
class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
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


@app.post("/plan_point_clouds", response_model=PlanResponse)
async def plan_point_clouds_endpoint(req: PlanPointCloudsRequest):
    """Plan grasps for a segmented object PC (uses pc_segment; pc_full ignored)."""
    if _SAMPLER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    infer_req = InferRequest(
        pc_base64=req.pc_segment_base64,
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
        logger.error("GraspGen plan_point_clouds failed: %s", e)
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


def _default_gripper_config() -> str:
    env = os.environ.get("GRASPGEN_GRIPPER_CONFIG")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve()
    models = here.parents[1] / "third_party" / "GraspGenModels" / "checkpoints"
    for name in (
        "graspgen_franka_panda.yml",
        "graspgen_robotiq_2f_140.yml",
        "graspgen_single_suction_cup_30mm.yml",
    ):
        cand = models / name
        if cand.exists():
            return str(cand)
    return str(models / "graspgen_franka_panda.yml")


def main(
    gripper_config: str = "",
    device: str = "cuda",
    port: int = 8121,
    host: str = "127.0.0.1",
) -> None:
    global _SAMPLER, _METADATA

    # Ensure GraspGen package is importable
    vendor = Path(__file__).resolve().parents[1] / "third_party" / "GraspGen"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    if not gripper_config:
        gripper_config = _default_gripper_config()
    if not Path(gripper_config).exists():
        raise FileNotFoundError(
            f"Gripper config not found: {gripper_config}. "
            "Clone https://huggingface.co/adithyamurali/GraspGenModels "
            "into capx/third_party/GraspGenModels or pass --gripper-config."
        )

    if device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    logger.info("Loading GraspGen config from %s", gripper_config)
    cfg = load_grasp_cfg(gripper_config)
    _METADATA = {
        "gripper_name": cfg.data.gripper_name,
        "model_name": cfg.eval.model_name,
        "gripper_config": gripper_config,
    }
    logger.info(
        "Initializing GraspGenSampler (model=%s, gripper=%s)",
        _METADATA["model_name"],
        _METADATA["gripper_name"],
    )
    _SAMPLER = GraspGenSampler(cfg)
    logger.info("GraspGen ready on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
