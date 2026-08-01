"""GG-CNN grasp synthesis FastAPI service.

Serves the Generative Grasping CNN from
https://github.com/dougsm/ggcnn on a local port (default 8119).

Endpoints:
  GET  /health  - readiness probe
  POST /plan    - predict antipodal grasps from a depth image
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import sys
from typing import Any

import cv2
import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from skimage.filters import gaussian
from skimage.transform import resize

# ---------------------------------------------------------------------------
# Logging / app
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GG-CNN Grasp Service")

_MODEL: Any | None = None
_DEVICE: str = "cuda"
_OUTPUT_SIZE: int = 300
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Depth preprocessing + grasp postprocess
# ---------------------------------------------------------------------------


def _inpaint_depth(depth: np.ndarray, missing_value: float = 0.0) -> np.ndarray:
    """Inpaint missing depth values (same approach as dougsm/ggcnn DepthImage)."""
    img = cv2.copyMakeBorder(depth.astype(np.float32), 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)
    scale = float(np.abs(img).max()) or 1.0
    img = img / scale
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
    return img[1:-1, 1:-1] * scale


def _preprocess_depth(
    depth: np.ndarray,
    output_size: int,
    inpaint: bool = True,
) -> tuple[np.ndarray, float, float, tuple[int, int]]:
    """Resize + mean-center depth for GG-CNN.

    Returns:
        network_input (1, H, W), scale_y, scale_x, original_hw
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")

    orig_h, orig_w = depth.shape
    depth = depth.astype(np.float32)
    if inpaint and np.any(depth == 0):
        depth = _inpaint_depth(depth)

    depth_rs = resize(depth, (output_size, output_size), preserve_range=True).astype(np.float32)
    depth_rs = np.clip(depth_rs - depth_rs.mean(), -1.0, 1.0)

    scale_y = orig_h / float(output_size)
    scale_x = orig_w / float(output_size)
    return depth_rs[None, ...], scale_y, scale_x, (orig_h, orig_w)


def _post_process_output(
    q_img: torch.Tensor,
    cos_img: torch.Tensor,
    sin_img: torch.Tensor,
    width_img: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = q_img.detach().cpu().numpy().squeeze()
    ang = (torch.atan2(sin_img, cos_img) / 2.0).detach().cpu().numpy().squeeze()
    width = width_img.detach().cpu().numpy().squeeze() * 150.0

    q = gaussian(q, 2.0, preserve_range=True)
    ang = gaussian(ang, 2.0, preserve_range=True)
    width = gaussian(width, 1.0, preserve_range=True)
    return q, ang, width


def _detect_grasps(
    q_img: np.ndarray,
    ang_img: np.ndarray,
    width_img: np.ndarray,
    no_grasps: int = 5,
    min_distance: int = 20,
    threshold_abs: float = 0.2,
) -> list[dict[str, float]]:
    from skimage.feature import peak_local_max

    peaks = peak_local_max(
        q_img,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        num_peaks=no_grasps,
    )
    grasps: list[dict[str, float]] = []
    for y, x in peaks:
        length = float(width_img[y, x])
        grasps.append(
            {
                "row": float(y),
                "col": float(x),
                "angle": float(ang_img[y, x]),
                "width_px": float(length / 2.0),
                "length_px": length,
                "quality": float(q_img[y, x]),
            }
        )
    return grasps


def _pixel_to_camera_pose(
    row: float,
    col: float,
    angle: float,
    depth: np.ndarray,
    cam_K: np.ndarray,
) -> np.ndarray | None:
    """Build a 4x4 camera-frame grasp pose (approach along +Z_cam)."""
    h, w = depth.shape
    r = int(np.clip(round(row), 0, h - 1))
    c = int(np.clip(round(col), 0, w - 1))
    z = float(depth[r, c])
    if z <= 0:
        # search neighbourhood for valid depth
        found = False
        for rad in range(1, 6):
            ys = slice(max(0, r - rad), min(h, r + rad + 1))
            xs = slice(max(0, c - rad), min(w, c + rad + 1))
            patch = depth[ys, xs]
            valid = patch[patch > 0]
            if valid.size:
                z = float(np.median(valid))
                found = True
                break
        if not found:
            return None

    fx, fy = float(cam_K[0, 0]), float(cam_K[1, 1])
    cx, cy = float(cam_K[0, 2]), float(cam_K[1, 2])
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy

    # Gripper frame: Z approach (camera forward), X along jaw closing (rotated by angle)
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array(
        [
            [ca, -sa, 0.0],
            [sa, ca, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return T


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    depth_base64: str
    """HxW depth image (meters preferred) encoded via np.save."""

    cam_K_base64: str | None = None
    """Optional 3x3 camera intrinsics for 3D pose backprojection."""

    segmap_base64: str | None = None
    """Optional HxW segmentation mask; non-matching pixels are zeroed in Q."""

    segmap_id: int = 1
    n_grasps: int = 5
    output_size: int = 300
    inpaint: bool = True
    min_distance: int = 20
    threshold_abs: float = 0.2
    width_scale_m: float = Field(
        default=0.0,
        description=(
            "If >0, convert length_px to meters via "
            "width_m = length_px * width_scale_m. "
            "Otherwise leave width in pixels (client may convert)."
        ),
    )


class GraspCandidate(BaseModel):
    row: float
    col: float
    angle: float
    width_px: float
    length_px: float
    quality: float
    width_m: float | None = None
    pose: list[list[float]] | None = None


class PlanResponse(BaseModel):
    grasps: list[GraspCandidate]
    q_base64: str | None = None
    ang_base64: str | None = None
    width_base64: str | None = None
    return_maps: bool = False


class HealthResponse(BaseModel):
    status: str
    device: str
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return HealthResponse(status="ok", device=_DEVICE, model=type(_MODEL).__name__)


def _do_plan(req: PlanRequest) -> PlanResponse:
    depth = _base64_to_numpy(req.depth_base64).astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")

    net_in, scale_y, scale_x, _orig_hw = _preprocess_depth(
        depth, output_size=req.output_size, inpaint=req.inpaint
    )

    if req.segmap_base64 is not None:
        seg = _base64_to_numpy(req.segmap_base64)
        if seg.shape[:2] != depth.shape[:2]:
            seg = resize(
                seg.astype(np.float32),
                depth.shape[:2],
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(seg.dtype)
        seg_rs = resize(
            (seg == req.segmap_id).astype(np.float32),
            (req.output_size, req.output_size),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )

    x = torch.from_numpy(net_in[None, ...]).to(_DEVICE)  # 1x1xHxW
    with torch.no_grad():
        pos, cos, sin, width = _MODEL(x)

    q_img, ang_img, width_img = _post_process_output(pos, cos, sin, width)

    if req.segmap_base64 is not None:
        q_img = q_img * seg_rs

    local_grasps = _detect_grasps(
        q_img,
        ang_img,
        width_img,
        no_grasps=req.n_grasps,
        min_distance=req.min_distance,
        threshold_abs=req.threshold_abs,
    )

    cam_K = None
    if req.cam_K_base64 is not None:
        cam_K = _base64_to_numpy(req.cam_K_base64).astype(np.float32)

    out: list[GraspCandidate] = []
    for g in local_grasps:
        # Map network coords back to original depth resolution
        row = g["row"] * scale_y
        col = g["col"] * scale_x
        length_px = g["length_px"] * ((scale_x + scale_y) * 0.5)
        width_px = length_px / 2.0
        width_m = float(length_px * req.width_scale_m) if req.width_scale_m > 0 else None

        pose_list = None
        if cam_K is not None:
            T = _pixel_to_camera_pose(row, col, g["angle"], depth, cam_K)
            if T is not None:
                pose_list = T.tolist()

        out.append(
            GraspCandidate(
                row=row,
                col=col,
                angle=g["angle"],
                width_px=width_px,
                length_px=length_px,
                quality=g["quality"],
                width_m=width_m,
                pose=pose_list,
            )
        )

    return PlanResponse(
        grasps=out,
        q_base64=_numpy_to_base64(q_img),
        ang_base64=_numpy_to_base64(ang_img),
        width_base64=_numpy_to_base64(width_img),
        return_maps=True,
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_on_gpu(_do_plan, req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GG-CNN plan failed")
        raise HTTPException(status_code=500, detail=f"GG-CNN plan failed: {e}") from e


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(
    device: str = "cuda",
    port: int = 8119,
    host: str = "127.0.0.1",
    network: str = "ggcnn2",
    weights: str | None = None,
    output_size: int = 300,
):
    """Launch the GG-CNN FastAPI service.

    Args:
        device: torch device string.
        port: listen port (default 8119).
        host: bind address.
        network: ``ggcnn`` or ``ggcnn2``.
        weights: path to state_dict ``.pt``; defaults to Cornell pretrained under
            ``capx/third_party/ggcnn/weights/``.
        output_size: network input spatial size.
    """
    global _MODEL, _DEVICE, _OUTPUT_SIZE

    _DEVICE = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    _OUTPUT_SIZE = output_size

    here = os.path.dirname(os.path.abspath(__file__))
    vendor_root = os.path.normpath(os.path.join(here, "..", "third_party", "ggcnn"))
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    if network.lower() == "ggcnn2":
        from models.ggcnn2 import GGCNN2 as _Net

        default_weights = os.path.join(
            vendor_root,
            "weights",
            "ggcnn2_weights_cornell",
            "epoch_50_cornell_statedict.pt",
        )
    else:
        from models.ggcnn import GGCNN as _Net

        default_weights = os.path.join(
            vendor_root,
            "weights",
            "ggcnn_weights_cornell",
            "ggcnn_epoch_23_cornell_statedict.pt",
        )

    weights_path = weights or default_weights
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"GG-CNN weights not found at {weights_path}. "
            "Download from https://github.com/dougsm/ggcnn/releases/tag/v0.1"
        )

    logger.info("Loading %s from %s on %s", network, weights_path, _DEVICE)
    model = _Net()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(_DEVICE)
    _MODEL = model

    logger.info("GG-CNN service ready on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
