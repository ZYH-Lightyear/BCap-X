#!/usr/bin/env python3
"""Smoke-test AnyGrasp local HTTP service on :8120.

Starts the server in --mock mode if nothing is listening, hits /health and
/plan + /plan_points with synthetic depth/point clouds, then tears down any
process it started.
"""

from __future__ import annotations

import base64
import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
HOST = os.environ.get("ANYGRASP_HOST", "127.0.0.1")
PORT = int(os.environ.get("ANYGRASP_PORT", "8120"))
BASE = f"http://{HOST}:{PORT}"


def _numpy_to_b64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _b64_to_numpy(s: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(s)) as f:
        return np.load(f)


def _tcp_up(timeout: float = 0.5) -> bool:
    import socket

    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_ready(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _tcp_up():
            try:
                r = requests.get(f"{BASE}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
        time.sleep(0.5)
    raise RuntimeError(f"AnyGrasp service not ready at {BASE} within {timeout}s")


def main() -> int:
    started: subprocess.Popen[str] | None = None
    if not _tcp_up():
        print(f"[smoke] nothing on {BASE}; launching mock server...")
        cmd = [
            sys.executable,
            "-m",
            "capx.serving.launch_anygrasp_server",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--mock",
        ]
        started = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
    else:
        print(f"[smoke] reusing existing service at {BASE}")

    try:
        _wait_ready()
        health = requests.get(f"{BASE}/health", timeout=5).json()
        print("[smoke] /health:", health)
        assert health.get("status") == "ready", health

        # Synthetic depth + intrinsics + blob segmap
        h, w = 120, 160
        depth = np.full((h, w), 0.8, dtype=np.float32)
        yy, xx = np.ogrid[:h, :w]
        blob = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < 25**2
        depth[blob] = 0.55
        cam_K = np.array([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1]], dtype=np.float32)
        seg = np.zeros((h, w), dtype=np.int32)
        seg[blob] = 1

        plan = requests.post(
            f"{BASE}/plan",
            json={
                "depth_base64": _numpy_to_b64(depth),
                "cam_K_base64": _numpy_to_b64(cam_K),
                "segmap_base64": _numpy_to_b64(seg),
                "segmap_id": 1,
                "z_range": [0.1, 2.0],
                "top_k": 5,
            },
            timeout=30,
        )
        plan.raise_for_status()
        pdata = plan.json()
        grasps = _b64_to_numpy(pdata["grasps_base64"])
        scores = _b64_to_numpy(pdata["scores_base64"])
        print(f"[smoke] /plan -> grasps{grasps.shape} scores{scores.shape} mock={pdata.get('mock')}")
        assert grasps.ndim == 3 and grasps.shape[-2:] == (4, 4), grasps.shape
        assert grasps.shape[0] >= 1, "expected at least one mock grasp"

        # Point-cloud path
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
        pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)[blob].astype(np.float32)
        region = np.ones((pts.shape[0],), dtype=bool)
        plan_pc = requests.post(
            f"{BASE}/plan_points",
            json={
                "points_base64": _numpy_to_b64(pts),
                "region_mask_base64": _numpy_to_b64(region),
                "top_k": 5,
            },
            timeout=30,
        )
        plan_pc.raise_for_status()
        pcdata = plan_pc.json()
        grasps2 = _b64_to_numpy(pcdata["grasps_base64"])
        print(f"[smoke] /plan_points -> grasps{grasps2.shape} mock={pcdata.get('mock')}")
        assert grasps2.shape[0] >= 1

        # Client wrapper
        sys.path.insert(0, str(REPO))
        os.environ["ANYGRASP_SERVICE_URL"] = BASE
        from capx.integrations.vision.anygrasp import init_anygrasp, init_anygrasp_points

        g, s, widths = init_anygrasp()(depth, cam_K, segmap=seg, segmap_id=1)
        print(f"[smoke] client plan -> {g.shape} {s.shape} {widths.shape}")
        g2, s2, w2 = init_anygrasp_points()(pts, region_mask=region)
        print(f"[smoke] client plan_points -> {g2.shape} {s2.shape} {w2.shape}")

        print("[smoke] PASS")
        return 0
    finally:
        if started is not None and started.poll() is None:
            started.send_signal(signal.SIGTERM)
            try:
                started.wait(timeout=10)
            except subprocess.TimeoutExpired:
                started.kill()


if __name__ == "__main__":
    raise SystemExit(main())
