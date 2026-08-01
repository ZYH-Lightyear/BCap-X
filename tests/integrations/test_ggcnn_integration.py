"""Integration smoke test for the local GG-CNN FastAPI service.

Requires the server to be running, e.g.::

    python -m capx.serving.launch_ggcnn_server --device cuda --port 8119

Or::

    python -m capx.serving.launch_servers --profile ggcnn
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import requests

# Load the client module directly to avoid importing capx.integrations.__init__
# (which pulls in robosuite / all APIs and can take a long time).
_CLIENT_PATH = Path(__file__).resolve().parents[2] / "capx" / "integrations" / "vision" / "ggcnn.py"
_spec = importlib.util.spec_from_file_location("ggcnn_client", _CLIENT_PATH)
assert _spec is not None and _spec.loader is not None
_ggcnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ggcnn)

SERVICE_URL = _ggcnn.SERVICE_URL
health_check = _ggcnn.health_check
init_ggcnn = _ggcnn.init_ggcnn


def _service_up() -> bool:
    return health_check(timeout=2.0)


@pytest.mark.skipif(not _service_up(), reason=f"GG-CNN service not up at {SERVICE_URL}")
def test_ggcnn_health():
    resp = requests.get(f"{SERVICE_URL}/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data


@pytest.mark.skipif(not _service_up(), reason=f"GG-CNN service not up at {SERVICE_URL}")
def test_ggcnn_plan_synthetic_depth():
    # Synthetic table + raised box so Q peaks are meaningful enough for smoke.
    h, w = 480, 640
    depth = np.full((h, w), 0.85, dtype=np.float32)
    depth[180:300, 220:400] = 0.70
    # add a bit of noise
    depth += np.random.default_rng(0).normal(0, 0.002, size=depth.shape).astype(np.float32)

    cam_K = np.array(
        [
            [600.0, 0.0, w / 2.0],
            [0.0, 600.0, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    plan = init_ggcnn()
    result = plan(depth=depth, cam_K=cam_K, n_grasps=3, return_maps=True, threshold_abs=0.05)

    assert "grasps" in result
    assert "scores" in result
    assert "poses" in result
    assert "q" in result
    assert result["q"].shape == (300, 300)
    # Should find at least one candidate on this synthetic scene with low threshold
    assert len(result["grasps"]) >= 1
    g0 = result["grasps"][0]
    assert "row" in g0 and "col" in g0 and "angle" in g0 and "quality" in g0
    if result["poses"].size:
        assert result["poses"].shape[1:] == (4, 4)


def test_ggcnn_client_imports():
    """Client module should import without a live server."""
    assert callable(init_ggcnn())
    assert SERVICE_URL


if __name__ == "__main__":
    if not _service_up():
        print(f"[SKIP] GG-CNN service not reachable at {SERVICE_URL}", file=sys.stderr)
        print(
            "Start with: python -m capx.serving.launch_ggcnn_server --port 8119",
            file=sys.stderr,
        )
        sys.exit(0)

    test_ggcnn_health()
    test_ggcnn_plan_synthetic_depth()
    print("GG-CNN integration smoke tests PASSED")
