---
name: graspgenx
description: >-
  Deploy and call NVIDIA GraspGenX (cross-embodiment 6-DOF grasp generation) as
  a CapX local-port FastAPI service. Use when installing GraspGenX, launching
  the HTTP server on :8123, planning grasps for any gripper (Franka, Robotiq,
  Unitree G1, Inspire, …), comparing GraspGenX to GraspGen (:8121) or
  Contact-GraspNet (:8115), or smoke-testing the CapX GraspGenX client.
---

# GraspGenX (CapX local port)

GraspGenX is a **cross-embodiment** grasp model: one checkpoint, conditioned on
a gripper's *swept volume*, predicts ranked 6-DOF grasps `(K, 4, 4)` + scores
`(K,)` for **any** gripper (including ones unseen in training).

Upstream: [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX). CapX vendors
it under `capx/third_party/GraspGenX` and exposes **HTTP on port 8123** (same
pattern as Contact-GraspNet `:8115`, GraspGen `:8121`, AnyGrasp `:8120`).

## Layout

| Path | Role |
|------|------|
| `capx/third_party/GraspGenX` | Upstream + `.venv` + auto-cloned `ext/` assets |
| `capx/third_party/GraspGenX/ext/graspgenx_checkpoints` | HF model weights |
| `capx/third_party/GraspGenX/ext/gripper_descriptions` | Gripper URDF + sweep-volume configs |
| `capx/serving/launch_graspgenx_server.py` | FastAPI server (`:8123`) |
| `capx/integrations/vision/graspgenx.py` | HTTP client |

## Install / clone (use proxy)

```bash
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export http_proxy=$ALL_PROXY https_proxy=$ALL_PROXY

cd capx/third_party
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone --depth 1 \
  https://github.com/NVlabs/GraspGenX.git

cd GraspGenX
# Optional: strip the tensorrt optional-dep if uv resolves it by mistake
uv sync --extra serve
uv pip install --python .venv/bin/python fastapi uvicorn tyro

# First import auto-clones HF checkpoints + gripper_descriptions into ext/
.venv/bin/python -c "import graspgenx; print('ok')"
```

## Launch server

```bash
cd /path/to/BCap-X
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1   # pick a free GPU

capx/third_party/GraspGenX/.venv/bin/python -m capx.serving.launch_graspgenx_server \
  --host 127.0.0.1 --port 8123 \
  --default-gripper franka_panda
```

Registry name: `graspgenx` / `_target_: capx.serving.launch_graspgenx_server.main`
(default_port **8123**).

Health: `curl http://127.0.0.1:8123/health` → `{"status":"ok",...}`  
Docs: `http://127.0.0.1:8123/docs`

## Client API

```python
from capx.integrations.vision.graspgenx import init_graspgenx, init_graspgenx_point_clouds

infer = init_graspgenx()
grasps, scores = infer(pc, gripper_name="franka_panda")  # pc: (N, 3) float32

plan = init_graspgenx_point_clouds()
grasps, scores, contact_pts = plan(pc_full, pc_segment, gripper_name="robotiq_2f_85")
```

Env override: `GRASPGENX_SERVICE_URL=http://127.0.0.1:8123`.

HTTP endpoints:

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| GET | `/metadata` | — |
| POST | `/infer` | `pc_base64`, `gripper_name`, `num_grasps`, `topk_num_grasps`, … |
| POST | `/plan_point_clouds` | `pc_segment_base64` (+ optional `gripper_name`) |

Arrays are numpy `.npy` payloads base64-encoded (same as Contact-GraspNet).

## Smoke test

```bash
python - <<'PY'
import numpy as np, requests, base64, io
pc = np.random.randn(1024, 3).astype(np.float32) * 0.04
buf = io.BytesIO(); np.save(buf, pc)
b64 = base64.b64encode(buf.getvalue()).decode()
r = requests.post("http://127.0.0.1:8123/infer",
                  json={"pc_base64": b64, "gripper_name": "franka_panda",
                        "num_grasps": 50, "topk_num_grasps": 10,
                        "min_grasps": 1, "max_tries": 2}, timeout=180)
r.raise_for_status()
print(r.json()["num_grasps"], r.json()["infer_ms"], r.json()["gripper_name"])
PY
```

## Workflow (with CapX perception)

1. Segment object (`$libero-segmentation-to-points` / SAM3) → object `points`.
2. Call GraspGenX `infer(points, gripper_name=…)`.
3. Pick top score / top-down filter; execute with `$libero-motion-control`.

## Pitfalls

- Run the **server** with GraspGenX’s `.venv` (torch 2.6+cu124); CapX client only needs `requests`.
- First import downloads ~1 GB+ checkpoints via git-lfs (needs proxy + `git-lfs`).
- Port **8123** — do not collide with GraspGen **8121** or Contact-GraspNet **8115**.
- Native upstream ZMQ server (`client-server/graspgenx_server.py`, port 5556) and MCP bridge remain optional; CapX uses HTTP.
- `uv sync` may try to pull the `tensorrt` extra on some uv versions — remove that optional-dep block or install with `uv sync --extra serve` after stripping it.

## Related

- GraspGen (per-gripper) CapX skill: `$graspgen` on `:8121`
- Contact-GraspNet: existing `plan_grasp` on `:8115`
- Upstream demos / wizard / MCP: `GraspGenX/scripts/`, `GraspGenX/mcp/`
