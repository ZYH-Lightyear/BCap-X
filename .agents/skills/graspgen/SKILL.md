---
name: graspgen
description: >-
  Deploy and call NVIDIA GraspGen (diffusion 6-DOF grasp generation) as a CapX
  local-port FastAPI service. Use when installing GraspGen, launching the HTTP
  server on :8121, planning grasps from segmented object point clouds, choosing
  Franka / Robotiq / suction gripper checkpoints, or smoke-testing GraspGen
  alongside Contact-GraspNet (:8115). Also covers cloning GraspGenModels and
  the dedicated GraspGen uv venv.
---

# GraspGen (CapX local port)

GraspGen predicts ranked 6-DOF grasps `(K, 4, 4)` + scores `(K,)` from an
**object-centric point cloud** (segmented object PC, typically already centered
or in camera/world frame consistent with your planner).

Upstream: [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen). CapX vendors it
under `capx/third_party/GraspGen` and exposes **HTTP on port 8121** (same pattern
as Contact-GraspNet `:8115`, SAM3 `:8114`, PyRoKi `:8116`).

## Layout

| Path | Role |
|------|------|
| `capx/third_party/GraspGen` | Upstream code + `.venv` (torch 2.1 / py3.10) |
| `capx/third_party/GraspGenModels` | HF checkpoints + sample data |
| `capx/serving/launch_graspgen_server.py` | FastAPI server |
| `capx/integrations/vision/graspgen.py` | HTTP client |

## Install / clone (use proxy)

```bash
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
export http_proxy=$ALL_PROXY https_proxy=$ALL_PROXY

cd capx/third_party
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone --depth 1 \
  https://github.com/NVlabs/GraspGen.git
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone --depth 1 \
  https://huggingface.co/adithyamurali/GraspGenModels

cd GraspGen
uv python install 3.10 && uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
uv pip install fastapi uvicorn tyro pyzmq msgpack msgpack-numpy
./install_uv_pointnet.sh
```

Gripper YAML files live under `GraspGenModels/checkpoints/`:

- `graspgen_franka_panda.yml` (default for CapX Franka)
- `graspgen_robotiq_2f_140.yml`
- `graspgen_single_suction_cup_30mm.yml`

## Launch server

```bash
cd /path/to/BCap-X
export PYTHONPATH=$PWD:$PWD/capx/third_party/GraspGen
export CUDA_VISIBLE_DEVICES=0   # pick a free GPU

capx/third_party/GraspGen/.venv/bin/python -m capx.serving.launch_graspgen_server \
  --host 127.0.0.1 --port 8121 \
  --gripper-config capx/third_party/GraspGenModels/checkpoints/graspgen_franka_panda.yml
```

Or via registry:

```bash
# profile / YAML _target_: capx.serving.launch_graspgen_server.main
# default_port: 8121
```

Health: `curl http://127.0.0.1:8121/health` → `{"status":"ok",...}`  
Docs: `http://127.0.0.1:8121/docs`

## Client API

```python
from capx.integrations.vision.graspgen import init_graspgen, init_graspgen_point_clouds

infer = init_graspgen()
grasps, scores = infer(pc)  # pc: (N, 3) float32

plan = init_graspgen_point_clouds()
grasps, scores, contact_pts = plan(pc_full, pc_segment)
```

Env override: `GRASPGEN_SERVICE_URL=http://127.0.0.1:8121`.

HTTP endpoints:

| Method | Path | Body |
|--------|------|------|
| GET | `/health` | — |
| GET | `/metadata` | — |
| POST | `/infer` | `pc_base64`, `num_grasps`, `topk_num_grasps`, … |
| POST | `/plan_point_clouds` | `pc_full_base64`, `pc_segment_base64` (uses segment) |

Arrays are numpy `.npy` payloads base64-encoded (same as Contact-GraspNet).

## Smoke test

```bash
# After server is up:
python - <<'PY'
import numpy as np, requests, base64, io
pc = np.random.randn(512, 3).astype(np.float32) * 0.05
buf = io.BytesIO(); np.save(buf, pc)
b64 = base64.b64encode(buf.getvalue()).decode()
r = requests.post("http://127.0.0.1:8121/infer",
                  json={"pc_base64": b64, "num_grasps": 50, "topk_num_grasps": 10,
                        "min_grasps": 1, "max_tries": 2}, timeout=180)
r.raise_for_status()
print(r.json()["num_grasps"], r.json()["infer_ms"])
PY
```

## Workflow (with CapX perception)

1. Segment object (`$libero-segmentation-to-points` / SAM3) → world or camera `points`.
2. Call GraspGen `infer(points)` or `plan_grasp`-style via `init_graspgen_point_clouds`.
3. Pick top score / top-down filter; execute with `$libero-motion-control`.

## Pitfalls

- Run the **server** with GraspGen’s `.venv` (pinned torch 2.1); CapX client only needs `requests`.
- Checkpoints must sit next to the gripper YAML (HF layout); `load_grasp_cfg` rewrites relative paths.
- Object PC should be dense enough (~1k–2k pts); use `remove_outliers=True` for real depth.
- Port **8121** — do not collide with Contact-GraspNet **8115**.
- Native upstream ZMQ server (`client-server/graspgen_server.py`, port 5556) is optional; CapX uses HTTP.

## Related

- Contact-GraspNet CapX skill path: existing `plan_grasp` on `:8115`
- Upstream ZMQ / MCP: `GraspGen/client-server/`, `GraspGen/mcp/`
