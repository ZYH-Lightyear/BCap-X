---
name: anygrasp-grasp
description: Deploy and use the AnyGrasp local HTTP grasp service in CaP-X (port 8120). Use when the user mentions AnyGrasp, anygrasp_sdk, GSNet grasp detection, or replacing/augmenting Contact-GraspNet with AnyGrasp.
---

# AnyGrasp Grasp Service

AnyGrasp ([graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)) runs as a local FastAPI service on **port 8120**, same pattern as SAM3 (:8114), Contact-GraspNet (:8115), and GG-CNN (:8119).

## Layout

| Path | Role |
| ---- | ---- |
| `capx/third_party/anygrasp_sdk/` | Vendored SDK (git clone) |
| `capx/serving/launch_anygrasp_server.py` | FastAPI server |
| `capx/integrations/vision/anygrasp.py` | HTTP client (`ANYGRASP_SERVICE_URL`) |
| `tests/smoke_anygrasp_server.py` | Port smoke test |

## Clone (behind corporate proxy)

```bash
cd capx/third_party
export ALL_PROXY=http://accelerator-cname-hnpmnhnmdul3rmxrwhgend.c.vegalb.com:80
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone https://github.com/graspnet/anygrasp_sdk.git
```

## License (required for real inference)

1. Feature ID for this machine (current host: `N28151241625519219823`):

```bash
cd capx/third_party/anygrasp_sdk/grasp_detection
cp gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so gsnet.so
python -c "from gsnet import get_feature_id; print(get_feature_id())"
```

2. Apply at the [AnyGrasp license form](https://forms.gle/XVV3Eip8njTYJEBo6).
3. Unzip the returned package to `capx/third_party/anygrasp_sdk/grasp_detection/license/` and place `checkpoint_detection.tar` at `grasp_detection/log/checkpoint_detection.tar` (or set `ANYGRASP_CHECKPOINT` / `--checkpoint-path`).

Until the license arrives, start with `--mock` so HTTP contracts can still be tested.

## Launch

```bash
# Mock HTTP smoke (no license/checkpoint)
python -m capx.serving.launch_anygrasp_server --host 127.0.0.1 --port 8120 --mock

# Real SDK (after license + checkpoint + MinkowskiEngine / pointnet2 / graspnetAPI)
python -m capx.serving.launch_anygrasp_server --host 127.0.0.1 --port 8120 \
  --checkpoint-path capx/third_party/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar

# Via unified launcher profile
python capx/serving/launch_servers.py --profile anygrasp
```

Health check:

```bash
curl -s http://127.0.0.1:8120/health
```

Endpoints: `GET /health`, `POST /plan` (depth+K+segmap), `POST /plan_points` (XYZ cloud).

## Client usage

```python
from capx.integrations.vision.anygrasp import init_anygrasp, init_anygrasp_points

plan = init_anygrasp()
grasps, scores, widths = plan(depth, cam_K, segmap=seg, segmap_id=1)

plan_pc = init_anygrasp_points()
grasps, scores, widths = plan_pc(points_xyz, region_mask=mask)
```

Env override: `ANYGRASP_SERVICE_URL=http://127.0.0.1:8120`.

## Smoke test

```bash
python tests/smoke_anygrasp_server.py
```

## YAML snippet

```yaml
api_servers:
  - _target_: capx.serving.launch_anygrasp_server.main
    device: cuda
    port: 8120
    host: 127.0.0.1
    # mock: true   # until license is installed
```

## Pitfalls

- Closed-source `.so` is Python-version specific; CaP-X uses 3.10 → `gsnet.cpython-310-*.so`.
- Real inference needs MinkowskiEngine v0.5.4, `pointnet2`, and `graspnetAPI` (see SDK README).
- Grasp tip ≠ `translation`; server already applies `translation + depth * R[:,0]`.
- Gripper frame: approach = X, open/close = Y ([graspnetAPI docs](https://graspnetapi.readthedocs.io/en/latest/grasp_format.html)).
- Port **8120** (8119 is GG-CNN).
