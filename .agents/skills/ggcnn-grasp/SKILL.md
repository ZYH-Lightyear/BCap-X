---
name: ggcnn-grasp
description: Use GG-CNN (local port 8119) to propose fast antipodal grasps from a depth image when Contact-GraspNet is unavailable or a lightweight depth-only grasp planner is preferred.
---

# GG-CNN Grasp Planning

Plan 2D antipodal grasps from a single depth image via the local GG-CNN service
([dougsm/ggcnn](https://github.com/dougsm/ggcnn)).

## Service

```bash
# Pretrained Cornell GG-CNN2 weights under capx/third_party/ggcnn/weights/
python -m capx.serving.launch_ggcnn_server --device cuda --port 8119 --host 127.0.0.1

# Or via unified launcher
python -m capx.serving.launch_servers --profile ggcnn
```

Override URL with `export GGCNN_SERVICE_URL=http://127.0.0.1:8119`.

## Client APIs

```python
from capx.integrations.vision.ggcnn import init_ggcnn, health_check

assert health_check()
plan = init_ggcnn()

result = plan(
    depth=depth_hw,          # float32 HxW meters
    cam_K=K_3x3,             # optional; enables 4x4 camera poses
    segmap=mask_hw,          # optional int mask
    segmap_id=1,
    n_grasps=5,
)
grasps = result["grasps"]    # row, col, angle, width_px, quality, pose?
poses = result["poses"]      # (N,4,4) if cam_K given
scores = result["scores"]
```

## Workflow

1. Obtain depth (and optional object mask) from the current observation.
2. Call `plan(...)`; prefer candidates with higher `quality`.
3. If `poses` is non-empty, convert the best camera-frame pose into the robot
   base frame and execute approach → close → lift.
4. If no peaks are returned, lower `threshold_abs` (e.g. `0.05`) or clear the
   mask and retry once.

## Pitfalls

- Input should be metric depth; zeros are inpainted by default.
- Pixel width (`width_px` / `length_px`) is in the original image scale; set
  `width_scale_m` only if you know meters-per-pixel at the grasp depth.
- GG-CNN is depth-only and lighter than Contact-GraspNet; prefer GraspNet for
  cluttered 6-DoF scenes when that service is up.

## Related Skills

Use `$libero-segmentation-to-points` for masks/points, `$libero-grasp-object` for
the Contact-GraspNet path, and `$libero-motion-control` for execution.
