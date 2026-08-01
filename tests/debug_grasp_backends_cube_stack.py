#!/usr/bin/env python3
"""Run Contact-GraspNet (:8115), GG-CNN (:8119), GraspGen (:8121), GraspGenX (:8123)
on real robosuite cube_stack observations and save one overlay PNG each under
outputs/debug/.

Grasps are generated for **every** segmented object in the scene (red + green cubes).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.vision.grasp_backends import (  # noqa: E402
    health_check,
    init_propose_grasp_pose,
)

OUT_DIR = REPO / "outputs" / "debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-object viz colors: (best grasp, other grasps, mask tint RGB)
OBJ_COLORS = {
    "red_cube": ((0, 255, 80), (255, 180, 0), (255, 40, 40)),
    "green_cube": ((80, 200, 255), (180, 120, 255), (40, 220, 40)),
}


def _clean_blob(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    cy, cx = float(ys.mean()), float(xs.mean())
    dist2 = (ys - cy) ** 2 + (xs - cx) ** 2
    keep = dist2 < np.percentile(dist2, 92)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    cleaned[ys[keep], xs[keep]] = 1
    return cleaned


def _color_cube_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Heuristic instance masks for red + green cubes in robosuite Stack."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    red = _clean_blob((r > 140) & (r - g > 50) & (r - b > 50))
    green = _clean_blob((g > 140) & (g - r > 40) & (g - b > 40))
    return {"red_cube": red, "green_cube": green}


def _depth_hw(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def _depth_to_points(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None):
    from capx.utils.depth_utils import depth_to_pointcloud

    h, w = depth.shape
    full = depth_to_pointcloud(depth, K, filter_invalid=False).reshape(h, w, 3)
    valid = np.isfinite(full).all(axis=-1) & (full[..., 2] > 0.05) & (full[..., 2] < 2.5)
    if mask is not None:
        valid = valid & (mask.astype(bool))
    pts = full[valid].astype(np.float32)
    return pts, full, valid


# Franka/Panda parallel-jaw control points (Contact-GraspNet convention).
_PANDA_CTRL = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.052687433, -5.9955313e-05, 0.0584],
        [-0.052687433, 5.9955313e-05, 0.0584],
        [0.052687433, -5.9955313e-05, 0.10527314],
        [-0.052687433, 5.9955313e-05, 0.10527314],
    ],
    dtype=np.float64,
)


def _panda_wireframe(opening: float = 0.08) -> tuple[np.ndarray, list[tuple[int, int]]]:
    cp = _PANDA_CTRL.copy()
    half = float(np.clip(opening, 0.01, 0.10)) * 0.5
    cp[1:, 0] = np.sign(cp[1:, 0]) * half
    mid = 0.5 * (cp[1] + cp[2])
    pts = np.stack([cp[0], mid, cp[1], cp[3], cp[1], cp[2], cp[4]], axis=0)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    return pts, edges


def _project(xyz: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    if xyz[2] <= 1e-6:
        return None
    uvw = K @ xyz
    return int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))


def _draw_gripper(
    draw: ImageDraw.ImageDraw,
    T: np.ndarray,
    K: np.ndarray,
    *,
    opening: float = 0.08,
    color: tuple[int, int, int] = (0, 255, 80),
    width: int = 3,
) -> tuple[int, int] | None:
    local, edges = _panda_wireframe(opening)
    world = (T[:3, :3] @ local.T).T + T[:3, 3]
    uvs: list[tuple[int, int] | None] = [_project(p, K) for p in world]
    for i, j in edges:
        if uvs[i] is None or uvs[j] is None:
            continue
        draw.line([uvs[i], uvs[j]], fill=color, width=width)
    if uvs[0] is not None:
        x, y = uvs[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
    return uvs[0]


def _ggcnn_opening_m(g: dict, depth: np.ndarray, K: np.ndarray) -> float:
    """Estimate gripper opening (m) from GG-CNN pixel width + depth."""
    r = int(np.clip(round(float(g["row"])), 0, depth.shape[0] - 1))
    c = int(np.clip(round(float(g["col"])), 0, depth.shape[1] - 1))
    z = float(depth[r, c])
    if z <= 0:
        return 0.06
    fx = float(K[0, 0])
    width_px = float(g.get("length_px", 2.0 * g.get("width_px", 20.0)))
    return float(np.clip(width_px * z / fx, 0.02, 0.10))


def _overlay_base(rgb: np.ndarray, masks: dict[str, np.ndarray], title: str) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGB").copy()
    base = np.asarray(img, dtype=np.float32)
    tint = np.zeros_like(base)
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        c = np.array(OBJ_COLORS[name][2], dtype=np.float32)
        tint[mask.astype(bool)] = c
    blended = base.copy()
    m = tint.any(axis=-1)
    blended[m] = 0.72 * base[m] + 0.28 * tint[m]
    img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(255, 255, 255))
    return img


@dataclass
class ObjCloud:
    name: str
    mask: np.ndarray
    pc: np.ndarray


def _subsample(pc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pc.shape[0] <= n:
        return pc
    return pc[rng.choice(pc.shape[0], n, replace=False)]


def main() -> int:
    from capx.envs.base import get_env

    print("[1/5] Loading franka_robosuite_cubes_low_level …", flush=True)
    env = get_env("franka_robosuite_cubes_low_level", enable_render=True)
    obs, info = env.reset()
    cam = obs["robot0_robotview"]
    rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8)
    depth = _depth_hw(cam["images"]["depth"])
    K = np.asarray(cam["intrinsics"], dtype=np.float64)
    masks = _color_cube_masks(rgb)
    rng = np.random.default_rng(0)

    objects: list[ObjCloud] = []
    for name, mask in masks.items():
        pc = _depth_to_points(depth, K, mask)[0]
        pc = _subsample(pc, 4000, rng)
        print(f"  {name}: mask_px={int(mask.sum())} pc={pc.shape[0]}", flush=True)
        if pc.shape[0] >= 32:
            objects.append(ObjCloud(name=name, mask=mask, pc=pc))
    if not objects:
        raise RuntimeError("No objects segmented (need red/green cubes)")

    Image.fromarray(rgb).save(OUT_DIR / "cube_stack_rgb.png")
    # Combined instance id map: 1=red, 2=green
    inst = np.zeros(rgb.shape[:2], dtype=np.int32)
    for i, (name, mask) in enumerate(masks.items(), start=1):
        inst[mask.astype(bool)] = i
        Image.fromarray((mask * 255).astype(np.uint8)).save(OUT_DIR / f"cube_stack_{name}_mask.png")

    # ----- 8115 Contact-GraspNet: depth + per-object mask -----
    print("[2/5] Contact-GraspNet :8115 (all objects) …", flush=True)
    assert health_check("graspnet"), "Contact-GraspNet not healthy"
    propose_gn = init_propose_grasp_pose("graspnet")
    img = _overlay_base(rgb, masks, "Contact-GraspNet :8115")
    draw = ImageDraw.Draw(img)
    total_gn = 0
    for obj in objects:
        grasp_pose_dict = propose_gn(
            depth=depth,
            cam_K=K.astype(np.float32),
            segmap=obj.mask.astype(np.int32),
            forward_passes=2,
            max_tries=10,
        )
        poses, scores = grasp_pose_dict["poses"], grasp_pose_dict["scores"]
        total_gn += len(scores)
        best_c, other_c, _ = OBJ_COLORS[obj.name]
        order = np.argsort(scores)[::-1][:6] if len(scores) else []
        print(
            f"  {obj.name}: {len(scores)} grasps best="
            f"{scores[order[:3]] if len(order) else []}",
            flush=True,
        )
        for rank, idx in enumerate(order):
            color = best_c if rank == 0 else other_c
            uv = _draw_gripper(
                draw, poses[idx], K, opening=0.08, color=color, width=3 if rank == 0 else 2
            )
            if uv:
                draw.text(
                    (uv[0] + 6, uv[1] - 8),
                    f"{obj.name[0]}{rank}:{float(scores[idx]):.2f}",
                    fill=color,
                )
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text(
        (8, 6),
        f"Contact-GraspNet :8115  n={total_gn} objs={len(objects)}",
        fill=(255, 255, 255),
    )
    out_gn = OUT_DIR / "cube_stack_graspnet_8115.png"
    img.save(out_gn)
    print(f"  saved {out_gn}", flush=True)

    # ----- 8119 GG-CNN: one plan per object mask -----
    print("[3/5] GG-CNN :8119 (all objects) …", flush=True)
    assert health_check("ggcnn"), "GG-CNN not healthy"
    propose_gg = init_propose_grasp_pose("ggcnn")
    per_obj_gg: dict[str, dict] = {}
    total_gg = 0
    for obj in objects:
        grasp_pose_dict = propose_gg(
            depth=depth,
            cam_K=K.astype(np.float32),
            segmap=obj.mask.astype(np.int32),
            num_grasps=5,
            quality_threshold=0.05,
        )
        per_obj_gg[obj.name] = grasp_pose_dict
        total_gg += len(grasp_pose_dict["scores"])
        print(f"  {obj.name}: {len(grasp_pose_dict['scores'])} grasps", flush=True)

    img = _overlay_base(rgb, masks, f"GG-CNN :8119  n={total_gg} objs={len(objects)}")
    draw = ImageDraw.Draw(img)
    for obj in objects:
        best_c, other_c, _ = OBJ_COLORS[obj.name]
        gg = per_obj_gg[obj.name]
        poses_gg = gg["poses"]
        grasps_gg = gg["grasps"] or []
        for i, g in enumerate(grasps_gg):
            color = best_c if i == 0 else other_c
            opening = _ggcnn_opening_m(g, depth, K)
            T = poses_gg[i] if i < len(poses_gg) else None
            if T is not None and T.shape == (4, 4):
                uv = _draw_gripper(
                    draw, T, K, opening=opening, color=color, width=3 if i == 0 else 2
                )
            else:
                uv = (int(g["col"]), int(g["row"]))
            if uv:
                draw.text(
                    (uv[0] + 8, uv[1] - 10),
                    f"{obj.name[0]}{i}:{g['quality']:.2f}",
                    fill=color,
                )
    out_gg = OUT_DIR / "cube_stack_ggcnn_8119.png"
    img.save(out_gg)
    print(f"  saved {out_gg}", flush=True)

    # ----- 8121 GraspGen (default propose_grasp): infer per-object cloud -----
    print("[4/5] GraspGen :8121 (default propose_grasp, all objects) …", flush=True)
    assert health_check(), "GraspGen not healthy"
    propose_g = init_propose_grasp_pose()  # default: graspgen
    img = _overlay_base(rgb, masks, "GraspGen :8121")
    draw = ImageDraw.Draw(img)
    total_g = 0
    for obj in objects:
        grasp_pose_dict = propose_g(
            pc_segment=obj.pc,
            num_grasps=80,
            topk_num_grasps=8,
            min_grasps=3,
            max_tries=4,
            remove_outliers=False,
        )
        poses, scores = grasp_pose_dict["poses"], grasp_pose_dict["scores"]
        total_g += len(scores)
        best_c, other_c, _ = OBJ_COLORS[obj.name]
        order = np.argsort(scores)[::-1][:6]
        print(
            f"  {obj.name}: {len(scores)} grasps best={scores[order[:3]] if len(scores) else []}",
            flush=True,
        )
        for rank, idx in enumerate(order):
            color = best_c if rank == 0 else other_c
            uv = _draw_gripper(
                draw, poses[idx], K, opening=0.08, color=color, width=3 if rank == 0 else 2
            )
            if uv:
                draw.text(
                    (uv[0] + 6, uv[1] - 8),
                    f"{obj.name[0]}{rank}:{float(scores[idx]):.2f}",
                    fill=color,
                )
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), f"GraspGen :8121  n={total_g} objs={len(objects)}", fill=(255, 255, 255))
    out_g = OUT_DIR / "cube_stack_graspgen_8121.png"
    img.save(out_g)
    print(f"  saved {out_g}", flush=True)

    # ----- 8123 GraspGenX: infer per-object cloud -----
    print("[5/5] GraspGenX :8123 (all objects) …", flush=True)
    assert health_check("graspgenx"), "GraspGenX not healthy"
    propose_x = init_propose_grasp_pose("graspgenx")
    img = _overlay_base(rgb, masks, "GraspGenX :8123")
    draw = ImageDraw.Draw(img)
    total_x = 0
    for obj in objects:
        grasp_pose_dict = propose_x(
            pc_segment=obj.pc,
            gripper_name="franka_panda",
            num_grasps=80,
            topk_num_grasps=8,
            min_grasps=3,
            max_tries=4,
            remove_outliers=False,
        )
        poses, scores = grasp_pose_dict["poses"], grasp_pose_dict["scores"]
        total_x += len(scores)
        best_c, other_c, _ = OBJ_COLORS[obj.name]
        order = np.argsort(scores)[::-1][:6]
        print(
            f"  {obj.name}: {len(scores)} grasps best={scores[order[:3]] if len(scores) else []}",
            flush=True,
        )
        for rank, idx in enumerate(order):
            color = best_c if rank == 0 else other_c
            uv = _draw_gripper(
                draw, poses[idx], K, opening=0.08, color=color, width=3 if rank == 0 else 2
            )
            if uv:
                draw.text(
                    (uv[0] + 6, uv[1] - 8),
                    f"{obj.name[0]}{rank}:{float(scores[idx]):.2f}",
                    fill=color,
                )
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), f"GraspGenX :8123  n={total_x} objs={len(objects)}", fill=(255, 255, 255))
    out_x = OUT_DIR / "cube_stack_graspgenx_8123.png"
    img.save(out_x)
    print(f"  saved {out_x}", flush=True)

    print("DONE")
    print(" ", out_gn)
    print(" ", out_gg)
    print(" ", out_g)
    print(" ", out_x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
