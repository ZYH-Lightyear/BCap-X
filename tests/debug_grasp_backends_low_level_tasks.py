#!/usr/bin/env python3
"""Run GG-CNN (:8119), GraspGen (:8121), GraspGenX (:8123) on every low-level
task listed in ``debug_tasks.txt``.

For each task × backend × object, save one grasp-overlay PNG under::

    outputs/debug/<task>/<backend>_<port>/<object>.png

Also writes ``rgb.png`` and ``<object>_mask.png`` per task folder.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Localhost must not go through the corporate HTTP proxy.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT_ROOT = REPO / "outputs" / "debug"
DEBUG_TASKS = REPO / "debug_tasks.txt"

# debug_tasks.txt low-level names → registered env names
ENV_ALIASES = {
    "nut_assembly": "franka_robosuite_nut_assembly_low_level",
    "nut_assembly_visual": "franka_robosuite_nut_assembly_low_level_visual",
}

BACKENDS = (
    ("ggcnn", 8119, "capx/integrations/vision/ggcnn.py", "ggcnn_client"),
    ("graspgen", 8121, "capx/integrations/vision/graspgen.py", "graspgen_client"),
    ("graspgenx", 8123, "capx/integrations/vision/graspgenx.py", "graspgenx_client"),
)

# Distinct tint / grasp colors cycled per object
_PALETTE = [
    ((0, 255, 80), (255, 180, 0), (255, 60, 60)),
    ((80, 200, 255), (180, 120, 255), (40, 200, 80)),
    ((255, 220, 40), (255, 100, 180), (60, 120, 255)),
    ((0, 255, 200), (255, 140, 60), (200, 80, 200)),
    ((180, 255, 80), (100, 180, 255), (255, 120, 40)),
]

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


@dataclass
class ObjCloud:
    name: str
    mask: np.ndarray
    pc: np.ndarray
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


def _load_client(rel: str, name: str):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_low_level_tasks(path: Path) -> list[str]:
    """Pairs in debug_tasks.txt: code_env then low_level; return low_level names."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    lows: list[str] = []
    for i in range(0, len(lines) - 1, 2):
        lows.append(lines[i + 1])
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in lows:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


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
        valid = valid & mask.astype(bool)
    pts = full[valid].astype(np.float32)
    return pts, full, valid


def _subsample(pc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pc.shape[0] <= n:
        return pc
    return pc[rng.choice(pc.shape[0], n, replace=False)]


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


def _mask_from_center(
    pc_full: np.ndarray, center_cam: np.ndarray, radius: float
) -> np.ndarray:
    d = np.linalg.norm(pc_full - center_cam[None, None, :], axis=-1)
    valid = np.isfinite(pc_full).all(axis=-1) & (pc_full[..., 2] > 0.05)
    return (d < radius) & valid


def _robot_to_cam(p_robot: np.ndarray, T_cam_robot: np.ndarray) -> np.ndarray:
    T_inv = np.linalg.inv(T_cam_robot)
    p = np.asarray(p_robot, dtype=np.float64)[:3]
    return T_inv[:3, :3] @ p + T_inv[:3, 3]


def _pick_camera(obs: dict) -> tuple[str, dict]:
    for name in ("robot0_robotview", "agentview"):
        if name in obs and isinstance(obs[name], dict) and "images" in obs[name]:
            return name, obs[name]
    for k, v in obs.items():
        if isinstance(v, dict) and "images" in v and "rgb" in v["images"]:
            return k, v
    raise KeyError(f"No camera observation in keys={list(obs.keys())}")


def _pose_dict_objects(obs: dict) -> dict[str, np.ndarray]:
    """Collect graspable object poses (robot base frame, xyz[+quat])."""
    out: dict[str, np.ndarray] = {}
    if "cube_poses" in obs:
        for k, v in obs["cube_poses"].items():
            out[str(k)] = np.asarray(v, dtype=np.float64)
    if "nut_poses" in obs:
        for k, v in obs["nut_poses"].items():
            kl = str(k).lower()
            # skip pegs / offsets — not grasp targets
            if "peg" in kl or "offset" in kl:
                continue
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size < 3:
                continue
            out[str(k)] = arr
    if "hammer_poses" in obs:
        for k, v in obs["hammer_poses"].items():
            out[str(k)] = np.asarray(v, dtype=np.float64)
    return out


def _radii_for_name(name: str) -> float:
    n = name.lower()
    if "handle" in n:
        return 0.045
    if "nut" in n:
        return 0.04
    if "hammer" in n:
        return 0.06
    if "cube" in n or "primary" in n or "secondary" in n:
        return 0.035
    return 0.04


def _color_fallback_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Heuristic colored blobs when no privileged poses exist (e.g. spill wipe dirt)."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    masks: dict[str, np.ndarray] = {}
    red = _clean_blob((r > 140) & (r - g > 50) & (r - b > 50))
    green = _clean_blob((g > 140) & (g - r > 40) & (g - b > 40))
    blue = _clean_blob((b > 140) & (b - r > 40) & (b - g > 40))
    # dark dirt / markers on light table
    dark = _clean_blob((r < 70) & (g < 70) & (b < 70) & ((r + g + b) > 30))
    if int(red.sum()) > 80:
        masks["red_blob"] = red
    if int(green.sum()) > 80:
        masks["green_blob"] = green
    if int(blue.sum()) > 80:
        masks["blue_blob"] = blue
    if int(dark.sum()) > 80:
        masks["dark_blob"] = dark
    return masks


def _objects_from_obs(
    obs: dict,
    cam: dict,
    depth: np.ndarray,
    K: np.ndarray,
    rng: np.random.Generator,
) -> list[ObjCloud]:
    rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8)
    T = np.asarray(cam["pose_mat"], dtype=np.float64)
    _, pc_full, _ = _depth_to_points(depth, K, None)

    pose_objs = _pose_dict_objects(obs)
    masks: dict[str, np.ndarray] = {}

    # Prefer instance segmentation when present (handover).
    seg = cam["images"].get("segmentation")
    if seg is not None and pose_objs:
        seg_hw = np.asarray(seg)
        if seg_hw.ndim == 3:
            seg_hw = seg_hw[..., 0]
        # Map each pose to nearest non-background seg id via projected center.
        for name, pose in pose_objs.items():
            p_cam = _robot_to_cam(pose[:3], T)
            if p_cam[2] <= 1e-6:
                continue
            uv = K @ p_cam
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            if 0 <= v < seg_hw.shape[0] and 0 <= u < seg_hw.shape[1]:
                sid = int(seg_hw[v, u])
                if sid > 0:
                    masks[name] = (seg_hw == sid).astype(np.uint8)

    for name, pose in pose_objs.items():
        if name in masks and int(masks[name].sum()) >= 32:
            continue
        p_cam = _robot_to_cam(pose[:3], T)
        m = _mask_from_center(pc_full, p_cam, _radii_for_name(name))
        masks[name] = m.astype(np.uint8)

    if not masks:
        # spill_wipe: use wipe_centroid ball + color blobs
        if "wipe_centroid" in obs:
            # wipe_centroid often world/table frame; try both cam via depth peak near color
            color_masks = _color_fallback_masks(rgb)
            masks.update(color_masks)
            if not masks:
                # fallback: table-top elevated points cluster as dirt_region
                table_z = np.median(pc_full[np.isfinite(pc_full[..., 2]), 2])
                elev = (
                    np.isfinite(pc_full).all(-1)
                    & (pc_full[..., 2] > 0.05)
                    & (pc_full[..., 2] < table_z - 0.002)
                )
                # actually above table: smaller z in cam is closer; use XY density
                # Use dark-ish residual after table plane: depth relative jump
                d = depth
                local = d - np.nanmedian(d)
                bump = _clean_blob((local < -0.005) & np.isfinite(d))
                if int(bump.sum()) > 80:
                    masks["dirt_region"] = bump
        else:
            masks.update(_color_fallback_masks(rgb))

    objects: list[ObjCloud] = []
    for i, (name, mask) in enumerate(masks.items()):
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_") or f"obj_{i}"
        pc = _depth_to_points(depth, K, mask)[0]
        pc = _subsample(pc, 4000, rng)
        print(f"    {safe}: mask_px={int(mask.sum())} pc={pc.shape[0]}", flush=True)
        if pc.shape[0] < 32:
            continue
        objects.append(
            ObjCloud(name=safe, mask=mask.astype(np.uint8), pc=pc, colors=_PALETTE[i % len(_PALETTE)])
        )
    return objects


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
    r = int(np.clip(round(float(g["row"])), 0, depth.shape[0] - 1))
    c = int(np.clip(round(float(g["col"])), 0, depth.shape[1] - 1))
    z = float(depth[r, c])
    if z <= 0:
        return 0.06
    fx = float(K[0, 0])
    width_px = float(g.get("length_px", 2.0 * g.get("width_px", 20.0)))
    return float(np.clip(width_px * z / fx, 0.02, 0.10))


def _overlay_one(
    rgb: np.ndarray,
    obj: ObjCloud,
    title: str,
) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGB").copy()
    base = np.asarray(img, dtype=np.float32)
    tint = np.zeros_like(base)
    tint[obj.mask.astype(bool)] = np.array(obj.colors[2], dtype=np.float32)
    blended = base.copy()
    m = tint.any(axis=-1)
    blended[m] = 0.72 * base[m] + 0.28 * tint[m]
    img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), title[:90], fill=(255, 255, 255))
    return img


def _run_ggcnn(plan, obj: ObjCloud, depth, K) -> tuple[list, np.ndarray]:
    gg = plan(
        depth=depth,
        cam_K=K.astype(np.float32),
        segmap=obj.mask.astype(np.int32),
        segmap_id=1,
        n_grasps=5,
        threshold_abs=0.05,
    )
    if not gg["grasps"]:
        gg = plan(
            depth=depth,
            cam_K=K.astype(np.float32),
            segmap=obj.mask.astype(np.int32),
            segmap_id=1,
            n_grasps=5,
            threshold_abs=0.01,
        )
    poses = gg.get("poses", np.zeros((0, 4, 4)))
    return gg["grasps"], poses


def _draw_ggcnn(img, grasps, poses, depth, K, colors):
    draw = ImageDraw.Draw(img)
    best_c, other_c, _ = colors
    for i, g in enumerate(grasps):
        color = best_c if i == 0 else other_c
        opening = _ggcnn_opening_m(g, depth, K)
        T = poses[i] if i < len(poses) else None
        if T is not None and np.asarray(T).shape == (4, 4):
            uv = _draw_gripper(draw, T, K, opening=opening, color=color, width=3 if i == 0 else 2)
        else:
            uv = (int(g["col"]), int(g["row"]))
        if uv:
            draw.text((uv[0] + 8, uv[1] - 10), f"{i}:{g['quality']:.2f}", fill=color)


def _draw_poses(img, poses, scores, K, colors, topk: int = 6):
    draw = ImageDraw.Draw(img)
    best_c, other_c, _ = colors
    if len(scores) == 0:
        return
    order = np.argsort(scores)[::-1][:topk]
    for rank, idx in enumerate(order):
        color = best_c if rank == 0 else other_c
        uv = _draw_gripper(
            draw, poses[idx], K, opening=0.08, color=color, width=3 if rank == 0 else 2
        )
        if uv:
            draw.text((uv[0] + 6, uv[1] - 8), f"{rank}:{float(scores[idx]):.2f}", fill=color)


def run_task(env_name: str, clients: dict) -> None:
    from capx.envs.base import get_env

    registered = ENV_ALIASES.get(env_name, env_name)
    print(f"\n======== {env_name} (-> {registered}) ========", flush=True)
    task_dir = OUT_ROOT / env_name
    task_dir.mkdir(parents=True, exist_ok=True)

    env = get_env(registered, enable_render=True)
    try:
        obs, _info = env.reset()
        cam_name, cam = _pick_camera(obs)
        rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8)
        depth = _depth_hw(cam["images"]["depth"])
        K = np.asarray(cam["intrinsics"], dtype=np.float64)
        Image.fromarray(rgb).save(task_dir / "rgb.png")
        print(f"  camera={cam_name} rgb={rgb.shape}", flush=True)

        rng = np.random.default_rng(0)
        objects = _objects_from_obs(obs, cam, depth, K, rng)
        if not objects:
            print("  WARNING: no objects segmented; skip backends", flush=True)
            return

        for obj in objects:
            Image.fromarray((obj.mask * 255).astype(np.uint8)).save(
                task_dir / f"{obj.name}_mask.png"
            )

        # GG-CNN
        name, port, _, _ = BACKENDS[0]
        out_dir = task_dir / f"{name}_{port}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_gg = clients["ggcnn"]
        print(f"  [{name}:{port}]", flush=True)
        for obj in objects:
            grasps, poses = _run_ggcnn(plan_gg, obj, depth, K)
            title = f"GG-CNN :{port}  {env_name} / {obj.name}  n={len(grasps)}"
            img = _overlay_one(rgb, obj, title)
            _draw_ggcnn(img, grasps, poses, depth, K, obj.colors)
            path = out_dir / f"{obj.name}.png"
            img.save(path)
            print(f"    saved {path.relative_to(REPO)}  grasps={len(grasps)}", flush=True)

        # GraspGen
        name, port, _, _ = BACKENDS[1]
        out_dir = task_dir / f"{name}_{port}"
        out_dir.mkdir(parents=True, exist_ok=True)
        infer_g = clients["graspgen"]
        print(f"  [{name}:{port}]", flush=True)
        for obj in objects:
            poses, scores = infer_g(
                obj.pc,
                num_grasps=80,
                topk_num_grasps=8,
                min_grasps=3,
                max_tries=4,
                remove_outliers=False,
            )
            title = f"GraspGen :{port}  {env_name} / {obj.name}  n={len(scores)}"
            img = _overlay_one(rgb, obj, title)
            _draw_poses(img, poses, scores, K, obj.colors)
            path = out_dir / f"{obj.name}.png"
            img.save(path)
            best = float(np.max(scores)) if len(scores) else float("nan")
            print(
                f"    saved {path.relative_to(REPO)}  grasps={len(scores)} best={best:.3f}",
                flush=True,
            )

        # GraspGenX
        name, port, _, _ = BACKENDS[2]
        out_dir = task_dir / f"{name}_{port}"
        out_dir.mkdir(parents=True, exist_ok=True)
        infer_x = clients["graspgenx"]
        print(f"  [{name}:{port}]", flush=True)
        for obj in objects:
            poses, scores = infer_x(
                obj.pc,
                gripper_name="franka_panda",
                num_grasps=80,
                topk_num_grasps=8,
                min_grasps=3,
                max_tries=4,
                remove_outliers=False,
            )
            title = f"GraspGenX :{port}  {env_name} / {obj.name}  n={len(scores)}"
            img = _overlay_one(rgb, obj, title)
            _draw_poses(img, poses, scores, K, obj.colors)
            path = out_dir / f"{obj.name}.png"
            img.save(path)
            best = float(np.max(scores)) if len(scores) else float("nan")
            print(
                f"    saved {path.relative_to(REPO)}  grasps={len(scores)} best={best:.3f}",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = _parse_low_level_tasks(DEBUG_TASKS)
    print("Low-level tasks:", tasks, flush=True)

    clients = {}
    ggcnn = _load_client(BACKENDS[0][2], BACKENDS[0][3])
    assert ggcnn.health_check(), "GG-CNN :8119 not healthy (set NO_PROXY for localhost)"
    clients["ggcnn"] = ggcnn.init_ggcnn()

    graspgen = _load_client(BACKENDS[1][2], BACKENDS[1][3])
    assert graspgen.health_check(), "GraspGen :8121 not healthy"
    clients["graspgen"] = graspgen.init_graspgen()

    graspgenx = _load_client(BACKENDS[2][2], BACKENDS[2][3])
    assert graspgenx.health_check(), "GraspGenX :8123 not healthy"
    clients["graspgenx"] = graspgenx.init_graspgenx()

    print("All backends healthy.", flush=True)

    for task in tasks:
        try:
            run_task(task, clients)
        except Exception as e:
            print(f"FAILED {task}: {type(e).__name__}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    print("\nDONE. Outputs under", OUT_ROOT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
