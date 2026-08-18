"""Smoke-test filled versus 3-D edge-only virtual gripper overlays.

The mesh glyph is only worth drawing if it lands on the real gripper: its whole
claim is "the hand is *here*".  This diagnostic deliberately avoids the
Cartesian readback/TCP-offset chain.  It reads the observed seven arm joints,
updates an isolated Panda URDF, and projects those exact FK visual meshes into
the calibrated camera images.

The right-hand proposal does not fill any image pixel inside the projected
gripper.  It draws the projected silhouette and visible 3-D crease edges with
one uniform, thin lavender stroke.  This is an isolated visual smoke: changing
it does not change the policy-visible Canvas renderer.

No perception or motion-planning services are needed — only the environment.

    python -m vaw.scripts.check_gripper_overlay --suite libero_object_swap --task-id 0
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vaw.context_runtime.geometry import project_world_to_pixel

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "gripper_overlay.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--preview-delta",
        default="0.00,0.00,0.00",
        help=(
            "optional robot-base XYZ translation applied to the virtual gripper, "
            "in metres; the zero default is the geometry-alignment smoke"
        ),
    )
    parser.add_argument(
        "--include-wrist",
        action="store_true",
        help="also render the raw wrist camera instead of the agentview close-up row",
    )
    args = parser.parse_args()

    from capx.envs.simulators.libero import FrankaLiberoTask
    from vaw.context_runtime.gripper_mesh import (
        load_panda_urdf_fk,
        mask_outline,
        overlay_projected_mesh_outline,
        rasterize_silhouette,
    )

    fk = load_panda_urdf_fk()
    if fk is None:
        raise SystemExit("robot_descriptions/yourdfpy Panda URDF is unavailable")

    env = FrankaLiberoTask(suite_name=args.suite, task_id=args.task_id, privileged=False, seed=args.seed)
    obs = env.get_observation()

    cart = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64).reshape(-1)
    joints = np.asarray(obs["robot_joint_pos"], dtype=np.float64).reshape(-1)[:7]
    opening = float(cart[7])
    hand = fk.frame(joints, "panda_hand", gripper_opening=opening)
    observed_triangles = fk.triangles(joints, opening)
    preview_delta = _parse_vector3(args.preview_delta)
    preview_triangles = observed_triangles + preview_delta
    print(f"observed arm joints {np.round(joints, 4).tolist()} opening {opening:.3f}")
    print(f"URDF-FK hand link {np.round(hand[:3, 3], 4).tolist()}")

    panels = []
    cameras = ("agentview", "robot0_eye_in_hand") if args.include_wrist else ("agentview",)
    for camera in cameras:
        cam = obs.get(camera)
        if cam is None:
            print(f"  (no {camera} in observation)")
            continue
        rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8).copy()
        h, w = rgb.shape[:2]
        uvz = project_world_to_pixel(
            preview_triangles.reshape(-1, 3),
            cam["intrinsics"],
            cam["pose_mat"],
        )
        mask = rasterize_silhouette(
            uvz[:, :2].reshape(-1, 3, 2), uvz[:, 2].reshape(-1, 3), w, h
        )
        print(f"  {camera}: {mask.sum()} px covered of {w}x{h}")
        legacy = rgb.astype(np.float64)
        legacy[mask] = (
            legacy[mask] * 0.36 + np.array([124.0, 58.0, 237.0]) * 0.64
        )
        legacy[mask_outline(mask)] = (196, 181, 253)

        edge_only = rgb.copy()
        overlay_projected_mesh_outline(
            edge_only,
            preview_triangles,
            uvz[:, :2].reshape(-1, 3, 2),
            uvz[:, 2].reshape(-1, 3),
            camera_position_base=np.asarray(cam["pose_mat"], dtype=np.float64)[
                :3, 3
            ],
        )

        raw_panel = _panel_label(rgb, f"{camera.upper()} · RAW RGB")
        legacy_panel = _panel_label(
            legacy.astype(np.uint8),
            "OLD · FILLED MASK",
        )
        edge_panel = _panel_label(
            edge_only,
            "PROPOSED · 3D EDGES ONLY",
        )
        panels.append(np.concatenate([raw_panel, legacy_panel, edge_panel], axis=1))
        if camera == "agentview" and not args.include_wrist:
            panels.append(
                np.concatenate(
                    [
                        _panel_label(_focus_crop(rgb, mask), "RAW · TARGET CROP"),
                        _panel_label(
                            _focus_crop(legacy.astype(np.uint8), mask),
                            "OLD · FILLED CROP",
                        ),
                        _panel_label(
                            _focus_crop(edge_only, mask),
                            "PROPOSED · 3D EDGE CROP",
                        ),
                    ],
                    axis=1,
                )
            )

    out = pathlib.Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=0)).save(out)
    print(
        f"wrote {out}  "
        "(left: raw, middle: old fill, right: proposed 3-D edge-only)"
    )


def _parse_vector3(value: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise SystemExit("--preview-delta must be comma-separated XYZ metres") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise SystemExit("--preview-delta must contain exactly three finite values")
    return result


def _focus_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Enlarge the target while retaining enough scene to judge occlusion."""

    values = np.asarray(image, dtype=np.uint8)
    height, width = values.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return values.copy()
    center_x = 0.5 * float(xs.min() + xs.max())
    center_y = 0.5 * float(ys.min() + ys.max())
    crop_width = max(240.0, float(xs.max() - xs.min()) * 3.2)
    crop_height = crop_width * height / width
    if crop_height < float(ys.max() - ys.min()) * 2.2:
        crop_height = float(ys.max() - ys.min()) * 2.2
        crop_width = crop_height * width / height
    crop_width = min(crop_width, float(width))
    crop_height = min(crop_height, float(height))
    left = int(round(np.clip(center_x - crop_width * 0.5, 0.0, width - crop_width)))
    top = int(round(np.clip(center_y - crop_height * 0.5, 0.0, height - crop_height)))
    right = min(width, int(round(left + crop_width)))
    bottom = min(height, int(round(top + crop_height)))
    return np.asarray(
        Image.fromarray(values[top:bottom, left:right]).resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


def _panel_label(image: np.ndarray, label: str) -> np.ndarray:
    panel = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(panel, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rounded_rectangle(
        (8, 8, bounds[2] + 24, bounds[3] + 19),
        radius=5,
        fill=(3, 7, 18, 220),
        outline=(196, 181, 253, 255),
        width=2,
    )
    draw.text((16, 13), label, fill=(245, 243, 255, 255), font=font)
    return np.asarray(panel, dtype=np.uint8)


if __name__ == "__main__":
    main()
