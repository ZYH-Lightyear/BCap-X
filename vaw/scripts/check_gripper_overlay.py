"""Check the gripper mesh overlay against what the simulator actually renders.

The mesh glyph is only worth drawing if it lands on the real gripper: its whole
claim is "the hand is *here*".  This diagnostic deliberately avoids the
Cartesian readback/TCP-offset chain.  It reads the observed seven arm joints,
updates an isolated Panda URDF, and projects those exact FK visual meshes into
the calibrated camera images.

If the outline traces the rendered gripper, joint ordering, URDF geometry and
camera calibration agree with MuJoCo.
No perception services needed — only the environment.

    python -m vaw.scripts.check_gripper_overlay --suite libero_object --task-id 0
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "gripper_overlay.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from PIL import Image

    from capx.envs.simulators.libero import FrankaLiberoTask
    from vaw.geometry import project_world_to_pixel
    from vaw.gripper_mesh import (
        load_panda_urdf_fk,
        mask_outline,
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
    tris = fk.triangles(joints, opening)
    print(f"observed arm joints {np.round(joints, 4).tolist()} opening {opening:.3f}")
    print(f"URDF-FK hand link {np.round(hand[:3, 3], 4).tolist()}")

    panels = []
    for camera in ("agentview", "robot0_eye_in_hand"):
        cam = obs.get(camera)
        if cam is None:
            print(f"  (no {camera} in observation)")
            continue
        rgb = np.asarray(cam["images"]["rgb"], dtype=np.uint8).copy()
        h, w = rgb.shape[:2]
        uvz = project_world_to_pixel(tris.reshape(-1, 3), cam["intrinsics"], cam["pose_mat"])
        mask = rasterize_silhouette(
            uvz[:, :2].reshape(-1, 3, 2), uvz[:, 2].reshape(-1, 3), w, h
        )
        print(f"  {camera}: {mask.sum()} px covered of {w}x{h}")
        blend = rgb.astype(np.float64)
        blend[mask] = blend[mask] * 0.45 + np.array([80.0, 160.0, 255.0]) * 0.55
        blend[mask_outline(mask)] = (255, 60, 60)
        panels.append(np.concatenate([rgb, blend.astype(np.uint8)], axis=1))

    out = pathlib.Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=0)).save(out)
    print(f"wrote {out}  (left: raw, right: silhouette overlaid)")


if __name__ == "__main__":
    main()
