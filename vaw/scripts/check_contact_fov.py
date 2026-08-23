"""Render one Contact camera pair at several fields of view for comparison.

Contact panels are wide-angle by default, so an object nearer the camera is
drawn noticeably larger than one further away and the policy has to guess
depth to compare heights.  Narrowing the FOV pushes the camera back and makes
the projection quasi-orthographic, but only up to the point where the camera
leaves the room and a wall occludes the subject.  This script replays a
recorded joint configuration so that trade-off can be looked at rather than
assumed.

    python -m vaw.scripts.check_contact_fov --task-id 4 --seed 1 \
        --joints 0.0976 0.8935 -0.0648 -1.2955 0.0752 2.2507 -0.7724 \
        --fovs 48 36 28 22 --out /tmp/fov
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--joints", type=float, nargs=7, required=True)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        required=True,
        help="contact frame centre in robot-base metres (the recorded TCP)",
    )
    parser.add_argument("--gripper", type=float, default=0.04)
    parser.add_argument("--fovs", type=float, nargs="+", default=[48.0, 36.0, 28.0, 22.0])
    parser.add_argument("--reference-fov", type=float, default=48.0)
    parser.add_argument("--reference-distance", type=float, default=0.26)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=358)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from capx.envs.simulators.libero import FrankaLiberoTask

    from vaw.context_runtime.contact_camera import (
        ContactCameraRequest,
        LiberoContactCameraProvider,
    )

    env = FrankaLiberoTask(
        suite_name=args.suite,
        task_id=args.task_id,
        privileged=False,
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    sim = env.handle.env.sim
    # Replay the recorded arm pose; the objects stay at their reset placement,
    # which is enough to compare projections of the same gripper geometry.
    sim.data.qpos[:7] = np.asarray(args.joints, dtype=np.float64)
    sim.data.qpos[7:9] = (args.gripper, -args.gripper)
    sim.forward()

    center = tuple(float(value) for value in args.center)
    args.out.mkdir(parents=True, exist_ok=True)
    # Hold the subtended size fixed while the FOV changes, otherwise the
    # comparison is a zoom and says nothing about perspective.
    reference_tangent = np.tan(np.deg2rad(args.reference_fov) * 0.5)
    for fovy in args.fovs:
        distance = args.reference_distance * float(
            reference_tangent / np.tan(np.deg2rad(fovy) * 0.5)
        )
        provider = LiberoContactCameraProvider(
            env,
            fovy_deg=float(fovy),
            distance_m=distance,
        )
        for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            pair = provider(
                ContactCameraRequest(
                    center_base_xyz=center,
                    frame_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                    width=args.width,
                    panel_height=args.height,
                    preferred_signs=signs,
                )
            )
            stacked = np.vstack(
                (pair.front["images"]["rgb"], pair.side["images"]["rgb"])
            )
            tag = f"{'p' if signs[0] > 0 else 'm'}{'p' if signs[1] > 0 else 'm'}"
            path = args.out / f"fov{int(fovy):03d}_{tag}.png"
            Image.fromarray(stacked).save(path)
            print(f"fovy={fovy:>5.1f} d={distance:.3f}m signs={signs} -> {path}")


if __name__ == "__main__":
    main()
