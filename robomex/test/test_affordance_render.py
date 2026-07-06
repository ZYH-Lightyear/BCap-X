from __future__ import annotations

import numpy as np

from robomex.perception import (
    project_world_to_pixel,
    save_grasp_affordance_3d,
    save_grasp_affordance_overlay,
)


def test_project_world_to_pixel_identity_camera() -> None:
    k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)

    uv = project_world_to_pixel([0.0, 0.0, 1.0], k, pose, (48, 64, 3))

    assert uv == (32.0, 24.0)


def test_save_grasp_affordance_overlay(tmp_path) -> None:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :] = [40, 40, 40]
    k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    mask = np.zeros((48, 64), dtype=bool)
    mask[18:30, 26:38] = True

    out = save_grasp_affordance_overlay(
        tmp_path / "affordance.png",
        rgb,
        k,
        pose,
        [
            {
                "strategy": "top_down",
                "pos": [0.0, 0.0, 1.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "score": 0.9,
                "ik_ok": True,
            }
        ],
        mask=mask,
        bbox=[20, 14, 44, 34],
    )

    assert out.endswith("affordance.png")
    assert (tmp_path / "affordance.png").exists()
    assert (tmp_path / "affordance.png").stat().st_size > 0


def test_save_grasp_affordance_3d(tmp_path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ],
        dtype=float,
    )

    out = save_grasp_affordance_3d(
        tmp_path / "affordance_3d.png",
        points,
        [
            {
                "strategy": "top_down",
                "pos": [0.02, 0.02, 0.05],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "score": 0.9,
                "ik_ok": True,
            }
        ],
    )

    assert out.endswith("affordance_3d.png")
    assert (tmp_path / "affordance_3d.png").exists()
    assert (tmp_path / "affordance_3d.png").stat().st_size > 0
