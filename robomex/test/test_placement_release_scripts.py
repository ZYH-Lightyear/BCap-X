from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def _load(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _disc_points(center=(0.4, 0.2, 0.02), radius=0.06, z=0.02):
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rings = []
    for scale in (0.25, 0.55, 0.85, 1.0):
        rings.append(
            np.column_stack(
                [
                    center[0] + radius * scale * np.cos(angles),
                    center[1] + radius * scale * np.sin(angles),
                    np.full_like(angles, z),
                ]
            )
        )
    return np.vstack(rings)


def test_placement_affordance_support_surface_outputs_object_center_contract() -> None:
    mod = _load("affordance/find_placement/scripts/placement_affordance.py", "placement_affordance_test")
    evidence = {}

    result = mod.estimate_placement_affordance(
        _disc_points(),
        target_name="plate",
        evidence=evidence,
    )

    assert result["mode"] == "support_surface"
    assert result["quaternion_wxyz"] == [0.0, 1.0, 0.0, 0.0]
    assert np.allclose(result["desired_object_center"][:2], [0.4, 0.2], atol=0.005)
    assert evidence["placement_affordance"] == result
    assert "place_point" not in evidence
    assert "place_strategy" not in evidence


def _basket_points(center=(0.4, 0.2), rim_z=0.12, floor_z=0.02, radius=0.06):
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rim = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full_like(angles, rim_z),
        ]
    )
    floor = _disc_points(center=(center[0], center[1], floor_z), radius=radius * 0.8, z=floor_z)
    return np.vstack([rim, floor])


def test_placement_affordance_open_container_mode() -> None:
    mod = _load("affordance/find_placement/scripts/placement_affordance.py", "placement_affordance_test2")

    result = mod.estimate_placement_affordance(
        _basket_points(),
        target_name="basket",
    )

    assert result["mode"] == "open_container"
    # The desired object center sits inside the container: above the interior
    # floor (plus clearance) and strictly below the rim. No rim + 0.10 default.
    z = result["desired_object_center"][2]
    assert result["zone_floor"] < z < result["rim_top"]
    assert z > 0.10


def test_placement_3d_visualization_saves_artifact(tmp_path) -> None:
    mod = _load("affordance/find_placement/scripts/placement_affordance.py", "placement_affordance_viz_test")
    points = _disc_points()
    affordance = mod.estimate_placement_affordance(points, target_name="plate")

    path = mod.save_placement_3d_visualization(
        points,
        affordance,
        str(tmp_path),
        held_object_frame={"object_center_offset_from_grasp": [0.02, 0.0, 0.0]},
    )

    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    assert affordance["artifacts"]["3d_visualization"] == path


def test_release_script_compensates_off_center_grasp() -> None:
    mod = _load("motion/release_at/scripts/release.py", "release_test")
    placement = {"desired_object_center": [0.5, 0.1, 0.2], "place_quat": [0.0, 1.0, 0.0, 0.0]}
    held = {"object_center_offset_from_grasp": [0.04, -0.02, 0.0]}

    tcp = mod.compute_tcp_release_pos(placement, held)

    assert np.allclose(tcp, [0.46, 0.12, 0.2])
    assert np.allclose(mod.place_quat_from_affordance(placement), [0.0, 1.0, 0.0, 0.0])
