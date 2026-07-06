from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_bowl_grasp_module():
    path = (
        Path(__file__).parents[1]
        / "skills"
        / "builtin"
        / "affordance"
        / "grasp_open_bowl"
        / "scripts"
        / "bowl_grasp.py"
    )
    spec = importlib.util.spec_from_file_location("bowl_grasp_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_bowl_points() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    rim = np.column_stack(
        [
            0.5 + 0.055 * np.cos(angles),
            0.15 + 0.055 * np.sin(angles),
            np.full_like(angles, 0.08),
        ]
    )
    lower = np.column_stack(
        [
            0.5 + 0.035 * np.cos(angles),
            0.15 + 0.035 * np.sin(angles),
            np.full_like(angles, 0.035),
        ]
    )
    bottom = np.column_stack(
        [
            0.5 + 0.018 * np.cos(angles),
            0.15 + 0.018 * np.sin(angles),
            np.full_like(angles, 0.02),
        ]
    )
    return np.vstack([rim, lower, bottom])


def _irregular_bowl_points() -> np.ndarray:
    angles = np.linspace(-0.25 * np.pi, 1.45 * np.pi, 90)
    radius = 0.050 + 0.012 * np.sin(2.0 * angles)
    rim = np.column_stack(
        [
            0.45 + 1.25 * radius * np.cos(angles),
            0.10 + 0.75 * radius * np.sin(angles),
            0.075 + 0.003 * np.cos(angles),
        ]
    )
    inner = np.column_stack(
        [
            0.45 + 0.70 * radius * np.cos(angles),
            0.10 + 0.45 * radius * np.sin(angles),
            np.full_like(angles, 0.045),
        ]
    )
    return np.vstack([rim, inner])


def test_propose_open_bowl_grasps_returns_topdown_rim_candidates() -> None:
    bowl_grasp = _load_bowl_grasp_module()
    result = bowl_grasp.propose_open_bowl_grasps(_synthetic_bowl_points(), num_angles=6)

    assert result["ok"]
    assert result["recommended_strategy"] == "open_bowl_rim_topdown"
    assert len(result["candidates"]) == 18
    candidate = result["selected_candidate"]
    assert candidate["strategy"] == "open_bowl_rim_topdown"
    assert candidate["ik_ok"] is True
    assert candidate["rim_z_margin"] in result["geometry"]["candidate_z_margins"]

    quat = np.asarray(candidate["quat"], dtype=float)
    assert np.isclose(np.linalg.norm(quat), 1.0)

    center = np.asarray(result["geometry"]["center"], dtype=float)
    pos = np.asarray(candidate["pos"], dtype=float)
    object_center = np.asarray(candidate["object_center"], dtype=float)
    center_offset = np.asarray(candidate["object_center_offset_from_grasp"], dtype=float)
    approach = np.asarray(candidate["approach_axis"], dtype=float)
    jaw = np.asarray(candidate["jaw_axis"], dtype=float)
    radial = np.asarray(candidate["radial_axis"], dtype=float)

    # Approach should be top-down, from above the rim toward the grasp point.
    assert np.allclose(approach, np.array([0.0, 0.0, -1.0]))
    assert np.asarray(candidate["pregrasp_pos"], dtype=float)[2] > pos[2]
    assert result["geometry"]["rim_height"] - pos[2] >= 0.003
    # Jaw direction should be tangent to the rim: orthogonal to the bowl radius.
    assert abs(float(np.dot(jaw, radial))) < 1e-6
    assert np.linalg.norm(pos[:2] - center[:2]) > 0.03
    assert np.allclose(object_center, np.array([center[0], center[1], pos[2]]))
    assert np.allclose(center_offset, object_center - pos)
    assert np.linalg.norm(center_offset[:2]) > 0.03


def test_propose_open_bowl_grasps_marks_ik_failures() -> None:
    bowl_grasp = _load_bowl_grasp_module()

    def always_fail(_pos, _quat):
        raise RuntimeError("ik failed")

    result = bowl_grasp.propose_open_bowl_grasps(
        _synthetic_bowl_points(),
        num_angles=3,
        solve_ik_fn=always_fail,
    )

    assert result["ok"]
    assert all(not c["ik_ok"] for c in result["candidates"])
    assert all("ik failed" in c["ik_error"] for c in result["candidates"])


def test_propose_open_bowl_grasps_accepts_custom_depth_offsets() -> None:
    bowl_grasp = _load_bowl_grasp_module()

    result = bowl_grasp.propose_open_bowl_grasps(
        _synthetic_bowl_points(),
        num_angles=2,
        z_margins=(0.006, 0.014),
    )

    assert len(result["candidates"]) == 4
    assert result["geometry"]["candidate_z_margins"] == [0.006, 0.014]
    assert {c["rim_z_margin"] for c in result["candidates"]} == {0.006, 0.014}


def test_propose_open_bowl_grasps_uses_observed_rim_points_not_synthetic_circle() -> None:
    bowl_grasp = _load_bowl_grasp_module()
    points = _irregular_bowl_points()

    result = bowl_grasp.propose_open_bowl_grasps(
        points,
        num_angles=6,
        z_margins=(0.006,),
    )

    rim_xy = points[points[:, 2] >= np.quantile(points[:, 2], 0.72), :2]
    assert result["ok"]
    for candidate in result["candidates"]:
        pos_xy = np.asarray(candidate["pos"][:2], dtype=float)
        source_xy = np.asarray(candidate["source_rim_point_xy"], dtype=float)
        assert np.min(np.linalg.norm(rim_xy - source_xy, axis=1)) < 1e-9
        assert np.linalg.norm(pos_xy - source_xy) < 0.004
