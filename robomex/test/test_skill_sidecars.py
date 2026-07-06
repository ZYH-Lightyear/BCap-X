from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def _load(rel: str, name: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _disc_points(center=(0.4, 0.2, 0.02), radius=0.06, z=0.02) -> np.ndarray:
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


def _bowl_points() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    rim = np.column_stack([0.5 + 0.055 * np.cos(angles), 0.15 + 0.055 * np.sin(angles), np.full_like(angles, 0.08)])
    lower = np.column_stack(
        [0.5 + 0.035 * np.cos(angles), 0.15 + 0.035 * np.sin(angles), np.full_like(angles, 0.035)]
    )
    bottom = np.column_stack(
        [0.5 + 0.018 * np.cos(angles), 0.15 + 0.018 * np.sin(angles), np.full_like(angles, 0.02)]
    )
    return np.vstack([rim, lower, bottom])


def test_all_builtin_sidecar_scripts_are_plain_python_importable() -> None:
    for script in sorted(ROOT.glob("*/*/scripts/*.py")):
        spec = importlib.util.spec_from_file_location(f"sidecar_{script.stem}", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.__name__.startswith("sidecar_")


def test_builtin_scripts_do_not_execute_robot_actions_or_use_legacy_evidence_keys() -> None:
    blocked_terms = {
        "goto_pose(",
        "open_gripper(",
        "close_gripper(",
        "move_to_joints(",
        "execute_joint_trajectory(",
        "step(",
        "target_box",
        "vlm_box_image",
        "sam3_mask_image",
        "place_point",
        "place_strategy",
        "fallback_place_pos",
        "target_points",
        "bowl_points",
    }
    for script in sorted(ROOT.glob("*/*/scripts/*.py")):
        text = script.read_text(encoding="utf-8")
        for term in blocked_terms:
            assert term not in text, (script, term)


def test_segment_object_sidecar_runs_live_observation_contract(tmp_path) -> None:
    module = _load("perception/segment_object/scripts/segment_object.py", "segment_object_sidecar")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    obs = {
        "agentview": {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": np.eye(3),
            "pose_mat": np.eye(4),
        }
    }
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    points = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=float)

    evidence = {}
    result = module.ground_object_from_observation(
        obs,
        target_name="test bowl",
        artifacts_dir=str(tmp_path),
        evidence=evidence,
        apis={
            "vlm_bbox_detection": lambda _rgb, _target: [2, 2, 5, 5],
            "segment_sam3_box_prompt": lambda _rgb, _box: [{"mask": mask, "score": 0.9}],
            "mask_to_world_points": lambda _mask, _depth, _intr, _pose: points,
            "filter_noise": lambda values: values,
        },
    )

    assert result["target"] == "test bowl"
    assert result["center_xyz"] == [0.15000000000000002, 0.2, 0.3]
    assert evidence["object_grounding"]["points_key"] == "grounding.points"


def test_affordance_motion_sidecars_are_callable_after_plain_import() -> None:
    placement_mod = _load("affordance/find_placement/scripts/placement_affordance.py", "placement_sidecar")
    bowl_mod = _load("affordance/grasp_open_bowl/scripts/bowl_grasp.py", "bowl_sidecar")
    release_mod = _load("motion/release_at/scripts/release.py", "release_sidecar")

    placement = placement_mod.estimate_placement_affordance(
        _disc_points(),
        target_name="plate",
    )
    assert placement["mode"] == "support_surface"

    bowl = bowl_mod.propose_open_bowl_grasps(
        _bowl_points(),
        num_angles=4,
    )
    assert bowl["ok"]
    assert bowl["selected_candidate"]["strategy"] == "open_bowl_rim_topdown"
    selected = bowl["selected_candidate"]
    for key in (
        "pos",
        "quat",
        "object_center",
        "object_center_offset_from_grasp",
        "pregrasp_pos",
        "ik_ok",
    ):
        assert key in selected
    assert "grasp_pos" not in selected
    assert "position" not in selected

    tcp = release_mod.compute_tcp_release_pos(
        {"desired_object_center": [0.5, 0.1, 0.2]},
        {"object_center_offset_from_grasp": [0.04, -0.02, 0.0]},
    )
    assert np.allclose(tcp, [0.46, 0.12, 0.2])
