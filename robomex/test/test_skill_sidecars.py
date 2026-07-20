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


def _basket_points(center=(0.6, 0.25, 0.05), half_xy=0.08, bottom_z=0.01, rim_z=0.14) -> np.ndarray:
    """Open container cloud: dense rim + sparse interior floor."""

    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    rim = np.column_stack(
        [
            center[0] + half_xy * np.cos(angles),
            center[1] + half_xy * np.sin(angles),
            np.full_like(angles, rim_z),
        ]
    )
    floor_angles = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    floor = []
    for scale in (0.2, 0.45, 0.7):
        floor.append(
            np.column_stack(
                [
                    center[0] + half_xy * scale * np.cos(floor_angles),
                    center[1] + half_xy * scale * np.sin(floor_angles),
                    np.full_like(floor_angles, bottom_z),
                ]
            )
        )
    return np.vstack([rim, *floor])


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
    obb_mod = _load("affordance/compute_obb_short_axis_grasp/scripts/obb_short_axis.py", "obb_short_axis")
    top_mod = _load("affordance/top_grasp_with_tcp_to_bottom_offset/scripts/top_grasp_offset.py", "top_offset")
    verify_mod = _load(
        "verification/verify_grasp_and_lift_via_robot_state/scripts/verify_grasp_state.py",
        "verify_grasp_state",
    )

    placement = placement_mod.estimate_placement_affordance(
        _disc_points(),
        target_name="plate",
    )
    assert placement["mode"] == "support_surface"
    assert "position" in placement
    assert "desired_object_center" in placement

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
        "object_center_at_grasp",
        "object_center_offset_from_grasp",
        "pregrasp_pos",
        "rim_height",
        "contact_depth",
        "ik_ok",
    ):
        assert key in selected
    assert "grasp_pos" not in selected
    assert selected["position"] == selected["pos"]
    assert selected["quaternion_wxyz"] == selected["quat"]
    assert selected["approach_position"] == selected["pregrasp_pos"]
    assert selected["lift_position"][2] > selected["position"][2]

    tcp = release_mod.compute_tcp_release_pos(
        {"desired_object_center": [0.5, 0.1, 0.2]},
        {"object_center_offset_from_grasp": [0.04, -0.02, 0.0]},
    )
    assert np.allclose(tcp, [0.46, 0.12, 0.2])
    tcp_prefers_position = release_mod.compute_tcp_release_pos(
        {"position": [0.5, 0.1, 0.15], "desired_object_center": [0.5, 0.1, 0.2]},
    )
    assert np.allclose(tcp_prefers_position, [0.5, 0.1, 0.15])
    checklist = release_mod.release_execution_checklist()
    assert any("does not automatically stop" in step for step in checklist)
    calls: list[dict] = []

    def _open(**kwargs):
        calls.append(kwargs)

    release_mod.settle_after_open(_open, settle_steps=60)
    assert calls and calls[0].get("settle_steps") == 60

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.2, 0.04, 0.0],
            [0.1, 0.02, 0.03],
        ],
        dtype=float,
    )
    obb = {"center": np.array([0.1, 0.02, 0.015]), "extent": np.array([0.2, 0.04, 0.03]), "R": np.eye(3)}
    grasp = obb_mod.compute_obb_short_axis_grasp(points, obb)
    assert grasp["ok"]
    assert grasp["strategy"] == "obb_short_axis_topdown"
    assert "pos" in grasp and "quat" in grasp

    top = top_mod.compute_top_grasp_with_tcp_to_bottom_offset(points, obb=obb)
    assert top["ok"]
    assert "tcp_to_bottom" in top
    assert "object_center_offset_from_grasp" in top

    gripper_state = {"gripper_open_ratio": 0.42, "close_gripper_executed": True}
    scene_question = verify_mod.build_scene_verification_question(
        "black bowl", gripper_state
    )
    wrist_question = verify_mod.build_wrist_verification_question(
        "black bowl", gripper_state
    )
    assert "black bowl" in scene_question
    assert "black bowl" in wrist_question
    assert '"gripper_open_ratio": 0.42' in scene_question
    assert "fixed thresholds" in scene_question
    parsed = verify_mod.parse_verification_response(
        '```json\n{"visibility":"visible","status":"held",'
        '"confidence":"high","reason":"coupled"}\n```'
    )
    assert parsed == {
        "visibility": "visible",
        "status": "held",
        "confidence": "high",
        "reason": "coupled",
    }


def test_grasp_verification_uses_scene_when_visible() -> None:
    verify_mod = _load(
        "verification/verify_grasp_and_lift_via_robot_state/scripts/verify_grasp_state.py",
        "verify_grasp_state_scene",
    )
    calls = []

    def query(question, *, images):
        calls.append((question, images))
        return {
            "visibility": "visible",
            "status": "held",
            "confidence": "high",
            "reason": "target moves with the visible gripper",
        }

    gripper_state = {"gripper_open_ratio": 0.79, "close_gripper_executed": True}
    verdict = verify_mod.verify_grasp_with_vlm(
        target_name="alphabet soup can",
        gripper_state=gripper_state,
        scene_image="scene",
        wrist_image="wrist",
        query_vlm=query,
    )
    assert verdict["success"]
    assert verdict["state"] == "held"
    assert verdict["view_used"] == "scene"
    assert verdict["gripper_state"] == gripper_state
    assert len(calls) == 1
    assert calls[0][1] == "scene"


def test_grasp_verification_falls_back_to_wrist_only_after_occlusion() -> None:
    verify_mod = _load(
        "verification/verify_grasp_and_lift_via_robot_state/scripts/verify_grasp_state.py",
        "verify_grasp_state_wrist",
    )
    calls = []
    replies = iter(
        [
            {
                "visibility": "occluded",
                "status": "uncertain",
                "confidence": "low",
                "reason": "arm blocks the target",
            },
            {
                "visibility": "visible",
                "status": "empty_or_slipped",
                "confidence": "high",
                "reason": "no target coupled to the gripper",
            },
        ]
    )

    def query(question, *, images):
        calls.append((question, images))
        return next(replies)

    verdict = verify_mod.verify_grasp_with_vlm(
        target_name="alphabet soup can",
        gripper_state={"gripper_open_ratio": 0.12},
        scene_image="scene",
        wrist_image="wrist",
        query_vlm=query,
    )
    assert not verdict["success"]
    assert verdict["state"] == "empty_or_slipped"
    assert verdict["view_used"] == "wrist"
    assert set(verdict["observations"]) == {"scene", "wrist"}
    assert [image for _, image in calls] == ["scene", "wrist"]
    assert "0.12" in calls[0][0]
    assert "0.12" in calls[1][0]


def test_gap_style_open_container_drop_keeps_object_inside_cavity() -> None:
    placement_mod = _load(
        "affordance/find_placement/scripts/placement_affordance.py",
        "placement_gap_container",
    )
    held = {
        "object_center_offset_from_grasp": [0.0, 0.0, 0.02],
        "object_center_at_grasp": [0.4, 0.0, 0.05],
        "top_z": 0.08,
        "bottom_z": 0.02,
    }
    affordance = placement_mod.compute_drop_affordance(
        _basket_points(),
        target_name="wicker basket",
        mode="open_container",
        held_object_frame=held,
        grasp_ee_z=0.07,
    )
    assert affordance["mode"] == "open_container"
    assert affordance["desired_object_center"][2] <= affordance["rim_top"] + 1e-9
    assert affordance["desired_object_center"][2] < affordance["rim_top"] + 0.10
    # TCP uses offset / ee_to_obj_z and must not equal the object center.
    assert not np.allclose(affordance["position"], affordance["desired_object_center"])
    assert affordance["approach_position"][2] > affordance["position"][2]


def test_gap_style_drop_without_held_frame_avoids_rim_plus_10cm() -> None:
    placement_mod = _load(
        "affordance/find_placement/scripts/placement_affordance.py",
        "placement_gap_no_held",
    )
    affordance = placement_mod.compute_drop_affordance(
        _basket_points(rim_z=0.14),
        target_name="basket",
        mode="open_container",
    )
    rim = affordance["rim_top"]
    # Old heuristic was rim + 0.10; GaP-style must stay in/near the cavity.
    assert affordance["desired_object_center"][2] <= rim + 1e-6
    assert affordance["position"][2] < rim + 0.10


def test_build_place_trajectory_uses_tcp_not_object_center() -> None:
    traj_mod = _load(
        "motion/plan_bounded_motion/scripts/place_trajectory.py",
        "place_trajectory",
    )
    affordance = {
        "position": [0.61, 0.27, 0.11],
        "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
        "desired_object_center": [0.61, 0.27, 0.09],
        "approach_position": [0.61, 0.27, 0.31],
        "approach_height": 0.20,
    }
    traj = traj_mod.build_place_trajectory(affordance)
    assert traj["feasible"] is True
    names = [w["name"] for w in traj["waypoints"]]
    phases = [w["phase"] for w in traj["waypoints"]]
    assert names == ["transport_hover", "release_descend", "open_settle", "retreat_up"]
    assert phases == ["transport", "release", "open", "retreat"]
    release = traj["waypoints"][1]
    assert release["position_xyz"] == [0.61, 0.27, 0.11]
    assert release["position_xyz"] != affordance["desired_object_center"]
    assert traj["waypoints"][2]["gripper"] == "open"
    assert traj["waypoints"][0]["gripper"] == "hold"
