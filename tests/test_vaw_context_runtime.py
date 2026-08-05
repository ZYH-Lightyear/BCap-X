from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime import FUNCTION_NAMES, ContextWorkspace, function_definitions
from vaw.context_runtime.geometry import (
    graspnet_pose_to_panda_hand,
    local_surface_depth,
)
from vaw.context_runtime.motion import CUROBO_TCP_TO_HAND_LOCAL_XYZ


class FakeContextApi:
    camera_name = "agentview"
    wrist_camera_name = "robot0_eye_in_hand"
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self) -> None:
        self.rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        self.depth = np.full((12, 16), 2.0, dtype=np.float64)
        self.intrinsics = np.array(
            [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.pose_mat = np.eye(4, dtype=np.float64)
        self.pose_mat[:3, 3] = [1.0, 2.0, 3.0]
        self.cartesian = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.joints = np.arange(7, dtype=np.float64) / 10.0
        self.bbox_outputs: list[list[float]] = []
        self.point_outputs: list[list[float]] = []
        self.solve_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.move_calls: list[np.ndarray] = []
        self.pointcloud_grasp_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.use_pointcloud_grasp = True
        self.solve_errors_remaining = 0
        self.orientation_used = "requested"
        self.move_error = False
        self._last_solve_position = self.cartesian[:3].copy()
        self._last_solve_quaternion = self.cartesian[3:7].copy()

    def get_observation(self):
        camera = {
            "images": {"rgb": self.rgb.copy(), "depth": self.depth.copy()},
            "intrinsics": self.intrinsics.copy(),
            "pose_mat": self.pose_mat.copy(),
        }
        return {
            "agentview": camera,
            "robot0_eye_in_hand": camera,
            "robot_cartesian_pos": self.cartesian.copy(),
            "robot_joint_pos": self.joints.copy(),
        }

    def vlm_bbox_detection(self, rgb, query):
        if self.bbox_outputs:
            return self.bbox_outputs.pop(0)
        height, width = rgb.shape[:2]
        return [1.0, 1.0, float(width - 1), float(height - 1)]

    def segment_sam3_box_prompt(self, rgb, box):
        return [{"mask": np.ones(rgb.shape[:2], dtype=bool), "score": 0.9}]

    def vlm_point_detection(self, rgb, query):
        if self.point_outputs:
            return self.point_outputs.pop(0)
        height, width = rgb.shape[:2]
        return [float(width // 2), float(height // 2)]

    def plan_grasp(self, depth, intrinsics, mask):
        first = np.eye(4, dtype=np.float64)
        first[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        first[:3, 3] = [0.1, 0.2, 0.3]
        second = np.eye(4, dtype=np.float64)
        second[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        second[:3, 3] = [0.4, 0.5, 0.6]
        return [first, second], [0.2, 0.8]

    def plan_grasp_from_point_clouds(self, pc_full, pc_segment):
        if not self.use_pointcloud_grasp:
            raise RuntimeError("point-cloud grasp disabled")
        self.pointcloud_grasp_calls.append(
            (
                np.asarray(pc_full, dtype=np.float64).copy(),
                np.asarray(pc_segment, dtype=np.float64).copy(),
            )
        )
        first = np.eye(4, dtype=np.float64)
        first[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        first[:3, 3] = [0.1, 0.2, 0.3]
        second = np.eye(4, dtype=np.float64)
        second[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        second[:3, 3] = [0.4, 0.5, 0.6]
        first = self.pose_mat @ first
        second = self.pose_mat @ second
        return [first, second], [0.2, 0.8]

    def filter_noise(self, points, colors=None):
        return np.asarray(points, dtype=np.float64), colors

    def solve_ik(self, position, quaternion_wxyz, *, return_info=False):
        position = np.asarray(position, dtype=np.float64).copy()
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
        self.solve_calls.append((position, quaternion))
        self._last_solve_position = position
        self._last_solve_quaternion = quaternion
        if self.solve_errors_remaining:
            self.solve_errors_remaining -= 1
            raise RuntimeError("solver unavailable")
        joints = np.arange(7, dtype=np.float64)
        if return_info:
            return joints, {"orientation_used": self.orientation_used}
        return joints

    def move_to_joints(self, joints):
        if self.move_error:
            raise RuntimeError("controller stalled")
        self.move_calls.append(np.asarray(joints, dtype=np.float64).copy())
        quaternion_xyzw = np.roll(self._last_solve_quaternion, -1)
        self.cartesian[:3] = self._last_solve_position + Rotation.from_quat(
            quaternion_xyzw
        ).apply(self._TCP_OFFSET)
        self.cartesian[3:7] = self._last_solve_quaternion

    def open_gripper(self):
        self.cartesian[7] = 1.0

    def close_gripper(self):
        self.cartesian[7] = 0.2


class FakeCuroboContextApi(FakeContextApi):
    def __init__(self) -> None:
        super().__init__()
        self.curobo_plan_calls: list[dict] = []
        self.curobo_execute_calls: list[np.ndarray] = []
        self.curobo_world_updates = 0
        self.curobo_success = True
        self.curobo_tracks_final = True
        self._curobo_target_position = np.zeros(3, dtype=np.float64)
        self._curobo_target_quaternion = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

    def update_curobo_world(self):
        self.curobo_world_updates += 1
        return {"world": self.curobo_world_updates}

    def plan_grasp_trajectory(
        self,
        object_name,
        *,
        object_mask,
        grasp_poses,
        **kwargs,
    ):
        position, quaternion = grasp_poses[0]
        self._curobo_target_position = np.asarray(position, dtype=np.float64).copy()
        self._curobo_target_quaternion = np.asarray(
            quaternion, dtype=np.float64
        ).copy()
        self.curobo_plan_calls.append(
            {
                "object_name": object_name,
                "object_mask": np.asarray(object_mask).copy(),
                "grasp_poses": grasp_poses,
                **kwargs,
            }
        )
        if not self.curobo_success:
            return False, None, None
        trajectory = np.stack(
            [
                self.joints.copy(),
                np.full(7, 0.25, dtype=np.float64),
                np.full(7, 0.5, dtype=np.float64),
            ]
        )
        return True, trajectory, 0

    def execute_joint_trajectory(self, trajectory, **kwargs):
        values = np.asarray(trajectory, dtype=np.float64).copy()
        self.curobo_execute_calls.append(values)
        if self.curobo_tracks_final:
            self.joints = values[-1].copy()
            quaternion_xyzw = np.roll(self._curobo_target_quaternion, -1)
            self.cartesian[:3] = self._curobo_target_position + Rotation.from_quat(
                quaternion_xyzw
            ).apply(np.asarray(CUROBO_TCP_TO_HAND_LOCAL_XYZ))
            self.cartesian[3:7] = self._curobo_target_quaternion
        return None


def test_m14_function_contract_is_exact_and_has_no_obb_or_legacy_ops() -> None:
    assert FUNCTION_NAMES == (
        "inspect",
        "locate_point",
        "propose_grasps",
        "propose_pose",
        "select",
        "delta_move",
        "rotate",
        "commit",
        "open_gripper",
        "close_gripper",
        "done",
    )
    definitions = function_definitions()
    assert [item["function"]["name"] for item in definitions] == list(FUNCTION_NAMES)
    encoded = json.dumps(definitions).lower()
    assert "obb" not in encoded
    for legacy in ("ground", "observe", "view", "preview", "move_xyz", "commit_gripper"):
        assert f'"{legacy}"' not in encoded
    by_name = {item["function"]["name"]: item["function"] for item in definitions}
    delta = by_name["delta_move"]["parameters"]["properties"]
    assert set(delta) == {"delta_xyz_m", "frame", "action_id"}
    assert delta["delta_xyz_m"]["items"]["minimum"] == -0.03
    assert delta["delta_xyz_m"]["items"]["maximum"] == 0.03
    rotate = by_name["rotate"]["parameters"]["properties"]
    assert set(rotate) == {"axis", "angle_deg", "frame", "action_id"}
    assert rotate["angle_deg"]["minimum"] == -90.0
    assert rotate["angle_deg"]["maximum"] == 90.0


def test_inspect_creates_region_without_serializing_private_arrays() -> None:
    api = FakeContextApi()
    api.bbox_outputs = [[2.0, 2.0, 10.0, 8.0]]
    workspace = ContextWorkspace(api, "put the mug in the basket")

    step = workspace.execute("inspect", query="basket")

    assert step.result == {
        "region_id": "region1",
        "bbox_xyxy_px": [2.0, 2.0, 10.0, 8.0],
    }
    assert step.manifest == {
        "revision": 1,
        "active_action_id": None,
        "valid_region_ids": ["region1"],
        "valid_point_ids": [],
        "valid_candidate_ids": [],
    }
    assert workspace.state.regions["region1"].query == "basket"
    assert workspace._private.region_masks["region1"].shape == (12, 16)
    public = json.dumps(step.manifest).lower()
    for forbidden in ("mask", "depth", "intrinsics", "pose_mat", "cloud", "obb"):
        assert forbidden not in public


def test_nested_inspect_restores_crop_coordinates_to_agentview() -> None:
    api = FakeContextApi()
    api.bbox_outputs = [
        [2.0, 2.0, 10.0, 8.0],
        [1.0, 1.0, 4.0, 3.0],
    ]
    workspace = ContextWorkspace(api, "pick the mug by its handle")
    parent = workspace.execute("inspect", query="mug").result["region_id"]

    child = workspace.execute(
        "inspect", query="handle", within_region_id=parent
    )

    assert child.result["bbox_xyxy_px"] == [3.0, 3.0, 6.0, 5.0]
    evidence = workspace.state.regions[child.result["region_id"]]
    assert evidence.within_region_id == parent


def test_locate_point_lifts_crop_point_to_base_xyz() -> None:
    api = FakeContextApi()
    api.bbox_outputs = [[2.0, 2.0, 10.0, 8.0]]
    api.point_outputs = [[2.0, 1.0]]
    workspace = ContextWorkspace(api, "place in the basket")
    region_id = workspace.execute("inspect", query="basket").result["region_id"]

    point = workspace.execute(
        "locate_point",
        query="free point near the center",
        within_region_id=region_id,
    )

    assert point.result["pixel_xy"] == [4.0, 3.0]
    assert point.result["position_xyz"] == [1.0, 2.0, 5.0]
    assert workspace.state.points["point1"].within_region_id == region_id


def test_point_pose_calls_solve_ik_without_workspace_bounds_gate() -> None:
    api = FakeContextApi()
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(
        api, "move above the basket", motion_backend="pyroki"
    )
    point_id = workspace.execute("locate_point", query="basket center").result["point_id"]

    proposal = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 10.0],
    )

    assert proposal.result["solve_ik"] == "returned"
    assert len(api.solve_calls) == 1
    position, quaternion_wxyz = api.solve_calls[0]
    assert np.allclose(position, [1.0, 2.0, 15.0])
    assert np.allclose(quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_delta_move_uses_observed_fingertip_tcp_and_does_not_execute() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "lift", motion_backend="pyroki")

    proposed = workspace.execute(
        "delta_move", delta_xyz_m=[0.03, -0.01, 0.02]
    )

    assert proposed.ok
    assert proposed.result["action_id"] == "a1"
    assert workspace.state.observation_revision == 1
    assert api.move_calls == []
    assert len(api.solve_calls) == 1
    target_position, _ = api.solve_calls[0]
    # Observed panda_hand is [0.4, 0.0, 0.3].  Inverting the local -0.1 m
    # hand offset yields fingertip TCP [0.4, 0.0, 0.4] before the delta.
    assert np.allclose(target_position, [0.43, -0.01, 0.42])
    adjustment = workspace.state.active_action.adjustment
    assert adjustment is not None
    assert adjustment.parent_action_id is None
    assert adjustment.frame == "base"
    assert np.allclose(adjustment.reference_pose.position_xyz, [0.4, 0.0, 0.4])


def test_delta_move_enforces_inclusive_per_axis_limit_without_clamping() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")

    boundary = workspace.execute(
        "delta_move", delta_xyz_m=[-0.03, 0.03, 0.001]
    )
    too_large = workspace.execute(
        "delta_move", delta_xyz_m=[0.030001, 0.0, 0.0]
    )
    zero = workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.0])
    non_finite = workspace.execute(
        "delta_move", delta_xyz_m=[float("nan"), 0.0, 0.0]
    )
    bad_frame = workspace.execute(
        "delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="camera"
    )

    assert boundary.ok
    assert "[-0.03, 0.03]" in too_large.result["error"]
    assert "non-zero" in zero.result["error"]
    assert "finite" in non_finite.result["error"]
    assert "frame" in bad_frame.result["error"]
    assert len(api.solve_calls) == 1


def test_tool_frame_delta_uses_reference_tcp_orientation() -> None:
    api = FakeContextApi()
    reference = Rotation.from_euler("z", 90.0, degrees=True)
    api.cartesian[3:7] = np.roll(reference.as_quat(), 1)
    workspace = ContextWorkspace(api, "tool move", motion_backend="pyroki")

    workspace.execute(
        "delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="tool"
    )

    target_position, _ = api.solve_calls[-1]
    assert np.allclose(target_position, [0.4, 0.01, 0.4], atol=1e-8)


def test_rotate_composes_base_and_tool_frames_in_the_documented_order() -> None:
    reference = Rotation.from_euler("z", 90.0, degrees=True)
    delta = Rotation.from_euler("x", 30.0, degrees=True)

    api_base = FakeContextApi()
    api_base.cartesian[3:7] = np.roll(reference.as_quat(), 1)
    base = ContextWorkspace(api_base, "rotate", motion_backend="pyroki")
    base_step = base.execute("rotate", axis="x", angle_deg=30.0, frame="base")

    api_tool = FakeContextApi()
    api_tool.cartesian[3:7] = np.roll(reference.as_quat(), 1)
    tool = ContextWorkspace(api_tool, "rotate", motion_backend="pyroki")
    tool_step = tool.execute("rotate", axis="x", angle_deg=30.0, frame="tool")

    assert base_step.ok and tool_step.ok
    base_quaternion = np.roll(api_base.solve_calls[-1][1], -1)
    tool_quaternion = np.roll(api_tool.solve_calls[-1][1], -1)
    assert np.allclose(
        Rotation.from_quat(base_quaternion).as_matrix(),
        (delta * reference).as_matrix(),
    )
    assert np.allclose(
        Rotation.from_quat(tool_quaternion).as_matrix(),
        (reference * delta).as_matrix(),
    )


def test_rotate_validates_axis_angle_and_keeps_position_fixed() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "rotate", motion_backend="pyroki")

    boundary = workspace.execute("rotate", axis="z", angle_deg=-90.0)
    too_large = workspace.execute("rotate", axis="z", angle_deg=90.001)
    zero = workspace.execute("rotate", axis="z", angle_deg=0.0)
    bad_axis = workspace.execute("rotate", axis="roll", angle_deg=10.0)
    bad_frame = workspace.execute(
        "rotate", axis="x", angle_deg=10.0, frame="camera"
    )

    assert boundary.ok
    assert np.allclose(api.solve_calls[0][0], [0.4, 0.0, 0.4])
    assert "[-90, 90]" in too_large.result["error"]
    assert "non-zero" in zero.result["error"]
    assert "axis" in bad_axis.result["error"]
    assert "frame" in bad_frame.result["error"]
    assert len(api.solve_calls) == 1


def test_incremental_edit_replaces_active_action_and_exact_cached_plan() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")
    first = workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0])
    first_id = first.result["action_id"]
    first_target = workspace.state.active_action.target_pose

    second = workspace.execute(
        "delta_move",
        delta_xyz_m=[0.02, 0.0, 0.0],
        action_id=first_id,
    )
    second_id = second.result["action_id"]

    assert second_id == "a2"
    assert np.allclose(
        workspace.state.active_action.target_pose.position_xyz,
        np.asarray(first_target.position_xyz) + [0.02, 0.0, 0.0],
    )
    assert workspace.state.active_action.adjustment.parent_action_id == first_id
    assert first_id not in workspace._private.motion_plans
    assert second_id in workspace._private.motion_plans
    assert workspace.state.observation_revision == 1
    assert workspace.execute("commit", action_id=first_id).result == {
        "error": f"unknown or expired action_id '{first_id}'"
    }

    committed = workspace.execute("commit", action_id=second_id)

    assert committed.ok
    assert len(api.solve_calls) == 2
    assert len(api.move_calls) == 1


def test_solve_ik_error_keeps_action_and_does_not_gate_later_commit() -> None:
    api = FakeContextApi()
    api.point_outputs = [[4.0, 3.0]]
    api.solve_errors_remaining = 1
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")
    point_id = workspace.execute("locate_point", query="target").result["point_id"]
    proposal = workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.0, 0.0, 0.1]
    )

    assert proposal.result["solve_ik"] == "error"
    assert workspace.state.active_action is not None

    committed = workspace.execute("commit", action_id=proposal.result["action_id"])
    assert committed.ok
    assert len(api.move_calls) == 1
    assert workspace.state.observation_revision == 2


def test_commit_reuses_the_exact_joints_returned_by_prediction() -> None:
    api = FakeContextApi()
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")
    point_id = workspace.execute("locate_point", query="target").result["point_id"]
    proposal = workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.0, 0.0, 0.1]
    )

    committed = workspace.execute("commit", action_id=proposal.result["action_id"])

    assert committed.ok
    assert len(api.solve_calls) == 1
    assert len(api.move_calls) == 1
    assert np.allclose(api.move_calls[0], np.arange(7))


def test_commit_position_error_compares_achieved_tcp_with_target_tcp() -> None:
    api = FakeContextApi()
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")
    point_id = workspace.execute("locate_point", query="target").result["point_id"]
    proposal = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.1],
        quaternion_xyzw=[1.0, 0.0, 0.0, 0.0],
    )

    target_position = api.solve_calls[-1][0].copy()
    committed = workspace.execute("commit", action_id=proposal.result["action_id"])

    # Top-down rotates the local -z hand offset to +z in the base frame.  The
    # observed hand origin is therefore 0.1 m above the requested contact TCP.
    assert np.allclose(api.cartesian[:3], target_position + [0.0, 0.0, 0.1])
    assert committed.result["position_error_m"] == 0.0
    assert workspace.state.last_receipt is not None
    assert workspace.state.last_receipt.position_error_m == 0.0


def test_failed_physical_call_refreshes_and_keeps_action_attribution() -> None:
    api = FakeContextApi()
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(api, "move", motion_backend="pyroki")
    point_id = workspace.execute("locate_point", query="target").result["point_id"]
    action_id = workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.0, 0.0, 0.1]
    ).result["action_id"]
    api.move_error = True

    failed = workspace.execute("commit", action_id=action_id)

    assert failed.result == {"error": "move_to_joints failed: controller stalled"}
    assert failed.revision_after == 2
    assert workspace.state.last_receipt is not None
    assert workspace.state.last_receipt.action_id == action_id
    assert workspace.state.recent_calls[-1].action_id == action_id
    assert workspace.state.active_action is None


def test_physical_revision_expires_evidence_and_history_is_three_transactions() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "open then inspect")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("locate_point", query="mug center")
    opened = workspace.execute("open_gripper")

    assert opened.result == {"gripper_opening": 1.0}
    assert opened.trace_receipt is not None
    assert opened.trace_receipt["receipt_id"] == "receipt1"
    assert opened.trace_receipt["revision_before"] == 1
    assert opened.trace_receipt["revision_after"] == 2
    assert workspace.state.regions == {}
    assert workspace.state.points == {}
    assert workspace.state.candidates == {}

    expired = workspace.execute("propose_grasps", region_id=region_id)
    assert expired.result == {"error": f"unknown or expired region_id '{region_id}'"}
    assert len(workspace.state.recent_calls) == 3
    assert [record.function_name for record in workspace.state.recent_calls] == [
        "locate_point",
        "open_gripper",
        "propose_grasps",
    ]


def test_grasp_candidates_cache_ik_and_selection_reuses_it() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="pyroki")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    candidates = workspace.execute("propose_grasps", region_id=region_id)

    assert candidates.result == {"candidate_ids": ["g1", "g2"]}
    assert len(api.solve_calls) == 2
    cached = workspace.state.candidates["g1"].prediction.joint_positions_rad
    selected = workspace.execute("select", candidate_id="g1")
    assert selected.result["action_id"] == "a1"
    assert selected.result["solve_ik"] == "returned"
    assert workspace.state.active_action.source_ref == "g1"
    assert workspace.state.active_action.prediction.joint_positions_rad == cached
    assert len(api.solve_calls) == 2


def test_candidate_preview_rejects_solve_ik_orientation_fallback() -> None:
    api = FakeContextApi()
    api.orientation_used = "top-down"
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="pyroki")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]

    workspace.execute("propose_grasps", region_id=region_id)

    prediction = workspace.state.candidates["g1"].prediction
    assert prediction.solve_ik == "error"
    assert prediction.joint_positions_rad is None
    assert "substituted orientation 'top-down'" in prediction.detail


def test_propose_grasps_uses_single_view_planner() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="pyroki")

    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    candidates = workspace.execute("propose_grasps", region_id=region_id)

    assert candidates.result == {"candidate_ids": ["g1", "g2"]}
    assert api.pointcloud_grasp_calls == []
    diagnostic = candidates.trace_diagnostics["grasp_candidates"]
    assert diagnostic["selected_source"] == "single_view"
    assert [attempt["source"] for attempt in diagnostic["attempts"]] == [
        "single_view"
    ]
    public = json.dumps(candidates.result).lower()
    for forbidden in ("mask", "depth", "intrinsics", "pose_mat", "cloud"):
        assert forbidden not in public


def test_propose_grasps_does_not_call_multiview_planner_when_available() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="pyroki")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]

    candidates = workspace.execute("propose_grasps", region_id=region_id)

    assert candidates.result == {"candidate_ids": ["g1", "g2"]}
    assert api.pointcloud_grasp_calls == []


def test_graspnet_pose_is_adapted_to_panda_hand_axes() -> None:
    graspnet_pose = np.eye(4, dtype=np.float64)
    graspnet_pose[:3, 3] = [0.1, 0.2, 0.3]

    hand_pose = graspnet_pose_to_panda_hand(graspnet_pose)

    assert np.allclose(hand_pose[:3, 3], graspnet_pose[:3, 3])
    assert np.allclose(hand_pose[:3, 2], graspnet_pose[:3, 2])
    assert np.allclose(hand_pose[:3, 1], -graspnet_pose[:3, 0])

    api = FakeContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="pyroki")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("propose_grasps", region_id=region_id)

    expected_rotation = Rotation.from_matrix(
        Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
        @ Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    )
    expected_wxyz = np.roll(expected_rotation.as_quat(), 1)
    assert np.allclose(api.solve_calls[0][1], expected_wxyz)


def test_context_workspace_defaults_to_curobo() -> None:
    workspace = ContextWorkspace(FakeCuroboContextApi(), "pick the mug")

    assert workspace.motion_backend_name == "curobo"


def test_curobo_select_plans_one_candidate_and_commit_executes_cached_trajectory() -> None:
    api = FakeCuroboContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="curobo")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("propose_grasps", region_id=region_id)

    selected = workspace.execute("select", candidate_id="g1")

    assert selected.result == {
        "action_id": "a1",
        "solve_ik": "returned",
        "trajectory_checked": True,
        "collision_checked": True,
    }
    assert len(api.curobo_plan_calls) == 1
    call = api.curobo_plan_calls[0]
    assert call["object_name"] == "mug"
    assert call["top_k_grasps"] == 1
    assert call["use_world_collision"] is True
    assert call["world_config"] is None
    assert call["grasp_pose_is_fingertip"] is True
    assert call["object_mask"].shape == api.rgb.shape[:2]
    cached = workspace._private.motion_plans["a1"].trajectory_rad.copy()

    committed = workspace.execute("commit", action_id="a1")

    assert committed.ok
    assert committed.result["position_error_m"] == 0.0
    assert len(api.curobo_execute_calls) == 1
    assert np.array_equal(api.curobo_execute_calls[0], cached)
    assert workspace.state.observation_revision == 2


def test_refined_grasp_preserves_source_and_grasp_aware_curobo_planning() -> None:
    api = FakeCuroboContextApi()
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="curobo")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("propose_grasps", region_id=region_id)
    selected = workspace.execute("select", candidate_id="g1")
    first_id = selected.result["action_id"]

    refined = workspace.execute(
        "rotate",
        axis="z",
        angle_deg=10.0,
        frame="tool",
        action_id=first_id,
    )

    assert refined.ok
    assert refined.result["action_id"] == "a2"
    assert workspace.state.active_action.source_ref == "g1"
    assert first_id not in workspace._private.motion_plans
    assert len(api.curobo_plan_calls) == 2
    adjusted_call = api.curobo_plan_calls[-1]
    assert adjusted_call["object_name"] == "mug"
    assert adjusted_call["world_config"] is None
    assert adjusted_call["object_mask"].shape == api.rgb.shape[:2]


def test_curobo_pose_builds_current_world_without_an_object_mask_dependency() -> None:
    api = FakeCuroboContextApi()
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(api, "move above basket", motion_backend="curobo")
    point_id = workspace.execute("locate_point", query="basket center").result[
        "point_id"
    ]

    proposed = workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.0, 0.0, 0.15]
    )

    assert proposed.result["trajectory_checked"] is True
    assert api.curobo_world_updates == 1
    call = api.curobo_plan_calls[0]
    assert call["object_name"] == "vaw_pose_target"
    assert call["world_config"] == {"world": 1}
    assert call["object_mask"].shape == (1, 1)


def test_curobo_plan_failure_keeps_proposal_but_commit_cannot_silently_fallback() -> None:
    api = FakeCuroboContextApi()
    api.curobo_success = False
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="curobo")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("propose_grasps", region_id=region_id)

    selected = workspace.execute("select", candidate_id="g1")

    assert selected.result["solve_ik"] == "error"
    assert selected.result["trajectory_checked"] is False
    assert workspace.state.active_action is not None

    committed = workspace.execute("commit", action_id="a1")

    assert committed.result == {"error": "CuRobo found no collision-free trajectory"}
    assert api.move_calls == []
    assert api.curobo_execute_calls == []
    assert workspace.state.observation_revision == 2


def test_curobo_commit_rejects_silent_final_joint_nonconvergence() -> None:
    api = FakeCuroboContextApi()
    api.curobo_tracks_final = False
    workspace = ContextWorkspace(api, "pick the mug", motion_backend="curobo")
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("propose_grasps", region_id=region_id)
    action_id = workspace.execute("select", candidate_id="g1").result["action_id"]

    committed = workspace.execute("commit", action_id=action_id)

    assert "joint residual" in committed.result["error"]
    assert workspace.state.last_receipt is not None
    assert workspace.state.last_receipt.action_id == action_id
    assert workspace.state.observation_revision == 2


def test_curobo_receipt_uses_the_same_native_fingertip_offset_as_planning() -> None:
    api = FakeCuroboContextApi()
    workspace = ContextWorkspace(api, "pick", motion_backend="curobo")

    assert np.allclose(
        workspace._tcp_to_hand_local_xyz,
        [0.0, 0.0, -0.1168],
    )


def test_dispatch_errors_are_structured_without_a_phase_machine() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task")

    missing = workspace.execute("propose_pose", point_id="point999", offset_xyz=[0, 0, 0])
    unknown = workspace.execute("not_a_function")

    assert missing.result == {"error": "unknown or expired point_id 'point999'"}
    assert unknown.result == {"error": "unknown function 'not_a_function'"}
    assert not hasattr(workspace, "phase")
    assert not hasattr(workspace, "allowed_next_tools")


def test_malformed_action_is_recorded_as_an_error_transaction() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task")

    step = workspace.execute_action("not json")

    assert step.function_name == "invalid"
    assert set(step.result) == {"error"}
    assert workspace.state.recent_calls[-1].function_name == "invalid"


def test_bbox_is_clamped_to_the_current_search_image() -> None:
    api = FakeContextApi()
    api.bbox_outputs = [[-20.0, -5.0, 100.0, 80.0]]
    workspace = ContextWorkspace(api, "inspect")

    result = workspace.execute("inspect", query="table")

    assert result.result["bbox_xyxy_px"] == [0.0, 0.0, 15.0, 11.0]


def test_local_surface_depth_rejects_an_equidistant_depth_edge() -> None:
    depth = np.zeros((5, 5), dtype=np.float64)
    depth[2, 1] = 0.4
    depth[2, 3] = 0.9

    with np.testing.assert_raises_regex(ValueError, "ambiguous depth discontinuity"):
        local_surface_depth(depth, (2.0, 2.0), radius_px=1)
