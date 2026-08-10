from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from vaw.context_runtime import (
    FUNCTION_NAMES,
    ActionPrediction,
    ContextWorkspace,
    function_definitions,
)
from vaw.context_runtime.motion import CUROBO_TCP_TO_HAND_LOCAL_XYZ, MotionPlan
from vaw.context_runtime.protocol import (
    IMAGINATION_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    imagination_function_definitions,
)


class FakeContextApi:
    camera_name = "agentview"
    wrist_camera_name = "robot0_eye_in_hand"
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self) -> None:
        self.rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        self.depth = np.full((120, 160), 0.5, dtype=np.float64)
        self.intrinsics = np.array(
            [[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
        )
        self.pose_mat = np.eye(4, dtype=np.float64)
        self.cartesian = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.joints = np.arange(7, dtype=np.float64) / 10.0
        self.solve_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.move_calls: list[np.ndarray] = []
        self.operation_log: list[str] = []
        self.move_error = False
        self.gripper_error = False
        self._last_position = self.cartesian[:3].copy()
        self._last_quaternion = self.cartesian[3:7].copy()

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
        del query
        h, w = rgb.shape[:2]
        return [w * 0.35, h * 0.25, w * 0.65, h * 0.8]

    def segment_sam3_box_prompt(self, rgb, box):
        del box
        return [{"mask": np.ones(rgb.shape[:2], dtype=bool), "score": 0.9}]

    def vlm_point_detection(self, rgb, query):
        del query
        h, w = rgb.shape[:2]
        return [w / 2, h / 2]

    def plan_grasp(self, depth, intrinsics, mask):
        del depth, intrinsics, mask
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        pose[:3, 3] = [0.0, 0.0, 0.45]
        return [pose], [0.8]

    def solve_ik(self, position, quaternion_wxyz, *, return_info=False):
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        self.solve_calls.append((position.copy(), quaternion.copy()))
        self._last_position = position.copy()
        self._last_quaternion = quaternion.copy()
        joints = np.arange(7, dtype=np.float64)
        return (joints, {"orientation_used": "requested"}) if return_info else joints

    def move_to_joints(self, joints):
        self.operation_log.append("arm")
        if self.move_error:
            raise RuntimeError("controller stalled")
        self.move_calls.append(np.asarray(joints, dtype=np.float64).copy())
        quat_xyzw = np.roll(self._last_quaternion, -1)
        self.cartesian[:3] = self._last_position + Rotation.from_quat(quat_xyzw).apply(
            self._TCP_OFFSET
        )
        self.cartesian[3:7] = self._last_quaternion

    def open_gripper(self):
        self.operation_log.append("gripper:open")
        if self.gripper_error:
            raise RuntimeError("gripper stalled")
        self.cartesian[7] = 1.0

    def close_gripper(self):
        self.operation_log.append("gripper:closed")
        if self.gripper_error:
            raise RuntimeError("gripper stalled")
        self.cartesian[7] = 0.1


class FakeCuroboContextApi(FakeContextApi):
    def __init__(self) -> None:
        super().__init__()
        self.curobo_execute_calls: list[np.ndarray] = []
        self._target_position = np.zeros(3)
        self._target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def update_curobo_world(self):
        return {"world": 1}

    def plan_grasp_trajectory(self, object_name, *, object_mask, grasp_poses, **kwargs):
        del object_name, object_mask, kwargs
        self._target_position = np.asarray(grasp_poses[0][0]).copy()
        self._target_quaternion = np.asarray(grasp_poses[0][1]).copy()
        return True, np.stack([self.joints, np.full(7, 0.5)]), 0

    def execute_joint_trajectory(self, trajectory, **kwargs):
        del kwargs
        values = np.asarray(trajectory).copy()
        self.curobo_execute_calls.append(values)
        self.joints = values[-1]
        quat_xyzw = np.roll(self._target_quaternion, -1)
        self.cartesian[:3] = self._target_position + Rotation.from_quat(
            quat_xyzw
        ).apply(np.asarray(CUROBO_TCP_TO_HAND_LOCAL_XYZ))
        self.cartesian[3:7] = self._target_quaternion


class FailedMotionBackend:
    name = "failed-test"
    tcp_to_hand_local_xyz = (0.0, 0.0, -0.1)

    @staticmethod
    def _plan() -> MotionPlan:
        return MotionPlan(
            "failed-test",
            ActionPrediction(solve_ik="error", detail="no executable plan"),
        )

    def preview(self, target):
        del target
        return self._plan().prediction

    def plan_pose(self, target, *, preview=None):
        del target, preview
        return self._plan()

    def plan_grasp(self, target, *, object_name, object_mask, preview=None):
        del target, object_name, object_mask, preview
        return self._plan()

    def execute(self, plan, target):
        raise AssertionError(f"failed plan must not execute: {plan}, {target}")


def test_dual_agent_function_contracts_are_disjoint_and_small() -> None:
    assert len(FUNCTION_NAMES) == 11
    assert FUNCTION_NAMES[0] == "detection_and_sam"
    assert "inspect" not in FUNCTION_NAMES
    assert IMAGINATION_FUNCTION_NAMES == (
        "delta_move",
        "rotate",
        "open_gripper",
        "close_gripper",
        "finish_imagination",
    )
    main = function_definitions()
    imagination = imagination_function_definitions()
    assert [item["function"]["name"] for item in main] == list(FUNCTION_NAMES)
    assert [item["function"]["name"] for item in imagination] == list(
        IMAGINATION_FUNCTION_NAMES
    )
    encoded = json.dumps(main + imagination).lower()
    assert "history" not in encoded and "receipt" not in encoded and "obb" not in encoded
    detection = main[0]["function"]
    assert "bbox detection" in detection["description"]
    assert "sam" in detection["description"].lower()
    assert "不刷新 observation" in detection["description"]
    assert "commit 后应重新观察真实画面" not in SYSTEM_PROMPT
    for name in ("delta_move", "rotate"):
        definition = next(x["function"] for x in main if x["function"]["name"] == name)
        assert "frame" in definition["parameters"]["required"]
        assert "refinement_goal" in definition["parameters"]["required"]
    for name in ("select", "propose_pose", "open_gripper", "close_gripper"):
        definition = next(x["function"] for x in main if x["function"]["name"] == name)
        assert "refinement_goal" in definition["parameters"]["required"]
    for definition in imagination:
        assert "refinement_goal" not in definition["function"]["parameters"]["properties"]


def test_gripper_preview_enters_imagination_without_physical_effect() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision

    result = workspace.execute("close_gripper")

    assert result.ok and result.result == {"preview": "updated"}
    assert workspace.state.owner == "imagination"
    assert workspace.state.imagination.target.gripper == "closed"
    assert workspace.state.observation_revision == revision
    assert api.operation_log == []


def test_continuous_imagination_edits_one_target_then_hands_off() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.set_refinement_goal("把夹爪向下并调正")
    first = workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base")
    target1 = workspace.state.imagination.target.pose
    second = workspace.execute("rotate", axis="z", angle_deg=10, frame="tool")
    target2 = workspace.state.imagination.target.pose
    third = workspace.execute("close_gripper")

    assert first.result["preview"] == second.result["preview"] == "updated"
    assert target2.position_xyz == target1.position_xyz
    assert third.result["preview"] == "updated"
    assert workspace.state.imagination.target.gripper == "closed"
    assert workspace.state.imagination.refinement_goal == "把夹爪向下并调正"

    handoff = workspace.execute("finish_imagination", status="ready")
    action_id = handoff.result["action_id"]
    assert workspace.state.owner == "main"
    assert workspace.state.action_review.action_id == action_id
    assert workspace.state.last_handoff.status == "review_required"
    assert handoff.result == {"status": "review_required", "action_id": action_id}


def test_commit_is_only_physical_boundary_and_invalidates_revision_state() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.02], frame="base")
    workspace.execute("close_gripper")
    action_id = workspace.execute("finish_imagination", status="ready").result["action_id"]
    before = workspace.state.observation_revision

    result = workspace.execute("commit", action_id=action_id)

    assert result.ok
    assert api.operation_log == ["arm", "gripper:closed"]
    assert workspace.state.observation_revision == before + 1
    assert workspace.state.action_review is None
    assert region_id not in workspace.state.regions
    assert workspace.state.last_physical_action.intent == "task"
    assert workspace.state.last_physical_action.executed_stages == "arm+gripper"
    assert workspace.state.last_physical_action.outcome == "completed"
    assert workspace._private.previous_observation is not None
    assert workspace._private.last_physical_artifacts.focus_pose is not None


def test_commit_records_failed_stage_without_claiming_task_effect() -> None:
    arm_api = FakeContextApi()
    arm_api.move_error = True
    arm_workspace = ContextWorkspace(arm_api, "arm failure", motion_backend="pyroki")
    arm_workspace.set_refinement_goal("move and close")
    arm_workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base")
    arm_workspace.execute("close_gripper")
    arm_action = arm_workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]

    arm_result = arm_workspace.execute("commit", action_id=arm_action)

    assert not arm_result.ok
    assert arm_api.operation_log == ["arm"]
    assert arm_workspace.state.last_physical_action.executed_stages == "arm"
    assert arm_workspace.state.last_physical_action.outcome == "arm_failed"

    gripper_api = FakeContextApi()
    gripper_api.gripper_error = True
    gripper_workspace = ContextWorkspace(
        gripper_api,
        "gripper failure",
        motion_backend="pyroki",
    )
    gripper_workspace.set_refinement_goal("only close")
    gripper_workspace.execute("close_gripper")
    gripper_action = gripper_workspace.execute(
        "finish_imagination",
        status="ready",
    ).result["action_id"]

    gripper_result = gripper_workspace.execute("commit", action_id=gripper_action)

    assert not gripper_result.ok
    assert gripper_api.operation_log == ["gripper:closed"]
    assert gripper_workspace.state.last_physical_action.executed_stages == "gripper"
    assert gripper_workspace.state.last_physical_action.outcome == "gripper_failed"


def test_grasp_proposal_returns_general_action_seeds() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    result = workspace.execute("propose_grasps", region_id=region)
    seed_id = result.result["seed_ids"][0]
    assert seed_id.startswith("s")
    assert workspace.state.owner == "main"

    selected = workspace.execute("select", seed_id=seed_id)
    assert selected.ok and workspace.state.owner == "imagination"
    assert workspace.state.imagination.target.pose is not None


def test_imagination_limit_produces_main_review_not_approval() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.execute("open_gripper")
    result = workspace.limit_imagination()
    assert result.result["status"] == "review_required"
    assert workspace.state.last_handoff.status == "review_required"
    artifact = workspace._private.review_artifacts[result.result["action_id"]]
    assert artifact.termination_reason == "turn_limit"
    assert workspace.state.owner == "main"


def test_imagination_limit_does_not_publish_unexecutable_review() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base")

    result = workspace.limit_imagination()

    assert result.result == {"status": "failed"}
    assert workspace.state.owner == "main"
    assert workspace.state.action_review is None
    assert workspace.state.last_handoff.status == "failed"


def test_main_revision_continues_from_reviewed_target() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base")
    workspace.execute("close_gripper")
    first_action = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    reviewed = workspace.state.action_review.target

    workspace.set_refinement_goal("继续把已审查目标向 base +Y 微调")
    workspace.execute("delta_move", delta_xyz_m=[0.0, 0.01, 0.0], frame="base")

    revised = workspace.state.imagination.target
    assert workspace.state.action_review is None
    assert revised.gripper == "closed"
    assert np.allclose(
        revised.pose.position_xyz,
        np.asarray(reviewed.pose.position_xyz) + np.array([0.0, 0.01, 0.0]),
    )
    second_action = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    assert second_action != first_action
