from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vaw.context_runtime import (
    FUNCTION_NAMES,
    ActionPrediction,
    ActionTarget,
    ContextWorkspace,
    Pose,
    function_definitions,
)
from vaw.context_runtime.motion import (
    CUROBO_TCP_TO_HAND_LOCAL_XYZ,
    CuroboMotionBackend,
    MotionBackendError,
    MotionPlan,
)
from vaw.context_runtime.private import build_edit_summary
from vaw.context_runtime.protocol import (
    ACTION_REVIEW_SYSTEM_PROMPT,
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    REVIEW_FUNCTION_NAMES,
    STANDARD_MAIN_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    imagination_function_definitions,
    main_function_definitions,
    review_function_definitions,
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
        self.bbox_queries: list[str] = []
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
        self.bbox_queries.append(str(query))
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
        self.curobo_execute_kwargs: list[dict] = []
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
        values = np.asarray(trajectory).copy()
        self.curobo_execute_calls.append(values)
        self.curobo_execute_kwargs.append(dict(kwargs))
        self.joints = values[-1]
        quat_xyzw = np.roll(self._target_quaternion, -1)
        self.cartesian[:3] = self._target_position + Rotation.from_quat(
            quat_xyzw
        ).apply(np.asarray(CUROBO_TCP_TO_HAND_LOCAL_XYZ))
        self.cartesian[3:7] = self._target_quaternion


class SettlingCuroboApi(FakeCuroboContextApi):
    def __init__(self, *, settle: bool, initial_error: float = 0.03) -> None:
        super().__init__()
        self.settle = settle
        self.initial_error = float(initial_error)

    def execute_joint_trajectory(self, trajectory, **kwargs):
        values = np.asarray(trajectory).copy()
        self.curobo_execute_calls.append(values)
        self.curobo_execute_kwargs.append(dict(kwargs))
        if len(self.curobo_execute_calls) == 1:
            self.joints = values[-1] + self.initial_error
        elif self.settle:
            self.joints = values[-1]


class FakeSemanticGroundingApi(FakeContextApi):
    def __init__(self, *, choice: int | None = 2) -> None:
        super().__init__()
        self.choice = choice
        self.query_images: list[np.ndarray] = []
        self.query_prompts: list[str] = []

    def query_vlm(self, prompt, images=None, **kwargs):
        del kwargs
        self.query_prompts.append(str(prompt))
        self.query_images.append(np.asarray(images).copy())
        if len(self.query_prompts) % 2 == 1:
            return (
                '[{"box":[16,24,64,84],"evidence":"generic can"},'
                '{"box":[96,12,144,60],"evidence":"exact label"}]'
            )
        return json.dumps({"candidate": self.choice})


class SpatiallyInconsistentGraspApi(FakeContextApi):
    def segment_sam3_box_prompt(self, rgb, box):
        del box
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        h, w = mask.shape
        mask[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8] = True
        return [{"mask": mask, "score": 0.9}]

    def plan_grasp(self, depth, intrinsics, mask):
        del depth, intrinsics, mask
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
        pose[:3, 3] = [0.25, 0.0, 0.45]
        return [pose], [0.9]


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
    assert len(FUNCTION_NAMES) == 15
    assert FUNCTION_NAMES[0] == "detection_and_sam"
    assert FUNCTION_NAMES[1:3] == ("propose_grasps", "locate_point")
    assert "inspect" not in FUNCTION_NAMES
    assert IMAGINATION_FUNCTION_NAMES == (
        "delta_move",
        "rotate",
        "show_rotation_gizmo",
        "finish_imagination",
    )
    main = function_definitions()
    standard = main_function_definitions()
    review = review_function_definitions()
    imagination = imagination_function_definitions()
    assert [item["function"]["name"] for item in main] == list(FUNCTION_NAMES)
    assert [item["function"]["name"] for item in imagination] == list(
        IMAGINATION_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in standard] == list(
        STANDARD_MAIN_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in review] == list(
        REVIEW_FUNCTION_NAMES
    )
    assert "commit" not in STANDARD_MAIN_FUNCTION_NAMES
    assert "reject_action" not in STANDARD_MAIN_FUNCTION_NAMES
    assert "delta_move" not in STANDARD_MAIN_FUNCTION_NAMES
    assert "rotate" not in STANDARD_MAIN_FUNCTION_NAMES
    assert "open_gripper" in STANDARD_MAIN_FUNCTION_NAMES
    assert "close_gripper" in STANDARD_MAIN_FUNCTION_NAMES
    assert "start_imagination" in STANDARD_MAIN_FUNCTION_NAMES
    assert "detection_and_sam" not in REVIEW_FUNCTION_NAMES
    assert REVIEW_FUNCTION_NAMES[:2] == ("commit", "reject_action")
    assert "revise_action" not in FUNCTION_NAMES
    assert "revise_action" not in REVIEW_FUNCTION_NAMES
    assert not set(IMAGINATION_FUNCTION_NAMES).intersection(STANDARD_MAIN_FUNCTION_NAMES)
    open_definition = next(
        x["function"] for x in standard if x["function"]["name"] == "open_gripper"
    )
    close_definition = next(
        x["function"] for x in standard if x["function"]["name"] == "close_gripper"
    )
    assert "Main-only" in open_definition["description"]
    assert "不会打开真实夹爪" in open_definition["description"]
    assert "Main-only" in close_definition["description"]
    assert "不会闭合真实夹爪" in close_definition["description"]
    encoded = json.dumps(main + imagination).lower()
    assert "history" not in encoded and "receipt" not in encoded and "obb" not in encoded
    detection = main[0]["function"]
    assert "bbox detection" in detection["description"]
    assert "sam" in detection["description"].lower()
    assert "不刷新 observation" in detection["description"]
    assert "motion planning 失败" in detection["description"]
    assert "commit 后应重新观察真实画面" not in SYSTEM_PROMPT
    for name in ("delta_move", "rotate"):
        definition = next(
            x["function"] for x in imagination if x["function"]["name"] == name
        )
        assert "frame" in definition["parameters"]["required"]
        assert "refinement_goal" not in definition["parameters"]["required"]
    delta = next(
        x["function"]
        for x in imagination
        if x["function"]["name"] == "delta_move"
    )
    pose = next(x["function"] for x in main if x["function"]["name"] == "propose_pose")
    grasps = next(
        x["function"] for x in main if x["function"]["name"] == "propose_grasps"
    )
    point = next(x["function"] for x in main if x["function"]["name"] == "locate_point")
    assert "默认几何生成器" in grasps["description"]
    assert "最终抓取接触/闭合位姿" in grasps["description"]
    assert "不生成抓取方向" in point["description"]
    assert "应先使用 propose_grasps" in pose["description"]
    assert "当前想象目标" in delta["description"]
    assert "厘米级局部" in delta["description"]
    frame_description = delta["parameters"]["properties"]["frame"]["description"]
    assert "base" in frame_description and "+Z 恒为竖直上抬" in frame_description
    assert "tool +Z" in frame_description and "可能朝向支撑面" in frame_description
    rotate = next(
        x["function"] for x in imagination if x["function"]["name"] == "rotate"
    )
    assert "右手定则" in rotate["description"]
    assert "5–15°" in rotate["description"]
    assert "±90° 猜测" in rotate["description"]
    assert "WORLD +Z 始终朝上" in IMAGINATION_SYSTEM_PROMPT
    assert "点云反向旋转" in IMAGINATION_SYSTEM_PROMPT
    assert "超过 30°" in IMAGINATION_SYSTEM_PROMPT
    assert "APPROACH BASE" in SYSTEM_PROMPT
    assert "此前没有真实 closed gripper" in SYSTEM_PROMPT
    assert "total_translation_base_m" in IMAGINATION_SYSTEM_PROMPT
    assert "局部几何的唯一" in SYSTEM_PROMPT
    assert "不能再次按像素判断毫米级 gap" in SYSTEM_PROMPT
    assert "达到内部微调上限时只会交回 failed" in SYSTEM_PROMPT
    assert "若按当前这颗 seed 闭合" in SYSTEM_PROMPT
    assert "不要另写一种" in SYSTEM_PROMPT
    assert "current_tcp" in pose["description"]
    assert "绝不能虚构 `current_tcp`" in SYSTEM_PROMPT
    assert "GRIP 仍大于 0 可能是物体阻挡手指" in SYSTEM_PROMPT
    assert "requested_arm_delta_base_m" in SYSTEM_PROMPT
    assert "reject_action(action_id)" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "evidence_invalidated" in SYSTEM_PROMPT
    assert "会立即作废当前 action_id" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "局部、可执行的下一步" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "gripper-only open" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "尚未移动到容器" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "你不是第二个" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "不得重新要求毫米级 gap" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "revise_action" not in ACTION_REVIEW_SYSTEM_PROMPT
    assert "0≈闭合，1≈张开" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "中间值具有歧义" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "固定 source" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "不是 pre-grasp" in IMAGINATION_SYSTEM_PROMPT
    assert "TCP→SOURCE" not in IMAGINATION_SYSTEM_PROMPT
    assert "不要连续往下压来假装包夹" in IMAGINATION_SYSTEM_PROMPT
    assert "该几何已经由 Imagination 裁决" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "两指闭合扫掠" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "张开的手指预先贴住物体" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "两边都能碰到" in IMAGINATION_SYSTEM_PROMPT
    assert "深度和高度够不够" in IMAGINATION_SYSTEM_PROMPT
    assert "指尖擦边" in IMAGINATION_SYSTEM_PROMPT
    assert "不证明真的抓住" in IMAGINATION_SYSTEM_PROMPT
    select_description = next(
        item["function"]["description"]
        for item in main
        if item["function"]["name"] == "select"
    )
    assert "两指扫掠" in select_description
    assert "不要另写抓取方向" in select_description
    assert "不要要求手指先贴住物体" in select_description
    assert "继续随机" in IMAGINATION_SYSTEM_PROMPT
    assert "非物理 Function 不刷新真实 observation" in SYSTEM_PROMPT
    assert "不得仅因下层切换" in SYSTEM_PROMPT
    assert "within_region_id" in SYSTEM_PROMPT
    assert "空间 ActionReview 与 gripper-only" in SYSTEM_PROMPT
    assert "你不能编辑夹爪状态" in IMAGINATION_SYSTEM_PROMPT
    assert "open_gripper" not in [
        item["function"]["name"] for item in imagination
    ]
    for name in ("select", "propose_pose", "start_imagination"):
        definition = next(x["function"] for x in main if x["function"]["name"] == name)
        assert "refinement_goal" in definition["parameters"]["required"]
    for definition in imagination:
        assert "refinement_goal" not in definition["function"]["parameters"]["properties"]


def test_action_target_is_atomic_arm_or_gripper_command() -> None:
    pose = Pose((0.4, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="exactly one"):
        ActionTarget()
    with pytest.raises(ValueError, match="exactly one"):
        ActionTarget(pose=pose, gripper="open")


def test_curobo_retries_exact_final_waypoint_when_reduced_api_is_still_settling() -> None:
    api = SettlingCuroboApi(settle=True)
    backend = CuroboMotionBackend(api)
    target = Pose((0.4, 0.0, 0.2), (0.0, 1.0, 0.0, 0.0))
    trajectory = np.stack([np.zeros(7), np.full(7, 0.5)])
    plan = MotionPlan(
        "curobo",
        ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(trajectory[-1]),
            trajectory_checked=True,
            collision_checked=True,
        ),
        trajectory,
    )

    backend.execute(plan, target)

    assert len(api.curobo_execute_calls) == 2
    np.testing.assert_allclose(api.curobo_execute_calls[1], trajectory[-1:])
    assert api.curobo_execute_kwargs[0] == {
        "subsample": 2,
        "tolerance": 0.025,
        "max_steps": 15,
    }
    assert api.curobo_execute_kwargs[1] == {
        "subsample": 1,
        "tolerance": 0.025,
        "max_steps": 120,
    }


def test_curobo_final_settle_does_not_relax_max_joint_threshold() -> None:
    api = SettlingCuroboApi(settle=False)
    backend = CuroboMotionBackend(api)
    target = Pose((0.4, 0.0, 0.2), (0.0, 1.0, 0.0, 0.0))
    trajectory = np.stack([np.zeros(7), np.full(7, 0.5)])
    plan = MotionPlan(
        "curobo",
        ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(trajectory[-1]),
            trajectory_checked=True,
            collision_checked=True,
        ),
        trajectory,
    )

    with pytest.raises(MotionBackendError, match="max joint error"):
        backend.execute(plan, target)

    assert len(api.curobo_execute_calls) == 2


def test_curobo_accepts_dimension_independent_small_joint_residuals() -> None:
    api = SettlingCuroboApi(settle=False, initial_error=0.012)
    backend = CuroboMotionBackend(api)
    target = Pose((0.4, 0.0, 0.2), (0.0, 1.0, 0.0, 0.0))
    trajectory = np.stack([np.zeros(7), np.full(7, 0.5)])
    plan = MotionPlan(
        "curobo",
        ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(trajectory[-1]),
            trajectory_checked=True,
            collision_checked=True,
        ),
        trajectory,
    )

    backend.execute(plan, target)

    # The seven-dimensional L2 residual is > 0.02 rad even though no single
    # joint has a material endpoint miss.
    assert np.linalg.norm(api.joints - trajectory[-1]) > 0.02
    assert len(api.curobo_execute_calls) == 1


def test_detection_is_idempotent_within_one_observation_revision() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    first = workspace.execute("detection_and_sam", query=" Alphabet   Soup Can ")
    second = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert second.result == first.result
    assert list(workspace.state.regions) == [first.result["region_id"]]
    assert len(api.bbox_queries) == 1
    assert second.trace_diagnostics == {
        "semantic_grounding": {
            "mode": "revision_local_cache",
            "region_id": first.result["region_id"],
        }
    }


def test_main_gripper_preview_creates_review_without_physical_effect() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision

    result = workspace.execute("close_gripper")

    assert result.ok and result.result == {"action_id": "a1"}
    assert workspace.state.owner == "main"
    assert workspace.state.imagination is None
    assert workspace.state.action_review.target.pose is None
    assert workspace.state.action_review.target.gripper == "closed"
    assert workspace.state.observation_revision == revision
    assert api.operation_log == []


def test_detection_privately_requests_exact_semantic_disambiguation() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert result.ok
    assert len(api.bbox_queries) == 1
    assert "exact semantic target described as alphabet soup can" in api.bbox_queries[0]
    assert "generic visual match" in api.bbox_queries[0]
    region = workspace.state.regions[result.result["region_id"]]
    assert region.query == "alphabet soup can"


def test_detection_reviews_enlarged_candidates_before_registering_region() -> None:
    api = FakeSemanticGroundingApi(choice=2)
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert result.ok
    assert result.result["bbox_xyxy_px"] == pytest.approx([96.0, 12.0, 144.0, 60.0])
    assert api.bbox_queries == []
    assert [image.shape for image in api.query_images] == [
        (120, 160, 3),
        (720, 1200, 3),
    ]
    assert "up to three distinct plausible candidates" in api.query_prompts[0]
    assert "enlarged candidate crops" in api.query_prompts[1]
    diagnostics = result.trace_diagnostics["semantic_grounding"]
    assert diagnostics["mode"] == "candidate_review"
    assert diagnostics["selected_candidate"] == 2


def test_detection_uses_private_hires_semantic_view_and_maps_box_back() -> None:
    api = FakeSemanticGroundingApi(choice=2)
    captures = 0

    def semantic_rgb_provider() -> np.ndarray:
        nonlocal captures
        captures += 1
        return np.zeros((240, 320, 3), dtype=np.uint8)

    workspace = ContextWorkspace(
        api,
        "task",
        motion_backend="pyroki",
        semantic_rgb_provider=semantic_rgb_provider,
    )

    first = workspace.execute("detection_and_sam", query="alphabet soup can")
    second = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert first.ok and second.ok
    assert first.result["bbox_xyxy_px"] == pytest.approx([48.0, 6.0, 72.0, 30.0])
    assert captures == 1
    assert api.query_images[0].shape == (240, 320, 3)
    diagnostics = first.trace_diagnostics["semantic_grounding"]
    assert diagnostics["semantic_rgb_shape"] == [240, 320, 3]
    assert diagnostics["observation_rgb_shape"] == [120, 160, 3]
    assert diagnostics["box_observation_px"] == pytest.approx([48.0, 6.0, 72.0, 30.0])

    workspace.refresh_observation()
    workspace.execute("detection_and_sam", query="alphabet soup can")
    assert captures == 2


def test_detection_rejects_ambiguous_semantic_review_without_region() -> None:
    api = FakeSemanticGroundingApi(choice=None)
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="ambiguous can")

    assert not result.ok
    assert "ambiguous" in result.result["error"]
    assert workspace.state.regions == {}


def test_continuous_imagination_edits_one_target_then_hands_off() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.set_refinement_goal("把夹爪向下并调正")
    first = workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base")
    target1 = workspace.state.imagination.target.pose
    second = workspace.execute("rotate", axis="z", angle_deg=10, frame="tool")
    target2 = workspace.state.imagination.target.pose

    assert first.result["preview"] == second.result["preview"] == "updated"
    assert target2.position_xyz == target1.position_xyz
    assert workspace.state.imagination.target.gripper is None
    assert workspace.state.imagination.refinement_goal == "把夹爪向下并调正"

    handoff = workspace.execute("finish_imagination", status="ready")
    action_id = handoff.result["action_id"]
    assert workspace.state.owner == "main"
    assert workspace.state.action_review.action_id == action_id
    assert workspace.state.last_handoff.status == "review_required"
    assert handoff.result == {"status": "review_required", "action_id": action_id}


def test_gripper_review_cannot_be_spatially_edited_by_imagination() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    result = workspace.execute("open_gripper")
    action_id = result.result["action_id"]

    edited = workspace.execute(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, -0.01],
        frame="base",
    )

    assert not edited.ok
    assert workspace.state.imagination is None
    assert workspace.state.action_review.action_id == action_id


def test_gripper_only_review_has_no_private_arm_baseline() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = workspace.execute("close_gripper").result["action_id"]

    assert workspace.state.action_review.action_id == action_id
    assert workspace.state.action_review.target.pose is None
    assert workspace._private.imagination_artifacts is None
    artifacts = workspace._private.review_artifacts[action_id]
    assert artifacts.motion_plan is None
    assert artifacts.edit_summary.initial_target.pose is None
    assert artifacts.edit_summary.current_target.gripper == "closed"


def test_commit_is_only_physical_boundary_and_invalidates_revision_state() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.02], frame="base")
    action_id = workspace.execute("finish_imagination", status="ready").result["action_id"]
    before = workspace.state.observation_revision

    result = workspace.execute("commit", action_id=action_id)

    assert result.ok
    assert api.operation_log == ["arm"]
    assert workspace.state.observation_revision == before + 1
    assert workspace.state.action_review is None
    assert region_id not in workspace.state.regions
    assert workspace.state.last_physical_action.intent == "task"
    assert workspace.state.last_physical_action.executed_stages == "arm"
    assert workspace.state.last_physical_action.outcome == "completed"
    assert workspace.state.last_physical_action.target_gripper is None
    assert workspace.state.last_physical_action.requested_arm_delta_base_m == pytest.approx(
        (0.0, 0.0, 0.02)
    )
    assert workspace._private.previous_observation is not None
    assert workspace._private.last_physical_artifacts.focus_pose is not None


def test_reject_action_explicitly_discards_review_without_physics() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    action_id = workspace.execute("open_gripper").result["action_id"]
    revision = workspace.state.observation_revision

    result = workspace.execute("reject_action", action_id=action_id)

    assert result.ok
    assert result.result == {
        "declined_action_id": action_id,
        "declined_how": "explicit",
        "declined_intent": "set gripper open",
    }
    assert workspace.state.action_review is None
    assert workspace._private.review_artifacts == {}
    assert workspace.state.observation_revision == revision
    assert api.operation_log == []


def test_commit_records_failed_stage_without_claiming_task_effect() -> None:
    arm_api = FakeContextApi()
    arm_api.move_error = True
    arm_workspace = ContextWorkspace(arm_api, "arm failure", motion_backend="pyroki")
    arm_workspace.set_refinement_goal("move and close")
    arm_workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base")
    arm_action = arm_workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]

    arm_result = arm_workspace.execute("commit", action_id=arm_action)

    assert not arm_result.ok
    assert arm_api.operation_log == ["arm"]
    assert arm_workspace.state.last_physical_action.executed_stages == "arm"
    assert arm_workspace.state.last_physical_action.outcome == "arm_failed"
    assert arm_workspace.state.last_physical_action.failed_action_id == arm_action
    assert arm_workspace.state.last_physical_action.evidence_invalidated is True
    assert arm_workspace.state.last_physical_action.recovery_hint
    assert arm_workspace.state.last_physical_action.error_detail
    assert arm_workspace.state.action_review is None

    gripper_api = FakeContextApi()
    gripper_api.gripper_error = True
    gripper_workspace = ContextWorkspace(
        gripper_api,
        "gripper failure",
        motion_backend="pyroki",
    )
    gripper_action = gripper_workspace.execute("close_gripper").result["action_id"]

    gripper_result = gripper_workspace.execute("commit", action_id=gripper_action)

    assert not gripper_result.ok
    assert gripper_api.operation_log == ["gripper:closed"]
    assert gripper_workspace.state.last_physical_action.executed_stages == "gripper"
    assert gripper_workspace.state.last_physical_action.outcome == "gripper_failed"
    assert gripper_workspace.state.last_physical_action.failed_action_id == gripper_action
    assert gripper_workspace.state.last_physical_action.evidence_invalidated is True
    assert gripper_workspace.state.last_physical_action.recovery_hint


def test_failed_grasp_commit_exposes_source_query_for_redetection() -> None:
    api = FakeContextApi()
    api.move_error = True
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    workspace.set_refinement_goal("grasp the can")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    workspace.execute("select", seed_id=seed_id)
    action_id = workspace.execute("finish_imagination", status="ready").result["action_id"]

    result = workspace.execute("commit", action_id=action_id)

    assert not result.ok
    last = workspace.state.last_physical_action
    assert last.outcome == "arm_failed"
    assert last.failed_action_id == action_id
    assert last.source_query == "can"
    assert last.evidence_invalidated is True
    assert last.recovery_hint
    assert "detection_and_sam" in last.recovery_hint
    assert region not in workspace.state.regions
    assert workspace.state.action_review is None


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


def test_grasp_proposal_rejects_seed_outside_its_source_geometry() -> None:
    workspace = ContextWorkspace(
        SpatiallyInconsistentGraspApi(),
        "task",
        motion_backend="pyroki",
    )
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]

    result = workspace.execute("propose_grasps", region_id=region)

    assert not result.ok
    assert workspace.state.seeds == {}
    rejected = result.trace_diagnostics["grasp_candidates"]["attempts"][0][
        "rejected"
    ]
    assert rejected[0]["reasons"] == ["outside_source_geometry"]
    assert rejected[0]["source_center_distance_m"] > rejected[0]["source_radius_m"]


def test_imagination_limit_fails_without_publishing_an_action() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.execute("start_imagination")
    result = workspace.limit_imagination()

    assert result.result == {"status": "failed"}
    assert workspace.state.last_handoff.status == "failed"
    assert workspace.state.action_review is None
    assert workspace._private.review_artifacts == {}
    assert (
        result.trace_diagnostics["imagination_handoff"]["termination_reason"]
        == "turn_limit"
    )
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


def test_review_cannot_return_the_same_target_to_imagination() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base")
    first_action = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]

    resumed = workspace.execute("revise_action", action_id=first_action)

    assert not resumed.ok
    assert "unknown function 'revise_action'" in resumed.result["error"]
    assert workspace.state.owner == "main"
    assert workspace.state.imagination is None
    assert workspace.state.action_review.action_id == first_action


def test_explicit_current_handoff_keeps_gripper_commands_out_of_imagination() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision
    workspace.set_refinement_goal("先仅张开真实夹爪")

    started = workspace.execute("start_imagination")
    assert started.ok and workspace.state.owner == "imagination"
    assert workspace.state.imagination.target.pose == workspace.state.robot.tcp_pose

    gizmo = workspace.execute("show_rotation_gizmo", frame="tool", axis="y")
    assert gizmo.result == {
        "preview": "updated",
        "rotation_guide": {"frame": "tool", "axis": "y"},
    }
    assert workspace._private.imagination_artifacts.rotation_gizmo_frame == "tool"
    assert workspace._private.imagination_artifacts.rotation_gizmo_axis == "y"

    opened = workspace.execute("open_gripper")
    assert not opened.ok
    assert "Main-only" in opened.result["error"]
    assert workspace.state.imagination.target.pose == workspace.state.robot.tcp_pose
    assert workspace.state.imagination.target.gripper is None
    assert workspace.state.observation_revision == revision
