from __future__ import annotations

import json
from dataclasses import replace

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
from vaw.context_runtime.packet import ContextCompiler
from vaw.context_runtime.motion import (
    CUROBO_TCP_TO_HAND_LOCAL_XYZ,
    CuroboMotionBackend,
    MotionBackendError,
    MotionPlan,
    PyrokiMotionBackend,
)
from vaw.context_runtime.attached_object import fit_gravity_stable_proxy
from vaw.context_runtime.errors import ContextFunctionError
from vaw.context_runtime.functions import _geometric_grasps, _top_width_advisory
from vaw.context_runtime.private import build_edit_summary
from vaw.context_runtime.protocol import (
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    MAIN_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    imagination_function_definitions,
    main_function_definitions,
)


class FakeContextApi:
    camera_name = "agentview"
    wrist_camera_name = "robot0_eye_in_hand"
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)
    seed_ik_matches_target = True

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


def _selected_action(workspace: ContextWorkspace, query: str = "can") -> str:
    region = workspace.execute("detection_and_sam", query=query).result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    return workspace.execute("select", seed_id=seed).result["action_id"]


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


class ScriptedSemanticGroundingApi(FakeContextApi):
    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self.replies = list(replies)
        self.query_images: list[np.ndarray] = []

    def query_vlm(self, prompt, images=None, **kwargs):
        del prompt, kwargs
        self.query_images.append(np.asarray(images).copy())
        return self.replies.pop(0)


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


class CountingFailedGraspPlanBackend:
    """IK preview succeeds; grasp trajectory planning always fails."""

    name = "count-fail-grasp"
    tcp_to_hand_local_xyz = (0.0, 0.0, -0.1)

    def __init__(self) -> None:
        self.grasp_calls = 0

    def preview(self, target):
        del target
        return ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(np.zeros(7)),
        )

    def plan_pose(self, target, *, preview=None):
        del target, preview
        return MotionPlan(
            self.name,
            ActionPrediction(solve_ik="error", detail="no executable plan"),
        )

    def plan_grasp(self, target, *, object_name, object_mask, preview=None):
        del target, object_name, object_mask, preview
        self.grasp_calls += 1
        return MotionPlan(
            self.name,
            ActionPrediction(
                solve_ik="error",
                detail="CuRobo found no collision-free trajectory",
            ),
        )

    def execute(self, plan, target):
        raise AssertionError(f"failed plan must not execute: {plan}, {target}")


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


class PlanBudgetMotionBackend:
    """Plan successfully a fixed number of times, then fail every request."""

    name = "budget-test"
    tcp_to_hand_local_xyz = (0.0, 0.0, -0.1)

    def __init__(self, successes: int) -> None:
        self.successes = int(successes)
        self.calls = 0

    def _plan(self) -> MotionPlan:
        self.calls += 1
        if self.calls > self.successes:
            return MotionPlan(
                self.name,
                ActionPrediction(solve_ik="error", detail="target is unreachable"),
            )
        return MotionPlan(
            self.name,
            ActionPrediction(
                solve_ik="returned",
                joint_positions_rad=tuple(np.zeros(7)),
            ),
            np.zeros((1, 7)),
        )

    def preview(self, target):
        del target
        return ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(np.zeros(7)),
        )

    def plan_pose(self, target, *, preview=None):
        del target, preview
        return self._plan()

    def plan_grasp(self, target, *, object_name, object_mask, preview=None):
        del target, object_name, object_mask, preview
        return self._plan()

    def execute(self, plan, target):
        del target
        assert plan.prediction.solve_ik == "returned"


class RecordingMotionBackend:
    tcp_to_hand_local_xyz = (0.0, 0.0, -0.1168)

    def __init__(self, name: str) -> None:
        self.name = name
        self.pose_plans: list[Pose] = []
        self.grasp_plans: list[Pose] = []
        self.executions: list[tuple[MotionPlan, Pose]] = []

    def preview(self, target):
        return ActionPrediction(
            solve_ik="returned",
            joint_positions_rad=tuple(np.zeros(7)),
        )

    def _plan(self, target):
        return MotionPlan(
            self.name,
            ActionPrediction(
                solve_ik="returned",
                joint_positions_rad=tuple(np.zeros(7)),
                trajectory_checked=self.name == "coarse",
                collision_checked=self.name == "coarse",
            ),
            np.zeros((1, 7)),
        )

    def plan_pose(self, target, *, preview=None):
        del preview
        self.pose_plans.append(target)
        return self._plan(target)

    def plan_grasp(self, target, *, object_name, object_mask, preview=None):
        del object_name, object_mask, preview
        self.grasp_plans.append(target)
        return self._plan(target)

    def execute(self, plan, target):
        self.executions.append((plan, target))


def test_main_and_imagination_function_contracts_are_disjoint_and_small() -> None:
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
    imagination = imagination_function_definitions()
    assert [item["function"]["name"] for item in main] == list(FUNCTION_NAMES)
    assert [item["function"]["name"] for item in imagination] == list(
        IMAGINATION_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in standard] == list(
        MAIN_FUNCTION_NAMES
    )
    assert "commit" in MAIN_FUNCTION_NAMES
    assert "reject_action" in MAIN_FUNCTION_NAMES
    assert "call_imagination" in MAIN_FUNCTION_NAMES
    assert "delta_move" in MAIN_FUNCTION_NAMES
    assert "rotate" not in MAIN_FUNCTION_NAMES
    assert "start_imagination" not in FUNCTION_NAMES
    assert set(IMAGINATION_FUNCTION_NAMES).intersection(MAIN_FUNCTION_NAMES) == {
        "delta_move"
    }
    open_definition = next(
        x["function"] for x in standard if x["function"]["name"] == "open_gripper"
    )
    close_definition = next(
        x["function"] for x in standard if x["function"]["name"] == "close_gripper"
    )
    assert "立即打开真实夹爪" in open_definition["description"]
    assert "立即闭合真实夹爪" in close_definition["description"]
    encoded = json.dumps(main + imagination).lower()
    assert "history" not in encoded and "receipt" not in encoded and "obb" not in encoded
    detection = main[0]["function"]
    assert "bbox detection" in detection["description"]
    assert "sam" in detection["description"].lower()
    assert "不移动机器人" in detection["description"]
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
    assert "TOP、PCA 和 GraspNet" in grasps["description"]
    assert "不生成方向" in point["description"]
    assert "planned 空间 Action" in pose["description"]
    assert "厘米级平移" in delta["description"]
    frame_description = delta["parameters"]["properties"]["frame"]["description"]
    assert "base" in frame_description and "+Z 恒为世界上方" in frame_description
    rotate = next(
        x["function"] for x in imagination if x["function"]["name"] == "rotate"
    )
    assert "右手定则" in rotate["description"]
    assert "平移不能修复" in rotate["description"]
    assert "绝对值不超过 10°" in rotate["description"]
    assert rotate["parameters"]["properties"]["angle_deg"]["minimum"] == -10.0
    assert rotate["parameters"]["properties"]["angle_deg"]["maximum"] == 10.0
    gizmo = next(
        x["function"] for x in imagination if x["function"]["name"] == "show_rotation_gizmo"
    )
    assert "不修改目标" in gizmo["description"]
    assert "正负号" in gizmo["description"]
    assert "base +Z 恒为上抬" in IMAGINATION_SYSTEM_PROMPT
    assert "CONTACT FRONT 和 SIDE" in IMAGINATION_SYSTEM_PROMPT
    assert "gripper-only fallback" in IMAGINATION_SYSTEM_PROMPT
    assert "琥珀色 carried-volume 是可选" in SYSTEM_PROMPT
    assert "Main ReAct Agent" in SYSTEM_PROMPT
    assert "call_imagination" in SYSTEM_PROMPT
    assert "Task Memory" in SYSTEM_PROMPT
    assert "executed 只表示命令完成" in SYSTEM_PROMPT
    assert "Current Function Event" in SYSTEM_PROMPT
    assert "action_proposal.executable 是能否 commit 的权威标志" in SYSTEM_PROMPT
    assert "只有回滚 Action 的 executable=true 才可直接 commit" in SYSTEM_PROMPT
    assert "不要重新 detection/propose 生成等价集合" in SYSTEM_PROMPT
    assert "薄/扁物体" in SYSTEM_PROMPT
    assert "只有目标离开局部视野、身份不确定或必须重新生成抓取方向时" in SYSTEM_PROMPT
    assert "Action 执行后蓝轮廓" in SYSTEM_PROMPT
    assert "不要仅因蓝轮廓已经消失而重新 detection" in SYSTEM_PROMPT
    assert "优先保留该 metric anchor" in SYSTEM_PROMPT
    # A1 (v42): contradictory lateral evidence forbids descending/releasing,
    # not correction itself. Read the current geometry; do not switch strategy
    # by counting similar commands.
    assert "被禁止的是继续下降和释放，而不是修正本身" in SYSTEM_PROMPT
    assert "而不是重复同一感知循环或按次数切换策略" in SYSTEM_PROMPT
    assert "物理接触只看当前 Canvas，不看已经发出过多少次同类命令" in SYSTEM_PROMPT
    assert "region/point 仍 verified 只表示画面里还能认出同一物体" in SYSTEM_PROMPT
    assert "容器已不再是可用开口" in SYSTEM_PROMPT
    assert "descending delta_move #" not in SYSTEM_PROMPT
    # v46: two level panels cannot separate "on the rim" from "in the opening",
    # so carrying tilts SIDE and the prompt has to say how to read the pair.
    assert 'CONTACT SIDE 会抬高为斜俯视，标题标出 "OBLIQUE <角度>° DOWN"' in SYSTEM_PROMPT
    assert "横向对齐以带 OBLIQUE 标记的那一幅为准" in SYSTEM_PROMPT
    assert "竖直方向同时混合了高度与进深" in SYSTEM_PROMPT
    assert "VIRTUAL TOP" not in SYSTEM_PROMPT
    # A3 (v42): positive routing triggers for call_imagination.
    assert "以下情形默认\n先委派" in SYSTEM_PROMPT
    assert "这些是路由建议而非门控" in SYSTEM_PROMPT
    assert "二维重叠" in SYSTEM_PROMPT
    assert "物体下部已经越过开口/边沿平面" in SYSTEM_PROMPT
    assert "而非停在边沿上" in SYSTEM_PROMPT
    assert "region/point 会自动对照新画面复验" in SYSTEM_PROMPT
    assert "status=occluded" in SYSTEM_PROMPT
    assert "world change check" in SYSTEM_PROMPT
    assert "seed 与 ActionProposal 仍是 revision-local" in SYSTEM_PROMPT
    assert "指尖低于物体顶面可以是正常抓取状态" in SYSTEM_PROMPT
    assert "掌部底面标为一条深青色窄带" in SYSTEM_PROMPT
    assert "窄带一旦贴到当前正下方的顶面或沿口" in SYSTEM_PROMPT
    assert "Preview 对位不能恢复那个真实几何" in IMAGINATION_SYSTEM_PROMPT
    assert "不得把" in SYSTEM_PROMPT and "指尖—顶面距离" in SYSTEM_PROMPT
    assert "指尖低于物体顶面并不代表碰撞" in IMAGINATION_SYSTEM_PROMPT
    assert "真正需要净空" in IMAGINATION_SYSTEM_PROMPT
    assert "掌部/横梁/指根" in IMAGINATION_SYSTEM_PROMPT
    assert "必须分别判断位置与方向" in IMAGINATION_SYSTEM_PROMPT
    assert "继续平移不能修复方向错误" in IMAGINATION_SYSTEM_PROMPT
    assert "PCA 可以是非 top-down" in IMAGINATION_SYSTEM_PROMPT
    assert "较大的 GRIP 可能表示物体阻挡" in SYSTEM_PROMPT
    assert "不得让已闭合的手指朝支撑面" in SYSTEM_PROMPT
    assert "open_gripper" not in [
        item["function"]["name"] for item in imagination
    ]
    refine = next(x["function"] for x in standard if x["function"]["name"] == "call_imagination")
    assert refine["parameters"]["required"] == ["action_id", "instruction"]
    assert "不是 commit 的前置条件" in refine["description"]
    for reason in ("geometry_unresolved", "plan_unavailable", "turn_limit", "subagent_error"):
        assert reason in SYSTEM_PROMPT
    commit_definition = next(
        x["function"] for x in standard if x["function"]["name"] == "commit"
    )
    assert "planned 与 refined 均可" in commit_definition["description"]
    assert "不是 commit 的前置" in SYSTEM_PROMPT
    for definition in imagination:
        assert "refinement_goal" not in definition["function"]["parameters"]["properties"]


def test_action_target_is_a_spatial_pose() -> None:
    pose = Pose((0.4, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))

    with pytest.raises(TypeError):
        ActionTarget()
    assert ActionTarget(pose=pose).pose == pose


def test_pyroki_local_backend_preserves_the_public_curobo_tcp_frame() -> None:
    api = FakeContextApi()
    backend = PyrokiMotionBackend(
        api,
        public_tcp_to_hand_local_xyz=CUROBO_TCP_TO_HAND_LOCAL_XYZ,
    )
    target = Pose((0.4, -0.1, 0.2), (1.0, 0.0, 0.0, 0.0))

    plan = backend.plan_pose(target)

    assert plan.prediction.solve_ik == "returned"
    requested_position = api.solve_calls[-1][0]
    rotation = Rotation.from_quat(target.quaternion_xyzw)
    expected = np.asarray(target.position_xyz) + rotation.apply(
        np.asarray(CUROBO_TCP_TO_HAND_LOCAL_XYZ) - api._TCP_OFFSET
    )
    np.testing.assert_allclose(requested_position, expected)


def test_pyroki_reseeds_every_solve_from_current_observed_arm_joints() -> None:
    class StatefulPyrokiApi(FakeContextApi):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = np.full(8, 9.0, dtype=np.float64)
            self.ik_seeds: list[np.ndarray] = []

        def solve_ik(self, position, quaternion_wxyz, *, return_info=False):
            self.ik_seeds.append(np.asarray(self.cfg, dtype=np.float64).copy())
            result = super().solve_ik(
                position,
                quaternion_wxyz,
                return_info=return_info,
            )
            # Emulate the stateful reduced API leaving its solved configuration
            # cached for the next call.
            self.cfg = np.full(8, -7.0, dtype=np.float64)
            return result

    api = StatefulPyrokiApi()
    backend = PyrokiMotionBackend(api)
    target = Pose((0.4, -0.1, 0.2), (1.0, 0.0, 0.0, 0.0))

    backend.preview(target)
    first_observed = api.joints.copy()
    api.joints = api.joints + 0.25
    backend.preview(target)

    assert len(api.ik_seeds) == 4
    np.testing.assert_allclose(api.ik_seeds[0][:7], first_observed)
    np.testing.assert_allclose(api.ik_seeds[2][:7], api.joints)
    assert api.ik_seeds[0][7] == 9.0
    assert api.ik_seeds[1][7] == -7.0
    assert api.ik_seeds[2][7] == -7.0
    assert api.ik_seeds[3][7] == -7.0


def test_far_imagination_target_stays_on_coarse_planner() -> None:
    coarse = RecordingMotionBackend("coarse")
    local = RecordingMotionBackend("local")
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=coarse,
        local_motion_backend=local,
    )
    action_id = _selected_action(workspace)
    assert coarse.grasp_plans
    assert not local.grasp_plans

    workspace.begin_imagination("lower by one centimetre", action_id)
    edited = workspace.execute_imagination(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, -0.01],
        frame="base",
    )
    assert edited.ok
    assert len(coarse.grasp_plans) >= 2
    assert not local.grasp_plans
    assert workspace._private.action_artifacts.motion_plan.backend == "coarse"
    assert edited.trace_diagnostics["motion_route"]["reason"] == "coarse_target"
    assert workspace.execute_imagination(
        "finish_imagination",
        status="ready",
    ).ok

    committed = workspace.execute("commit", action_id=action_id)

    assert committed.ok
    assert len(coarse.executions) == 1
    assert not local.executions


def test_near_imagination_target_uses_local_planner_and_executor() -> None:
    coarse = RecordingMotionBackend("coarse")
    local = RecordingMotionBackend("local")
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=coarse,
        local_motion_backend=local,
    )
    action_id = _selected_action(workspace)
    action = workspace.state.action_proposal
    assert action is not None
    robot = workspace.state.robot
    assert robot is not None
    workspace.state.robot = replace(robot, tcp_pose=action.target.pose)

    workspace.begin_imagination("lower by one centimetre", action_id)
    edited = workspace.execute_imagination(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, -0.01],
        frame="base",
    )

    assert edited.ok
    assert local.grasp_plans
    assert workspace._private.action_artifacts.motion_plan.backend == "local"
    assert edited.trace_diagnostics["motion_route"]["reason"] == "local_target"
    assert workspace.execute_imagination(
        "finish_imagination",
        status="ready",
    ).ok
    committed = workspace.execute("commit", action_id=action_id)
    assert committed.ok
    assert not coarse.executions
    assert len(local.executions) == 1


def test_large_orientation_change_stays_on_coarse_planner() -> None:
    coarse = RecordingMotionBackend("coarse")
    local = RecordingMotionBackend("local")
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=coarse,
        local_motion_backend=local,
    )
    current = workspace.state.robot.tcp_pose
    assert current is not None
    target_rotation = Rotation.from_quat(current.quaternion_xyzw) * Rotation.from_euler(
        "z", 30.0, degrees=True
    )

    backend, route = workspace.motion_backend_for_target(
        Pose(current.position_xyz, tuple(target_rotation.as_quat()))
    )

    assert backend is coarse
    assert route["reason"] == "coarse_target"
    assert route["translation_m"] == 0.0
    assert route["rotation_deg"] == pytest.approx(30.0)


def test_main_direct_delta_uses_local_planner_without_curobo_fallback() -> None:
    coarse = RecordingMotionBackend("coarse")
    local = RecordingMotionBackend("local")
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=coarse,
        local_motion_backend=local,
    )

    result = workspace.execute(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, 0.01],
        frame="base",
    )

    assert result.ok
    assert not coarse.pose_plans and not coarse.executions
    assert len(local.pose_plans) == 1
    assert len(local.executions) == 1


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
        "tolerance": 0.02,
        "max_steps": 240,
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


def test_main_gripper_controls_are_direct_physical_actions() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision

    result = workspace.execute("close_gripper")

    assert result.ok and "gripper_opening" in result.result
    assert workspace.state.imagination is None
    assert workspace.state.action_proposal is None
    assert workspace.state.observation_revision == revision + 1
    assert api.operation_log == ["gripper:closed"]
    assert workspace.state.last_physical_action.target_gripper == "closed"


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


def test_detection_retries_empty_candidate_and_review_replies_locally() -> None:
    api = ScriptedSemanticGroundingApi(
        [
            "",
            '[{"box":[16,24,64,84],"evidence":"exact label"}]',
            "not json",
            '{"candidate":1}',
        ]
    )
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert result.ok
    assert len(api.query_images) == 4
    diagnostics = result.trace_diagnostics["semantic_grounding"]
    assert diagnostics["candidate_attempts"][0]["reply_length"] == 0
    assert diagnostics["candidate_attempts"][1]["reply_length"] > 0
    assert diagnostics["candidate_attempts"][0]["parse_error"] == (
        "response contains no valid JSON value"
    )
    assert [item["reply"] for item in diagnostics["review_attempts"]] == [
        "not json",
        '{"candidate":1}',
    ]


def test_detection_keeps_raw_replies_when_candidate_parsing_exhausts_retries() -> None:
    api = ScriptedSemanticGroundingApi(["", "still not json", "also not json"])
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="alphabet soup can")

    assert not result.ok
    assert "candidate parsing failed" in result.result["error"]
    diagnostics = result.trace_diagnostics["semantic_grounding"]
    assert [item["reply"] for item in diagnostics["candidate_attempts"]] == [
        "",
        "still not json",
        "also not json",
    ]
    assert diagnostics["candidate_reply"] == "also not json"


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
    # The carried region survives revalidation (the fake scene is static), so
    # an identical query reuses it without touching the semantic camera.
    workspace.execute("detection_and_sam", query="alphabet soup can")
    assert captures == 1
    # A genuinely new grounding request in the new revision re-captures the
    # private hi-res semantic view.
    workspace.execute("detection_and_sam", query="tomato sauce can")
    assert captures == 2


def test_detection_rejects_ambiguous_semantic_review_without_region() -> None:
    api = FakeSemanticGroundingApi(choice=None)
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = workspace.execute("detection_and_sam", query="ambiguous can")

    assert not result.ok
    assert "ambiguous" in result.result["error"]
    assert workspace.state.regions == {}


def test_continuous_refinement_edits_one_action_proposal_then_returns_ready() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    workspace.begin_imagination("把夹爪向下并调正", action_id)
    first = workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base"
    )
    target1 = workspace.state.action_proposal.target.pose
    second = workspace.execute_imagination(
        "rotate", axis="z", angle_deg=10, frame="tool"
    )
    target2 = workspace.state.action_proposal.target.pose

    assert first.result["preview"] == second.result["preview"] == "updated"
    assert target2.position_xyz == target1.position_xyz
    assert first.result["action_id"] == second.result["action_id"] == action_id
    assert workspace.state.imagination.instruction == "把夹爪向下并调正"

    handoff = workspace.execute_imagination("finish_imagination", status="ready")
    assert workspace.state.imagination is None
    assert workspace.state.action_proposal.action_id == action_id
    assert workspace.state.action_proposal.refined is True
    assert handoff.result == {"status": "ready", "action_id": action_id}


def test_direct_gripper_action_does_not_create_or_replace_spatial_action() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    result = workspace.execute("open_gripper")

    assert result.ok
    assert workspace.state.action_proposal is None
    action_id = _selected_action(workspace)
    workspace.begin_imagination("向下微调", action_id)
    edited = workspace.execute_imagination(
        "delta_move",
        delta_xyz_m=[0.0, 0.0, -0.01],
        frame="base",
    )

    assert edited.ok
    assert workspace.state.action_proposal.action_id == action_id
    assert workspace.state.action_proposal.target.pose is not None


def test_planned_action_commits_directly_without_imagination() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed).result["action_id"]
    revision = workspace.state.observation_revision
    assert workspace.state.action_proposal.refined is False

    result = workspace.execute("commit", action_id=action_id)

    assert result.ok
    assert workspace.state.imagination is None
    assert workspace.state.observation_revision == revision + 1
    assert workspace.state.action_proposal is None


def test_commit_rejects_action_without_executable_cached_plan() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.0],
    ).result["action_id"]
    revision = workspace.state.observation_revision

    result = workspace.execute("commit", action_id=action_id)

    assert not result.ok
    assert "no executable cached plan" in result.result["error"]
    assert workspace.state.action_proposal.action_id == action_id
    assert workspace.state.imagination is None
    assert workspace.state.observation_revision == revision


def test_main_direct_delta_move_executes_without_action_proposal() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision
    result = workspace.execute(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    )

    assert result.ok
    assert workspace.state.observation_revision == revision + 1
    assert workspace.state.action_proposal is None
    assert workspace.state.last_physical_action.executed_stages == "arm"


def test_refined_commit_invalidates_revision_state() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region_id).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed).result["action_id"]
    workspace.begin_imagination("上移 2cm", action_id)
    workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.02], frame="base"
    )
    action_id = workspace.execute_imagination(
        "finish_imagination", status="ready"
    ).result["action_id"]
    before = workspace.state.observation_revision

    result = workspace.execute("commit", action_id=action_id)

    assert result.ok
    assert api.operation_log == ["arm"]
    assert workspace.state.observation_revision == before + 1
    assert workspace.state.action_proposal is None
    assert region_id not in workspace.state.regions
    assert workspace.state.last_physical_action.intent == "approach can for grasp"
    assert workspace.state.last_physical_action.executed_stages == "arm"
    assert workspace.state.last_physical_action.outcome == "completed"
    assert workspace.state.last_physical_action.target_gripper is None
    assert workspace.state.last_physical_action.requested_arm_delta_base_m is not None
    assert workspace._private.previous_observation is not None
    assert workspace._private.last_physical_artifacts.focus_pose is not None


def test_reject_action_explicitly_discards_review_without_physics() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed).result["action_id"]
    revision = workspace.state.observation_revision

    result = workspace.execute("reject_action", action_id=action_id)

    assert result.ok
    assert result.result == {
        "declined_action_id": action_id,
        "declined_how": "explicit",
        "declined_intent": "approach can for grasp",
    }
    assert workspace.state.action_proposal is None
    assert workspace._private.action_artifacts is None
    assert workspace.state.observation_revision == revision
    assert api.operation_log == []


def test_commit_records_failed_stage_without_claiming_task_effect() -> None:
    arm_api = FakeContextApi()
    arm_api.move_error = True
    arm_workspace = ContextWorkspace(arm_api, "arm failure", motion_backend="pyroki")
    arm_action = _selected_action(arm_workspace)
    arm_workspace.begin_imagination("move", arm_action)
    arm_workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    )
    arm_action = arm_workspace.execute_imagination(
        "finish_imagination", status="ready"
    ).result[
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
    assert arm_workspace.state.action_proposal is None

    gripper_api = FakeContextApi()
    gripper_api.gripper_error = True
    gripper_workspace = ContextWorkspace(
        gripper_api,
        "gripper failure",
        motion_backend="pyroki",
    )
    gripper_result = gripper_workspace.execute("close_gripper")

    assert not gripper_result.ok
    assert gripper_api.operation_log == ["gripper:closed"]
    assert gripper_workspace.state.last_physical_action.executed_stages == "gripper"
    assert gripper_workspace.state.last_physical_action.outcome == "gripper_failed"
    assert gripper_workspace.state.last_physical_action.evidence_invalidated is True


def test_failed_grasp_commit_exposes_source_query_for_redetection() -> None:
    api = FakeContextApi()
    api.move_error = True
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed_id).result["action_id"]
    workspace.begin_imagination("keep selected grasp", action_id)
    action_id = workspace.execute_imagination(
        "finish_imagination", status="ready"
    ).result["action_id"]

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
    assert workspace.state.action_proposal is None


def test_grasp_proposal_returns_general_action_seeds() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    result = workspace.execute("propose_grasps", region_id=region)
    seed_id = result.result["seed_ids"][0]
    assert seed_id.startswith("s")
    families = {
        artifact.family
        for artifact in workspace._private.seed_artifacts.values()
        if artifact.family
    }
    assert families & {"top", "pca", "cgn"}
    selected = workspace.execute("select", seed_id=seed_id)
    assert selected.ok
    assert workspace.state.action_proposal.target.pose is not None


def test_select_forwards_plan_detail_and_reuses_measured_seed_plan() -> None:
    backend = CountingFailedGraspPlanBackend()
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend=backend)
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]

    first = workspace.execute("select", seed_id=seed_id)
    second = workspace.execute("select", seed_id=seed_id)

    assert first.ok
    assert first.result["solve_ik"] == "error"
    assert first.result["detail"] == "CuRobo found no collision-free trajectory"
    assert "plan_reused" not in first.result
    assert second.ok
    assert second.result["plan_reused"] is True
    assert second.result["detail"] == first.result["detail"]
    assert backend.grasp_calls == 1
    assert workspace._private.seed_artifacts[seed_id].motion_plan is not None

    commit = workspace.execute(
        "commit", action_id=second.result["action_id"]
    )
    assert not commit.ok
    assert "no executable cached plan" in commit.result["error"]


def test_select_reuses_successful_seed_plan() -> None:
    backend = RecordingMotionBackend("coarse")
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend=backend)
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]

    first = workspace.execute("select", seed_id=seed_id)
    second = workspace.execute("select", seed_id=seed_id)

    assert first.ok
    assert second.result["plan_reused"] is True
    assert len(backend.grasp_plans) == 1


def test_propose_grasps_keeps_geometric_families_and_limits_graspnet() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]

    result = workspace.execute("propose_grasps", region_id=region)

    assert result.ok
    families = [
        workspace._private.seed_artifacts[seed_id].family
        for seed_id in result.result["seed_ids"]
    ]
    assert "top" in families
    assert "pca" in families
    assert families.count("cgn") <= 2
    selected = result.trace_diagnostics["grasp_candidates"]["selected"]
    assert {item["family"] for item in selected} == set(families)


def test_new_grasp_proposal_replaces_previous_seed_catalogue() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]

    first = workspace.execute("propose_grasps", region_id=region).result["seed_ids"]
    second = workspace.execute("propose_grasps", region_id=region).result["seed_ids"]

    assert set(first).isdisjoint(second)
    assert set(workspace.state.seeds) == set(second)
    assert set(workspace._private.seed_artifacts) == set(second)
    assert not workspace.execute("select", seed_id=first[0]).ok


@pytest.mark.parametrize(
    "body_axis",
    (
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ),
)
def test_pca_grasps_close_across_the_object_long_axis(body_axis: np.ndarray) -> None:
    side_axis = np.array([-body_axis[1], body_axis[0], 0.0])
    center = np.array([0.45, -0.08, 0.04])
    points = np.asarray(
        [
            center + along * body_axis + across * side_axis + height * np.array([0, 0, 1])
            for along in np.linspace(-0.08, 0.08, 17)
            for across in np.linspace(-0.015, 0.015, 5)
            for height in np.linspace(-0.02, 0.02, 5)
        ]
    )

    grasps, _diagnostic = _geometric_grasps(points)
    top = next(item for item in grasps if item.family == "top")
    pca = [item for item in grasps if item.family == "pca"]

    top_rotation = Rotation.from_quat(top.target.quaternion_xyzw).as_matrix()
    assert np.allclose(top_rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
    assert abs(float(np.dot(top_rotation[:, 1], body_axis))) < 1e-6

    assert len(pca) == 2
    for item in pca:
        rotation = Rotation.from_quat(item.target.quaternion_xyzw).as_matrix()
        assert abs(float(np.dot(rotation[:, 1], body_axis))) < 1e-6
        assert abs(float(np.dot(rotation[:, 2], body_axis))) == pytest.approx(1.0)
        assert abs(float(rotation[2, 2])) < 1e-6


def test_top_grasp_yaw_follows_the_obb_narrow_edge() -> None:
    def top_closing_axis(body_axis: np.ndarray) -> np.ndarray:
        side_axis = np.array([-body_axis[1], body_axis[0], 0.0])
        center = np.array([0.45, -0.08, 0.04])
        points = np.asarray(
            [
                center + along * body_axis + across * side_axis + height * np.array([0, 0, 1])
                for along in np.linspace(-0.08, 0.08, 17)
                for across in np.linspace(-0.015, 0.015, 5)
                for height in np.linspace(-0.02, 0.02, 5)
            ]
        )
        grasps, _diagnostic = _geometric_grasps(points)
        top = next(item for item in grasps if item.family == "top")
        return Rotation.from_quat(top.target.quaternion_xyzw).as_matrix()[:, 1]

    closing_for_x = top_closing_axis(np.array([1.0, 0.0, 0.0]))
    closing_for_y = top_closing_axis(np.array([0.0, 1.0, 0.0]))

    assert abs(float(np.dot(closing_for_x, closing_for_y))) < 1e-6


def test_geometric_grasps_canonicalize_the_parallel_jaw_half_turn() -> None:
    center = np.array([0.45, -0.08, 0.04])
    points = np.asarray(
        [
            center + np.array([along, across, height])
            for along in np.linspace(-0.08, 0.08, 17)
            for across in np.linspace(-0.015, 0.015, 5)
            for height in np.linspace(-0.02, 0.02, 5)
        ]
    )
    # Two observed-hand orientations that differ by a half turn about the
    # (downward) approach axis: the same physical grasp, opposite finger sides.
    hand_a = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0]))
    hand_b = Rotation.from_matrix(np.diag([-1.0, 1.0, -1.0]))

    def families(reference: Rotation) -> list[Rotation]:
        grasps, _diagnostic = _geometric_grasps(
            points,
            reference_quaternion_xyzw=tuple(float(v) for v in reference.as_quat()),
        )
        top = next(item for item in grasps if item.family == "top")
        return [Rotation.from_quat(top.target.quaternion_xyzw)]

    for reference in (hand_a, hand_b):
        for rotation in families(reference):
            matrix = rotation.as_matrix()
            # Still a top-down grasp closing across the narrow (Y) edge.
            assert np.allclose(matrix[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
            assert abs(float(matrix[1, 1])) == pytest.approx(1.0, abs=1e-6)
            # The half-turn representative nearer the observed hand is chosen.
            assert (reference.inv() * rotation).magnitude() <= np.pi / 2.0 + 1e-6

    top_a = families(hand_a)[0]
    top_b = families(hand_b)[0]
    relative = top_a.inv() * top_b
    assert relative.magnitude() == pytest.approx(np.pi, abs=1e-6)
    axis = relative.as_rotvec() / relative.magnitude()
    assert abs(float(axis[2])) == pytest.approx(1.0, abs=1e-6)


def test_top_grasp_depth_adapts_to_thin_objects_without_crossing_support() -> None:
    center = np.array([0.48, -0.24, 0.01])
    points = np.asarray(
        [
            center + np.array([x, y, z])
            for x in np.linspace(-0.04, 0.04, 9)
            for y in np.linspace(-0.02, 0.02, 5)
            for z in np.linspace(-0.009, 0.009, 5)
        ]
    )

    grasps, _diagnostic = _geometric_grasps(points)
    top = next(item for item in grasps if item.family == "top")
    bottom_z = float(np.percentile(points[:, 2], 5))
    top_z = float(np.percentile(points[:, 2], 95))

    assert bottom_z <= top.target.position_xyz[2] <= top_z
    assert top.target.position_xyz[2] == pytest.approx(0.5 * (bottom_z + top_z))


def test_top_grasp_is_width_gated_by_the_obb_narrow_extent() -> None:
    center = np.array([0.48, -0.12, 0.05])
    points = np.asarray(
        [
            center + np.array([x, y, z])
            for x in np.linspace(-0.07, 0.07, 15)
            for y in np.linspace(-0.05, 0.05, 11)
            for z in np.linspace(-0.03, 0.03, 7)
        ]
    )

    grasps, diagnostic = _geometric_grasps(points)

    assert all(item.family != "top" for item in grasps)
    rejection = next(
        entry for entry in diagnostic["rejected"] if entry["family"] == "top"
    )
    assert rejection["reasons"] == ["narrow_extent_exceeds_gripper_opening"]
    assert 0.08 < rejection["narrow_extent_m"] <= 0.10
    assert rejection["max_opening_m"] == pytest.approx(0.08)
    advisory = _top_width_advisory(diagnostic)
    assert advisory is not None and "cm" in advisory


def test_degenerate_top_footprint_closes_tangentially_to_the_base_direction() -> None:
    center = np.array([0.45, -0.08, 0.05])
    points = np.asarray(
        [
            center + np.array([x, y, z])
            for x in np.linspace(-0.03, 0.03, 9)
            for y in np.linspace(-0.03, 0.03, 9)
            for z in np.linspace(-0.04, 0.04, 9)
        ]
    )

    grasps, _diagnostic = _geometric_grasps(points)
    top = next(item for item in grasps if item.family == "top")
    rotation = Rotation.from_quat(top.target.quaternion_xyzw).as_matrix()

    radial = center[:2] / np.linalg.norm(center[:2])
    assert np.allclose(rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
    assert abs(float(rotation[2, 1])) < 1e-6
    assert abs(float(np.dot(rotation[:2, 1], radial))) < 1e-6


def test_top_contact_depth_descends_from_the_obb_top_face() -> None:
    center = np.array([0.50, 0.10, 0.12])
    points = np.asarray(
        [
            center + np.array([x, y, z])
            for x in np.linspace(-0.03, 0.03, 9)
            for y in np.linspace(-0.015, 0.015, 5)
            for z in np.linspace(-0.08, 0.08, 17)
        ]
    )

    grasps, _diagnostic = _geometric_grasps(points)
    top = next(item for item in grasps if item.family == "top")

    proxy = fit_gravity_stable_proxy("expected", 0, points, padding_m=0.0)
    assert proxy is not None
    top_face_z = proxy.center_base_xyz[2] + 0.5 * proxy.extent_xyz_m[2]
    assert top.target.position_xyz[2] == pytest.approx(top_face_z - 0.04)
    assert top.target.position_xyz[2] > proxy.center_base_xyz[2]


def test_dispatch_ignores_unknown_provider_field_when_required_argument_exists() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")

    step = workspace.execute(
        "detection_and_sam",
        query="can",
        text="provider-added synonym",
    )

    assert step.ok
    assert step.arguments == {"query": "can"}


def test_dispatch_still_rejects_typo_that_omits_required_argument() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")

    step = workspace.execute("detection_and_sam", qurey="can")

    assert not step.ok
    assert "missing a required argument" in step.result["error"]


def test_propose_grasps_omits_ik_mismatch_seeds() -> None:
    class SelectiveIkApi(FakeContextApi):
        def __init__(self) -> None:
            super().__init__()
            self._reachable_calls = 0

        def seed_ik_matches_target(self, target, prediction):
            del target, prediction
            self._reachable_calls += 1
            return self._reachable_calls != 2

    workspace = ContextWorkspace(SelectiveIkApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    result = workspace.execute("propose_grasps", region_id=region)

    assert result.ok
    dropped = result.trace_diagnostics["grasp_candidates"]["dropped_unreachable"]
    assert dropped
    assert dropped[0]["reason"] == "ik_mismatch"
    assert len(result.result["seed_ids"]) == len(
        result.trace_diagnostics["grasp_candidates"]["selected"]
    )
    assert len(workspace.state.seeds) == len(result.result["seed_ids"])


def test_grasp_proposal_rejects_seed_outside_its_source_geometry() -> None:
    workspace = ContextWorkspace(
        SpatiallyInconsistentGraspApi(),
        "task",
        motion_backend="pyroki",
    )
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]

    result = workspace.execute("propose_grasps", region_id=region)

    rejected = result.trace_diagnostics["grasp_candidates"]["attempts"][0]["rejected"]
    assert rejected[0]["reasons"] == ["outside_source_geometry"]
    assert rejected[0]["source_center_distance_m"] > rejected[0]["source_radius_m"]
    geometric = result.trace_diagnostics["grasp_candidates"]["attempts"][1]
    assert geometric["source"] == "geometric"
    if result.ok:
        assert all(
            artifact.family != "cgn"
            for artifact in workspace._private.seed_artifacts.values()
        )
    else:
        assert workspace.state.seeds == {}


def test_imagination_limit_hands_back_the_last_verified_edit_as_partial() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    original_target = workspace.state.action_proposal.target
    workspace.begin_imagination("test limit", action_id)
    edited = workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.02], frame="base"
    )
    assert edited.ok
    edited_target = workspace.state.action_proposal.target
    assert edited_target != original_target

    result = workspace.limit_imagination()

    assert result.result["status"] == "partial"
    assert result.result["reason"] == "turn_limit"
    assert result.result["action_id"] == action_id
    assert "review the Preview" in result.result["advisory"]
    assert workspace.state.imagination is None
    # Budget exhaustion is not a quality verdict: the planner-checked edit
    # survives for Main's Preview review instead of rolling back.
    assert workspace.state.action_proposal.target == edited_target
    assert workspace.state.action_proposal.refined is True
    assert workspace.state.observation_revision == 1
    assert workspace.state.imagination_attempts.failed == 0
    assert workspace.execute("commit", action_id=action_id).ok
    assert (
        result.trace_diagnostics["imagination_handoff"]["status"] == "partial"
    )
    assert (
        result.trace_diagnostics["imagination_handoff"]["termination_reason"]
        == "turn_limit"
    )


def test_imagination_limit_without_executable_plan_still_rolls_back() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.0],
    ).result["action_id"]
    original = workspace.state.action_proposal
    workspace.begin_imagination("unreachable", action_id)
    workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base"
    )

    result = workspace.limit_imagination()

    # Nothing executable to hand back: this is the only turn-limit path that
    # still restores the entry proposal, reported as plan_unavailable.
    assert result.result["status"] == "failed"
    assert result.result["reason"] == "plan_unavailable"
    assert workspace.state.imagination is None
    assert workspace.state.action_proposal is original
    assert not workspace.execute("commit", action_id=action_id).ok


def test_refinement_cannot_implicitly_create_a_current_tcp_action() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    revision = workspace.state.observation_revision
    with pytest.raises(ContextFunctionError, match="unknown or expired action_id"):
        workspace.begin_imagination("微调当前 TCP", "a-missing")
    assert workspace.state.action_proposal is None
    assert workspace.state.observation_revision == revision


def test_rotate_rejects_angle_outside_ten_degrees() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    workspace.begin_imagination("small rotate only", action_id)
    start_pose = workspace.state.action_proposal.target.pose

    rejected = workspace.execute_imagination(
        "rotate", axis="x", angle_deg=-90, frame="base"
    )
    assert not rejected.ok
    assert "10" in rejected.result["error"]
    assert workspace.state.action_proposal.target.pose == start_pose

    too_large = workspace.execute_imagination(
        "rotate", axis="z", angle_deg=10.1, frame="base"
    )
    assert not too_large.ok
    assert "10" in too_large.result["error"]

    accepted = workspace.execute_imagination(
        "rotate", axis="z", angle_deg=-10, frame="base"
    )
    assert accepted.ok
    assert accepted.result["preview"] == "updated"


def test_gizmo_does_not_require_a_following_rotate() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    workspace.begin_imagination("inspect then translate", action_id)
    gizmo = workspace.execute_imagination(
        "show_rotation_gizmo", frame="base", axis="z"
    )
    assert gizmo.ok
    moved = workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    )
    assert moved.ok
    assert moved.result["preview"] == "updated"


def test_agent_failed_imagination_is_geometry_unresolved_and_records_attempts() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    workspace.begin_imagination("cannot resolve contact", action_id)

    result = workspace.execute_imagination("finish_imagination", status="failed")

    assert result.result["status"] == "failed"
    assert result.result["reason"] == "geometry_unresolved"
    assert result.result["action_id"] == action_id
    assert workspace.state.action_proposal.action_id == action_id
    attempts = workspace.state.imagination_attempts.summary()
    assert attempts["total"] == 1
    assert attempts["failed"] == 1
    assert attempts["last_reason"] == "geometry_unresolved"
    packet = ContextCompiler().compile(workspace)
    assert packet.manifest()["imagination_attempts"]["failed"] == 1


def test_ready_without_executable_plan_is_plan_unavailable() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.0],
    ).result["action_id"]
    workspace.begin_imagination("accept unreachable", action_id)

    result = workspace.execute_imagination("finish_imagination", status="ready")

    assert result.result["status"] == "failed"
    assert result.result["reason"] == "plan_unavailable"
    assert workspace.state.action_proposal.action_id == action_id
    assert workspace.state.imagination is None


def test_unplannable_edit_keeps_the_last_executable_proposal() -> None:
    # propose_pose consumes the first plan, the accepted edit the second.
    backend = PlanBudgetMotionBackend(successes=2)
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend=backend)
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.0],
    ).result["action_id"]
    workspace.begin_imagination("lower onto the contact", action_id)

    accepted = workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base"
    )
    assert accepted.result["preview"] == "updated"
    reachable_target = workspace.state.action_proposal.target

    rejected = workspace.execute_imagination(
        "delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base"
    )

    # The transaction reports success: nothing was corrupted, the target simply
    # did not move.  Main-visible state must still be the last executable one.
    assert rejected.ok
    assert rejected.result["preview"] == "unchanged"
    assert rejected.result["reason"] == "plan_unavailable"
    assert workspace.state.action_proposal.target == reachable_target
    assert workspace._private.action_artifacts.motion_plan.prediction.solve_ik == "returned"

    ready = workspace.execute_imagination("finish_imagination", status="ready")

    assert ready.result["status"] == "ready"
    assert workspace.state.action_proposal.target == reachable_target
    assert workspace.state.action_proposal.refined is True


def test_unplannable_edit_does_not_become_the_next_edit_reference() -> None:
    backend = PlanBudgetMotionBackend(successes=1)
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend=backend)
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.0],
    ).result["action_id"]
    workspace.begin_imagination("rotate into the opening", action_id)
    entry_target = workspace.state.action_proposal.target

    assert (
        workspace.execute_imagination(
            "rotate", axis="z", angle_deg=8.0, frame="base"
        ).result["preview"]
        == "unchanged"
    )
    assert (
        workspace.execute_imagination(
            "rotate", axis="z", angle_deg=8.0, frame="base"
        ).result["preview"]
        == "unchanged"
    )

    # Both rejected edits derive from the entry pose, never from each other.
    assert workspace.state.action_proposal.target == entry_target


def test_imagination_attempts_clear_when_the_revision_advances() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    action_id = _selected_action(workspace)
    workspace.begin_imagination("cannot resolve", action_id)
    workspace.execute_imagination("finish_imagination", status="failed")
    assert workspace.state.imagination_attempts.summary()["failed"] == 1

    assert workspace.execute("commit", action_id=action_id).ok

    assert workspace.state.imagination_attempts.summary() is None
    packet = ContextCompiler().compile(workspace)
    assert "imagination_attempts" not in packet.manifest()
