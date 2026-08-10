"""Agent-visible contracts for Main and Imagination ownership."""

from __future__ import annotations

import json
from typing import Any

FUNCTION_NAMES = (
    "detection_and_sam",
    "locate_point",
    "propose_grasps",
    "propose_pose",
    "select",
    "delta_move",
    "rotate",
    "open_gripper",
    "close_gripper",
    "commit",
    "done",
)

IMAGINATION_FUNCTION_NAMES = (
    "delta_move",
    "rotate",
    "open_gripper",
    "close_gripper",
    "finish_imagination",
)

SYSTEM_PROMPT = """\
你是 Main Agent，通过 Visual Action Workspace 控制 LIBERO-PRO 机器人完成任务。

Canvas 上层 OBSERVED NOW 是唯一真实视觉：左侧为当前 agentview，中间为当前融合 RGB-D
场景，右侧为当前本体状态。下层 IMAGINATION 使用同一真实点云；灰色是当前机器人，紫色
TARGET GRIPPER 是未执行目标。规划与紫色几何都不是物理事实。

你负责理解任务、调用感知、选择动作起点，并审查 Imagination 最终交回的 ActionReview。
select、propose_pose、delta_move、rotate、open_gripper、close_gripper 会把控制权交给独立的
Imagination Agent；它会连续微调并交回 review_required 或 failed。review_required 只表示轮到你审查，
不是批准：你必须查看最终 Preview，只有自己判断几何合理时才 commit；否则换 seed 或重新进入
Imagination。
不要替它执行局部微调。
当 Function 会启动 Imagination 时，必须在 refinement_goal 参数中写一句短的目标几何；不要把
整段理由、旧失败、预算或未经验证的物理效果写进去。

detection_and_sam 的 region 是当前观测中目标身份与二维位置的权威检测/分割结果，
但不证明接触、抓持、支撑或包含，也不刷新 observation。
commit 是唯一改变真实世界的 Function。命令成功不等于任务效果成功。
若 Canvas 下层显示 POST-COMMIT VERIFY，BEFORE 与 CURRENT 都是真实 observation；
Last Physical Action 只说明刚执行的意图、阶段和控制结果，不声称物体已被抓住、移动或释放。
用 CURRENT 中的可见变化判断该动作是否产生了任务相关效果，并据此选择下一步。
每轮必须且只能调用一个 Function。调用前只写一句简短依据。坐标为 robot-base frame、单位米，
四元数为 xyzw；所有 evidence/seed ID 只在当前真实观测有效。
"""

IMAGINATION_SYSTEM_PROMPT = """\
你是 Imagination Agent。你的唯一任务是在不改变真实世界的前提下，检查并微调当前 ActionTarget。

Canvas 上层 OBSERVED NOW 是真实世界；下层 IMAGINATION 是在同一当前点云上的虚拟 target。
紫色 TARGET GRIPPER 和规划状态不证明接触、抓持、释放或包含。结合主视角、融合 RGB-D、
固定角落的 BASE/WORLD 坐标提示、refinement goal 和 Edit Summary 审查当前 target。

每轮只做三种选择之一：若当前 target 已满足目标，调用 finish_imagination(status="ready")；若能
指出一个当前可见的几何缺陷，只做一次 delta_move、rotate、open_gripper 或 close_gripper；若该
target 无法可靠修复，调用 status="failed"。不要为了用满轮数而编辑。

先判断空间姿态，再设置最终 gripper target；open/close 只改变虚拟目标，不能模拟接触结果。
若一次空间编辑使 motion prediction 变为 error，不要沿同一趋势盲目累计；应撤回、换方向，或
在无法恢复时结束为 failed。Edit Summary 中的累计位移、累计旋转和最近两次编辑是当前控制
状态，不是对话历史；若两次编辑相互抵消且没有新的可见改进，应接受当前 target 或失败，而不是
继续振荡。refinement goal 是审查意图，不是已经成立的视觉事实。

不要调用感知、commit 或 done。每轮必须且只能调用一个 Function，调用前只写一句简短依据。
delta_move 的 frame 必须显式填写；每轴单次不超过 0.03m。rotate 的 frame 必须显式填写。
"""


def _function(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]] | None = None,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        },
    }


def _refinement_goal() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "一句短的目标几何；只描述希望 Imagination 检查或达到什么",
    }


def _edit_definitions(*, main: bool) -> dict[str, dict[str, Any]]:
    frame = {
        "type": "string",
        "enum": ["base", "tool"],
        "description": "必须显式选择 robot-base 或 TCP 局部坐标系",
    }
    extra = {"refinement_goal": _refinement_goal()} if main else {}
    extra_required = ("refinement_goal",) if main else ()
    return {
        "delta_move": _function(
            "delta_move",
            "编辑想象目标的位置；只更新 preview，不执行。可连续调用。",
            {
                "delta_xyz_m": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "minimum": -0.03,
                        "maximum": 0.03,
                    },
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "所选 frame 中的 [dx,dy,dz]，每轴单次不超过 0.03m",
                },
                "frame": frame,
                **extra,
            },
            ("delta_xyz_m", "frame", *extra_required),
        ),
        "rotate": _function(
            "rotate",
            "编辑想象目标的方向；只更新 preview，不执行。可连续调用。",
            {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {
                    "type": "number",
                    "minimum": -90.0,
                    "maximum": 90.0,
                    "description": "非零角度，绝对值不超过 90°",
                },
                "frame": frame,
                **extra,
            },
            ("axis", "angle_deg", "frame", *extra_required),
        ),
        "open_gripper": _function(
            "open_gripper",
            "把想象目标的夹爪状态设为 open；不会打开真实夹爪。",
            extra,
            extra_required,
        ),
        "close_gripper": _function(
            "close_gripper",
            "把想象目标的夹爪状态设为 closed；不会闭合真实夹爪，也不保证抓住物体。",
            extra,
            extra_required,
        ),
    }


def function_definitions() -> list[dict[str, Any]]:
    edits = _edit_definitions(main=True)
    within = {"type": "string", "description": "可选的当前 region 搜索范围"}
    common = [
        _function(
            "detection_and_sam",
            "对当前真实 observation 依次执行语义 bbox detection 和 SAM 分割，返回当前图像中的二维 region；不刷新 observation、不移动机器人，也不证明接触、抓持、支撑或包含关系。仅在需要新的目标 region 时调用。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "locate_point",
            "定位语义操作点并提升为 robot-base XYZ；不移动机器人。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "为 region 生成多个粗略 ActionSeed；不执行，也不保证抓取成功。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "propose_pose",
            "从 point 和 offset 创建单个空间目标并进入 Imagination；不执行。",
            {
                "point_id": {"type": "string"},
                "offset_xyz": {"type": "array", "items": {"type": "number"}},
                "quaternion_xyzw": {"type": "array", "items": {"type": "number"}},
                "refinement_goal": _refinement_goal(),
            },
            ("point_id", "offset_xyz", "refinement_goal"),
        ),
        _function(
            "select",
            "选择一个 ActionSeed 并进入 Imagination；不执行。",
            {
                "seed_id": {"type": "string"},
                "refinement_goal": _refinement_goal(),
            },
            ("seed_id", "refinement_goal"),
        ),
        edits["delta_move"],
        edits["rotate"],
        edits["open_gripper"],
        edits["close_gripper"],
        _function(
            "commit",
            "唯一物理操作，也表示 Main 对当前 ActionReview 的显式批准；执行后刷新真实 observation。",
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "done",
            "根据当前真实视觉声明 episode 结束；success 只是 Agent belief。",
            {"success": {"type": "boolean"}},
            ("success",),
        ),
    ]
    return common


def imagination_function_definitions() -> list[dict[str, Any]]:
    edits = _edit_definitions(main=False)
    return [
        edits["delta_move"],
        edits["rotate"],
        edits["open_gripper"],
        edits["close_gripper"],
        _function(
            "finish_imagination",
            "结束本次想象审查。ready 请求 Main 审查最终 Preview；failed 放弃该 target。",
            {"status": {"type": "string", "enum": ["ready", "failed"]}},
            ("status",),
        ),
    ]


def parse_action(
    payload: str | dict[str, Any], *, allowed: tuple[str, ...] = FUNCTION_NAMES
) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"action is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or name not in allowed:
        raise ValueError(f"unknown function '{name}'")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments are not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return name, arguments


__all__ = [
    "FUNCTION_NAMES",
    "IMAGINATION_FUNCTION_NAMES",
    "IMAGINATION_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "function_definitions",
    "imagination_function_definitions",
    "parse_action",
]
