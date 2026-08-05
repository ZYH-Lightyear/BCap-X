"""The small, agent-visible M1.4 function contract.

JSON Schema communicates argument shape to a model.  It is deliberately not a
behaviour state machine: this module has no phase, allowed-next-tool list,
numeric workspace limits, or semantic ordering rules.
"""

from __future__ import annotations

import json
from typing import Any

FUNCTION_NAMES = (
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

SYSTEM_PROMPT = """\
你通过 Visual Action Workspace Context Runtime 控制 LIBERO-PRO 中的机器人。
每轮你会收到一张当前 Context 图、一个 minimal manifest，以及最近最多三个完整的 Function
transaction。User Task 只定义最终目标，不规定动作流程。每轮必须且只能调用一个 Function。

Context 图是当前 episode 中唯一的视觉观测。不存在图外相机、隐含物体状态或自动更新的世界模型。
你必须根据当前视觉世界、Function 的真实因果效果和通用物理知识自主决定下一步，不得把历史调用
顺序理解为任务阶段，也不得假定任何 Function 存在默认的下一个 Function。

视觉接口：
- AGENTVIEW · PRIMARY 是未经标注的当前全局 RGB，用于理解对象身份、支撑面、容器、障碍物、
  遮挡和整体空间关系。必须观察底层场景，不得只读文字或 UI 标记。
- GRIPPER-LOCAL · GEOMETRY 是当前 TCP 周围由 agentview 与 wrist RGB-D 融合得到的局部几何。
  LOCAL 3/4 显示局部整体关系；JAW PLANE 显示两指闭合方向上的几何关系。自然颜色点只代表当前
  传感器可见表面；空白表示未观测区域，不表示自由空间。蓝色几何是由当前 joints FK 得到的真实
  夹爪自身，不是待操作物体。
- Dynamic Decision Workspace 显示当前相关 grounding、candidate、Action Proposal、receipt 或
  error。它只是证据组织方式，不表示任务阶段或下一步动作。
- 紫色几何表示尚未执行的预测姿态。`solve_ik: returned`、trajectory/collision checked 只描述
  求解器或规划器完成了对应计算，不证明位姿在任务语义上正确，也不证明世界效果已经发生。

请利用你已有的世界知识理解重力、刚体、接触、支撑、遮挡、碰撞、容器关系和物体 affordance，
并预测可用动作的直接物理后果。世界知识只能生成受当前视觉证据约束的假设，不能覆盖视觉证据或
凭空生成接触、持有、包含和任务成功等事实。

证据边界：
- `inspect` 返回的 region 是当前 revision 中 query 身份和二维位置的权威依据，不得因自己的视觉
  分类而将其改认成其他对象；但 region 不证明接触、支撑、持有或包含关系。
- candidate 和 Action Proposal 是待验证的几何/动作假设，不是指令或成功保证。
- Function 成功返回只证明调用被处理。receipt、manifest、绿色标记和 gripper_opening 都不证明
  预期的物理或任务效果发生。不得虚构 result、reward、environment success 或隐藏状态。

选择 Function 时：
- 先判断当前与 User Task 有关的空间关系；
- 预测所选 Function 的直接物理后果，并与当前明显的替代动作比较；
- 综合任务进展、碰撞风险、可逆性和降低不确定性的价值作出选择；
- 证据不足时，优先选择可逆、小幅、能够获取信息或改善几何关系的感知或运动；
- 物理动作后只根据新的当前图像更新判断，不把预期效果当成已经发生。

Function 的因果边界：
- 感知 Function 只创建当前 revision 的证据，不移动机器人。
- select、propose_pose、delta_move 和 rotate 只创建或编辑 Action Proposal，不直接移动机器人。
- commit 只执行指定 active Action Proposal 中缓存的机械臂运动，不自动改变夹爪。
- open_gripper 和 close_gripper 只在当前机械臂位姿改变两指，不移动 TCP，也不保证接触或夹持。
- done(success=...) 只表达你的最终判断；必须由当前视觉证据支持。

所有公共 3D 坐标使用 robot-base frame，单位为米；公共四元数顺序为 xyzw。frame=base 使用固定
robot-base 坐标轴，frame=tool 使用参考 TCP 局部坐标轴。region、point、candidate 和 action id
只在创建它们的 revision 中有效。gripper_opening 是 0=closed、1=open 的连续数值，本身不表示
是否夹住物体。

每次 Function call 前必须输出一条简短的“决策依据”，只包含：当前关键视觉关系、所选动作的
直接预期效果，以及它为什么优于当前明显的替代动作。不要输出冗长思维过程；如果视觉证据不足，
应明确保留不确定性，而不是编造确定结论。
"""


def _function(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def function_definitions() -> list[dict[str, Any]]:
    """Return the M1.4 hybrid tool list in deterministic order."""

    within = {
        "type": "string",
        "description": "可选：在该当前 revision region 内搜索",
    }
    return [
        _function(
            "inspect",
            "在当前图像中创建带 bbox 和私有 mask 的 region 证据。",
            {
                "query": {"type": "string", "description": "对象、部件、表面或区域"},
                "within_region_id": within,
            },
            ["query"],
        ),
        _function(
            "locate_point",
            "标记一个语义图像点，并用 RGB-D 将其提升为 robot-base XYZ。",
            {
                "query": {"type": "string", "description": "语义操作点"},
                "within_region_id": within,
            },
            ["query"],
        ),
        _function(
            "propose_grasps",
            "从当前 region 的私有 mask 生成 grasp candidates。",
            {"region_id": {"type": "string", "description": "当前 region 证据"}},
            ["region_id"],
        ),
        _function(
            "propose_pose",
            "以 point XYZ 加显式 robot-base offset 创建动作；省略 orientation 时保持当前 EE orientation。",
            {
                "point_id": {"type": "string", "description": "当前 point 证据"},
                "offset_xyz": {
                    "type": "array",
                    "description": "robot-base [dx, dy, dz]，单位为米",
                    "items": {"type": "number"},
                },
                "quaternion_xyzw": {
                    "type": "array",
                    "description": "可选 robot-base target quaternion [x, y, z, w]",
                    "items": {"type": "number"},
                },
            },
            ["point_id", "offset_xyz"],
        ),
        _function(
            "select",
            "从当前 candidate 创建 active action，并运行当前 motion backend 的 prediction/planning。",
            {
                "candidate_id": {
                    "type": "string",
                    "description": "当前 grasp candidate",
                }
            },
            ["candidate_id"],
        ),
        _function(
            "delta_move",
            "相对当前真实 TCP 或指定 active action target 平移，并创建新的 Action Proposal；不会直接执行。",
            {
                "delta_xyz_m": {
                    "type": "array",
                    "description": "所选 frame 中的 [dx, dy, dz] 米制偏移；每轴限于 [-0.03, 0.03]",
                    "items": {"type": "number", "minimum": -0.03, "maximum": 0.03},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "frame": {
                    "type": "string",
                    "enum": ["base", "tool"],
                    "default": "base",
                    "description": "偏移采用 robot-base 或参考 TCP 局部坐标系",
                },
                "action_id": {
                    "type": "string",
                    "description": "可选：从该当前 active action target 继续微调",
                },
            },
            ["delta_xyz_m"],
        ),
        _function(
            "rotate",
            "绕所选坐标系的 x/y/z 轴旋转当前真实 TCP 或指定 active action target，并创建新的 Action Proposal。",
            {
                "axis": {
                    "type": "string",
                    "enum": ["x", "y", "z"],
                    "description": "旋转轴",
                },
                "angle_deg": {
                    "type": "number",
                    "minimum": -90.0,
                    "maximum": 90.0,
                    "description": "有符号角度，单位 degree；必须非零",
                },
                "frame": {
                    "type": "string",
                    "enum": ["base", "tool"],
                    "default": "tool",
                    "description": "旋转轴属于 robot-base 或参考 TCP 局部坐标系",
                },
                "action_id": {
                    "type": "string",
                    "description": "可选：从该当前 active action target 继续微调",
                },
            },
            ["axis", "angle_deg"],
        ),
        _function(
            "commit",
            "执行指定的当前 action 并刷新 observation；不会改变夹爪状态。",
            {"action_id": {"type": "string", "description": "active action proposal"}},
            ["action_id"],
        ),
        _function("open_gripper", "打开夹爪并刷新 observation。"),
        _function("close_gripper", "闭合夹爪并刷新 observation。"),
        _function(
            "done",
            "使用 Agent 自己的成功判断结束 episode。",
            {
                "success": {
                    "type": "boolean",
                    "description": "仅表示 Agent belief，绝非 environment truth",
                }
            },
            ["success"],
        ),
    ]


def parse_action(payload: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse only what dispatch requires; handler/backend errors stay visible."""

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"action is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("action missing name")
    if name not in FUNCTION_NAMES:
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


__all__ = ["FUNCTION_NAMES", "SYSTEM_PROMPT", "function_definitions", "parse_action"]
