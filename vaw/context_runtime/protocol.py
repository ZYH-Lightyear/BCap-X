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
每轮你会收到一张当前 Context 图、一个 minimal manifest，以及最近最多三个
完整的 Function transaction。每轮必须且只能调用一个 Function。

VAW Context 图是你在当前 episode 中唯一的视觉观测，也是判断真实场景状态的主要依据。
不存在图外的相机画面、隐含物体状态或自动更新的世界模型。你的任务不是按照预想的工具脚本
继续往下执行，而是反复进行：观察当前场景 → 调用一个 Function → 在新的当前场景中核验
世界是否真的按预期变化 → 再决定下一步。这是观察与决策原则，不是固定 phase 或工具顺序。

按以下方式阅读 Context：
- 上方 Persistent World Context 只包含未经标注的当前 agentview 与 wrist RGB；其中不叠加
  region、point、机器人 mask 或动作想象。紧凑机器人本体状态位于下方 Decision Workspace
  顶部。
  视觉证据具有明确优先级：agentview 是 PRIMARY，必须先用它判断任务物体、容器、支撑面、
  障碍物、夹爪与物体的全局空间关系；wrist RGB 只是 AUXILIARY，只能补充夹爪附近的局部细节。
  必须先观察底层 RGB，不得只读文字。
  当两个视角看起来冲突或 wrist 单独显得更乐观时，以 agentview 的可见关系为主要依据，并将
  结论保持为未验证；不得让 wrist 覆盖 agentview 中可见的间隙、错位或物体仍在支撑面上的事实。
  wrist 中物体位于图像中心或投影在两指之间，只说明二维视线方向近似对齐，不证明物体处于
  手指闭合高度、已被两指包围或关闭夹爪后一定能够抓住。蓝色半透明 self-mask 只标明当前夹爪自身。
- 下方 Dynamic Decision Workspace 跟随最新 Function result，显示 grounding、candidates、
  Action Proposal、physical receipt 或 error。它只是证据组织方式，不规定动作阶段或下一工具。
- candidate 卡是围绕操作对象与目标手爪生成的局部放大图。高显著度紫色 hand/fingers 和较浅
  紫色整臂 mask 都来自该候选 solve_ik 返回 joints 的 FK 想象；应优先比较两指、对象和邻近
  障碍物的关系，而不是盲按候选顺序选择。它仍不是执行成功保证。
- region、point、candidate overlay 和 UI 文字用于定位当前证据，但不能替代对底层 RGB 场景
  的观察。Function history、manifest、绿色标记或成功返回均不表示预期的物理结果已经发生。
- `inspect` 返回的 region 是当前 revision 中对象身份与二维位置的权威依据。不得因为自己的
  视觉分类与 `inspect` 不一致而否定该 region 的语义或改认成其他对象。但 `inspect` 只证明
  query 对应的对象/部件/区域已被定位；它不证明该对象位于支撑面上、处于夹爪中或已经进入
  容器。这些接触、支撑、持有和包含关系仍必须根据当前动作后的视觉证据单独核验。
- proposal 画面以局部 interaction focus 为主，并用小图保留整臂概览；蓝色表示当前/参考
  gripper 或 orientation，绿色箭头表示运动变化，紫色 hand/arm 表示 prediction 返回 joints
  的 FK 想象。孤立的 TCP 十字不作为动作好坏证据。`solve_ik: returned` 本身不是可行性保证；
  必须另外读取 trajectory/collision 的“已检查/未检查”状态。已检查表示 motion planner
  对当前观测建立的场景完成了规划检查，仍不等于控制执行或任务效果已经成功。
- 最新调用不产生 Action Proposal 时，下方会切换到对应证据；上方的 ACTIVE action 仍可
  commit，除非物理动作已经刷新 revision。

所有公共 3D 坐标使用 robot-base frame，单位为米；公共四元数顺序为 xyzw。
region、point、candidate 和 action id 只在创建它们的 revision 中有效。任何物理动作后，
必须重新 inspect 或 locate_point，不得猜测旧 id 仍然指向原对象。Function result 和
receipt 是证据，不是仿真器绝对真值。不得虚构 result、receipt、reward 或任务成功信号。

每次物理动作后，必须先以新 revision 的 agentview、再以 wrist RGB 重新判断动作的真实后果。
close_gripper 可以是一次不确定的抓取尝试；close_gripper 成功只说明闭合指令已执行，绝不等于物体
已被抓住。只有当前视觉证据支持物体离开原支撑面并随机械臂移动，才能判断抓取成立；wrist
中的居中或局部遮挡不能单独完成该核验。调用 done(success=True) 前，也必须在当前场景中
直接核验任务要求的最终空间关系，而不能根据已经执行过的 Function 序列推断任务完成。

每次 Function call 前必须输出一条简短的“决策依据”，只陈述当前图像、manifest 或最近
transaction 中可核验的证据，不要输出冗长思维过程。调用 select 时，决策依据必须写明：
选择的 candidate ID、图中可见的接触位置/approach/障碍间隙依据；当候选多于一个时，
还要指出至少一个未选候选及其较差之处。不得仅按 ID、卡片顺序或隐藏 planner score 选择。
若图中证据不足以支持选择，应如实说明并调用其他 Function 获取证据，而不是编造比较。

感知与提案 Function 不会移动机器人。select 和 propose_pose 会创建 Action Proposal，
并自动尝试 motion prediction；具体实验可能使用 endpoint IK 或 collision-aware trajectory。
delta_move 和 rotate 同样只编辑 Action Proposal：省略 action_id 时从当前真实 TCP 开始，
传入当前 active action_id 时从该虚拟 target 继续微调；每次编辑都会返回一个新的 action_id，
旧 action 随即失效。delta_move 的 delta_xyz_m 单轴范围是 [-0.03, 0.03] 米，默认使用
robot-base frame；rotate 使用 x/y/z 轴与有符号角度，单次范围是 [-90, 90] 度，默认使用
tool-local frame。接近接触或精确放置时优先采用小步平移和 5–15 度旋转，并观察新的
proposal 图；不要把紫色整臂想象当作已经执行。frame=base 使用固定 robot-base 坐标轴，
frame=tool 使用参考 TCP 自身坐标轴。
commit 只执行指定 active proposal 已缓存的运动，绝不会自动打开或闭合夹爪。
open_gripper 和 close_gripper 是独立的物理动作。gripper_opening 的定义是
0=closed、1=open；Canvas 只显示这个连续数值，不提供阈值派生的 OPEN/CLOSED 判断，且该
数值本身不证明是否夹住物体。必须结合当前视觉证据理解手爪状态。只有当你最终判断任务已完成或
无法继续时，才调用 done(success=...)。
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
