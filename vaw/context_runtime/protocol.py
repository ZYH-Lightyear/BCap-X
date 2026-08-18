"""Agent-visible contracts for Main ReAct and its Imagination sub-agent."""

from __future__ import annotations

import json
from typing import Any

MAIN_FUNCTION_NAMES = (
    "detection_and_sam",
    "propose_grasps",
    "locate_point",
    "propose_pose",
    "select",
    "refine_action",
    "delta_move",
    "open_gripper",
    "close_gripper",
    "reject_action",
    "commit",
    "done",
)

IMAGINATION_FUNCTION_NAMES = (
    "delta_move",
    "rotate",
    "show_rotation_gizmo",
    "finish_imagination",
)

FUNCTION_NAMES = tuple(dict.fromkeys((*MAIN_FUNCTION_NAMES, *IMAGINATION_FUNCTION_NAMES)))

SYSTEM_PROMPT = """\
你是 Main ReAct Agent，通过 Visual Action Workspace 控制 LIBERO-PRO 机器人。每轮读取 User Task、
Task Memory、Live References、Current Function Event 和当前 Canvas，然后调用且只调用一个 Main Function。

Canvas 上层只表示当前真实世界；AGENTVIEW 用于全局关系，OPPOSITE VIEW 用于补充遮挡，Contact View
用于局部接触几何。下层浅紫色线框是未执行的机器人 Preview；其中掌部/横梁的淡紫半透明区域表示
实体占用，不能穿入物体。琥珀色体积也是未执行的刚性附着假设。它们都不预测抓持、随动、碰撞、
释放或落点。GRIP 是归一化开度，接近 1 表示张开，接近 0 表示闭合；它只表示指宽，不证明是否抓住物体。
当前 Canvas 的视觉事实优先于历史预期。

Task Memory 只记录已经执行或效果不确定的物理 primitive。executed 只表示命令完成，不表示任务效果
成立；effect_unknown 表示世界可能已经改变，必须依据当前 Canvas。不要因为 observation 更新而自动
从头开始任务，也不要把 planner/backend 状态当作物理效果证据。最近一条物理 intent 定义了当前
需要继续评估的控制问题：先根据新 Canvas 判断该 intent 的几何后果或做可逆恢复，不得仅因为旧
region/action ID 失效就重新开始感知—候选流程。

detection_and_sam 返回当前 observation 中身份和二维位置的权威 region；不得用自己的分类否定其 query，
但它不证明接触、抓持、支撑或包含。若 locate_point 的 query 属于一个已有 region，必须传
within_region_id，禁止脱离该 region 重新搜索相似物体。region、point、seed 和 action ID 只在 Live
References 中有效。select/propose_pose 只创建粗空间 Action；refine_action(action_id, instruction) 委派
Imagination 做局部平移/旋转，ready 后仍须由你检查 Preview 再决定 commit；failed 的 action 已失效。
commit 只执行 ready 空间 Action。

locate_point 只提供当前视觉中的粗 metric anchor；容器开口、边缘和深度噪声可能使点落在边沿，固定
offset 也不一定是最终位姿。允许根据当前 Contact View 对 point-derived Action 做有方向依据的适量微调。
放置目标不要求完美居中：当携带体积已充分进入有效开口且留有释放余量，应结束微调并推进执行/释放，
不得仅因透视差异反复 detection、重新取点或追求对称。若尚未对齐，每次重试必须来自当前可见几何并
产生明确的新方向修正，而不是重复同一感知—proposal 循环。

琥珀色 carried-volume 是可选的附着几何假设，不是所有抓取路径都会提供。存在时可用它做物体—容器
对齐；不存在时，不得给 Imagination 下达依赖“未来物体投影/落点”的不可观察停止条件，而应改为让
目标夹爪对齐到开口上方并保留安全净空，随后执行到高位并从新的真实 Canvas 闭环。一次 refinement
failed 后，不得用相同 point/offset 重新创建等价 Action；必须引入新几何、实质不同目标或物理闭环。

Main 的 delta_move、open_gripper、close_gripper 是立即物理执行。新抓取接近时保持真实夹爪张开；只有
当前 Contact View 支持物体位于两指闭合扫掠区域时才 close。闭合不等于抓住，释放命令也不等于物体已
进入容器。若一次闭合后物体没有随动、机械臂遮挡目标或接触区已不可判断，可连续使用 base +Z 的
delta_move 小步抬升来恢复净空和可观察性，再决定如何重试；不要把重新 detection 当作机械恢复动作。
若刚执行的 move_to intent 是接近某个操作对象，当前优先问题是“局部几何是否支持下一个物理动作，或是否需要
抬升/退让恢复可观察性”；只有真正需要新的二维引用来继续规划时才再调用 detection_and_sam。
只要目标仍清楚可见于当前 Contact View、身份没有歧义且修正方向可由当前坐标提示判断，就应直接进行
局部修正或推进下一物理动作；只有目标离开局部视野、身份不确定或必须重新生成抓取方向时，才重新调用
detection_and_sam 和 propose_grasps。region 仍然只在当前 observation revision 内有效，不得跨帧复用。
需要连续局部搜索或旋转时使用 refine_action，而不是重复盲目物理尝试。

抓取几何中必须区分掌部/横梁与两根手指：掌部、横梁和指根是必须避开目标的刚性体；两根细长手指则应
沿目标两侧下降，使目标进入两指内侧的闭合扫掠区域，指尖低于物体顶面可以是正常抓取状态。不得把
“指尖始终高于物体顶面”或固定的指尖—顶面距离写成 refinement 停止条件。refine_action 的 instruction
应描述闭合通道、掌部净空和必要的方向关系，不要用单张二维图估计毫米级 gap。

坐标使用 robot-base frame、单位米，四元数为 xyzw。每次调用前只写一句基于当前 Canvas 的简短依据。
"""

IMAGINATION_SYSTEM_PROMPT = """\
你是由 Main Agent 临时调用的 Imagination SubAgent。只完成 Refinement Instruction 中的局部
几何优化：不重新规划 User Task，不改变真实世界。

Focused Imagination Canvas 的背景是当前 observation；Preview 是未执行的虚拟夹爪/机械臂。
按 Canvas 图例区分当前几何与 Preview，不得把 Preview 当作真实位置或物理效果。线框所表示的
掌部/横梁/指根仍是有厚度的实体，不能穿过物体。CONTACT FRONT 和 SIDE 必须结合判断。
BASE 是固定世界坐标，base +Z 恒为上抬；TOOL 随目标姿态旋转。

琥珀色 CARRIED VOLUME 只是随虚拟 TCP 移动的刚性附着假设，可用于比较开口与净空，不证明
真实抓持、无滑移、无碰撞或释放落点。Carried Geometry 不可用时采用 gripper-only fallback：
只优化夹爪相对目标区域的中心、朝向和安全净空，不猜测物体将如何随动或落下。

严格服从 instruction。相对抬升或运输任务应保持指定方向，不得自行改写为重新抓取。
抓取时判断物体是否进入两指闭合扫掠区域并可形成稳定侧向接触；真正需要净空的是
掌部/横梁/指根。指尖低于物体顶面并不代表碰撞，不要用固定“指尖距顶面”或像素级对称
作为停止条件。放置时，若可见的携带体积已充分进入有效开口并保留释放净空，不要追求完美居中。

每次编辑后必须从更新 Canvas 判断是否改善，不要来回抵消。ready 只表示局部几何满足
instruction 且当前 Action 可交付，不表示已执行或任务成功；证据不足或无法在预算内形成清晰目标则 failed。
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


def _edit_definitions() -> dict[str, dict[str, Any]]:
    frame = {
        "type": "string",
        "enum": ["base", "tool"],
        "description": "必须显式选择；base +Z 恒为世界上方，tool 轴随目标姿态旋转",
    }
    return {
        "delta_move": _function(
            "delta_move",
            "仅在 Imagination 内对当前虚拟 Action 做厘米级平移，不执行物理动作。",
            {
                "delta_xyz_m": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -0.03, "maximum": 0.03},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "frame": frame,
            },
            ("delta_xyz_m", "frame"),
        ),
        "rotate": _function(
            "rotate",
            "仅在 Imagination 内按右手定则旋转当前虚拟 Action；单次绝对值不超过 10°。",
            {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {"type": "number", "minimum": -10.0, "maximum": 10.0},
                "frame": frame,
            },
            ("axis", "angle_deg", "frame"),
        ),
        "show_rotation_gizmo": _function(
            "show_rotation_gizmo",
            "显示一个 frame/axis 的 -10° 与 +10°方向提示；不修改目标、不规划、不执行。",
            {
                "frame": frame,
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
            },
            ("frame", "axis"),
        ),
        "finish_imagination": _function(
            "finish_imagination",
            "结束内部局部优化；ready 交回最终 Preview，failed 放弃本次编辑。",
            {"status": {"type": "string", "enum": ["ready", "failed"]}},
            ("status",),
        ),
    }


def function_definitions() -> list[dict[str, Any]]:
    """Return the unique public name catalog, using Main semantics on overlap."""

    main = main_function_definitions()
    main_names = {item["function"]["name"] for item in main}
    return [
        *main,
        *[
            item
            for item in imagination_function_definitions()
            if item["function"]["name"] not in main_names
        ],
    ]


def main_function_definitions() -> list[dict[str, Any]]:
    within = {"type": "string", "description": "可选的当前 region 搜索范围"}
    point_within = {
        "type": "string",
        "description": "query 属于已检测对象或其局部时必须提供该对象的当前 region ID",
    }
    return [
        _function(
            "detection_and_sam",
            "在当前真实 observation 中执行 bbox detection + SAM，返回权威二维 region；不移动机器人，也不证明物理关系。同一 revision/query 会复用结果。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "为已有 region 生成 TOP、PCA 和 GraspNet 粗 ActionSeed；只提供候选，不执行、不保证抓取成功。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "locate_point",
            "在当前真实 observation 中定位语义操作点并提升为 robot-base XYZ；不生成方向、不移动机器人。它是可按可见几何微调的粗 metric anchor，可能位于容器边缘，不是保证最优的最终位姿。已存在目标 region 时必须用 within_region_id 限定。",
            {"query": {"type": "string"}, "within_region_id": point_within},
            ("query",),
        ),
        _function(
            "propose_pose",
            "从有效 point 创建或替换一个 Main 拥有的粗 Action 和 Preview；不执行，也不会自动启动 Imagination。",
            {
                "point_id": {"type": "string"},
                "offset_xyz": {"type": "array", "items": {"type": "number"}},
                "quaternion_xyzw": {"type": "array", "items": {"type": "number"}},
            },
            ("point_id", "offset_xyz"),
        ),
        _function(
            "select",
            "选择一个有效 ActionSeed，创建或替换 Main 拥有的粗 Action 和 Preview；不执行，也不会自动启动 Imagination。",
            {"seed_id": {"type": "string"}},
            ("seed_id",),
        ),
        _function(
            "refine_action",
            "把一个显式待执行空间 Action 委派给 Imagination SubAgent 做局部位置/方向优化；整个内部循环作为一次 Function 返回。有琥珀 carried-volume 时可优化物体—目标关系；没有时应把 instruction 写成目标夹爪—目标区域的对齐和安全净空，不得要求判断未来物体落点。抓取 instruction 应要求目标进入两指闭合扫掠区域且掌部/横梁/指根保持净空；不得要求指尖始终高于物体顶面或保持固定指尖—顶面距离。",
            {
                "action_id": {"type": "string"},
                "instruction": {
                    "type": "string",
                    "description": "一句局部几何任务：目标关系、需保持的条件和停止标准",
                },
            },
            ("action_id", "instruction"),
        ),
        _function(
            "delta_move",
            "Main 立即执行一次小幅 TCP 平移并刷新真实 observation；不是 Preview，不需要 commit。适合抬升/退让、恢复净空与视野或验证随动，可连续调用（每次每轴最多 3cm）；局部几何需要连续预览或旋转时使用 refine_action。",
            {
                "delta_xyz_m": {
                    "type": "array",
                    "items": {"type": "number", "minimum": -0.03, "maximum": 0.03},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "frame": {
                    "type": "string",
                    "enum": ["base", "tool"],
                    "description": "必须显式选择；base +Z 恒为世界上方，tool 轴随当前 TCP 旋转",
                },
            },
            ("delta_xyz_m", "frame"),
        ),
        _function(
            "open_gripper",
            "Main 立即打开真实夹爪并刷新 observation；不创建 Preview，不需要 commit，并使旧 evidence/action ID 失效。",
        ),
        _function(
            "close_gripper",
            "Main 立即闭合真实夹爪并刷新 observation；不创建 Preview，不需要 commit，使旧 evidence/action ID 失效，也不保证抓住物体。",
        ),
        _function(
            "reject_action",
            "放弃当前 pending action，不执行、不刷新 observation。",
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "commit",
            "执行一个由 refine_action 返回 ready 的空间 Action，随后刷新真实 observation。粗 proposal、夹爪命令和直接 delta_move 不能由它执行。",
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


def imagination_function_definitions() -> list[dict[str, Any]]:
    definitions = _edit_definitions()
    return [definitions[name] for name in IMAGINATION_FUNCTION_NAMES]


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
    "MAIN_FUNCTION_NAMES",
    "SYSTEM_PROMPT",
    "function_definitions",
    "imagination_function_definitions",
    "main_function_definitions",
    "parse_action",
]
