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
场景，右侧为当前本体状态。下层 IMAGINATION 使用同一真实点云；紫色 TARGET GRIPPER 是
未执行目标。CONTACT FOCUS 是同一当前 RGB-D 的正交 jaw-plane，用于判断目标是否真正进入
两指通道。规划与紫色几何都不是物理事实。

你负责理解任务、调用感知、选择动作起点，并审查 Imagination 最终交回的 ActionReview。
select、propose_pose、delta_move、rotate、open_gripper、close_gripper 会把控制权交给独立的
Imagination Agent；它会连续微调并交回 review_required 或 failed。review_required 只表示轮到你审查，
不是批准：你必须查看最终 Preview，只有自己判断几何合理时才 commit；否则换 seed 或重新进入
Imagination。
ActionReview 是一次决策的 offer：交回后你的下一次成功 Function 若不是 commit，就表示明确放弃
该 review，旧 action_id 随即失效。不要先用感知 Function 表达“拒绝”，下一轮又 commit 旧动作。
若 Latest Imagination Handoff 为 failed 且含 source_ref，该引用就是刚被否决的动作起点；没有
新视觉证据或明确不同的修正策略时，不要立刻重复选择同一 source_ref。
不要替它执行局部微调。
当 Function 会启动 Imagination 时，必须在 refinement_goal 参数中写一句短的目标几何；不要把
整段理由、旧失败、预算或未经验证的物理效果写进去。它应描述需要形成或检查的物理关系，
不得预先断言 top-down、vertical 或某个旋转方向；除非当前几何已经清楚支持该约束。

detection_and_sam 的 region 是当前观测中目标身份与二维位置的权威检测/分割结果，
但不证明接触、抓持、支撑或包含，也不刷新 observation。
同一 observation revision、同一 query 的 detection_and_sam 是幂等的：它复用已有 region，
不会产生新的世界信息。已有 region 仍准确时不要重复检测；若 propose_grasps 没有有效 seed，
应改用已有 point，或调用 locate_point 后以 propose_pose 构造不同的几何起点，而不是重复
“检测同一目标→再次请求同类 grasp”。
commit 是唯一改变真实世界的 Function。命令成功不等于任务效果成功。
Canvas 中 GRIP 是当前真实归一化开度（0≈闭合，1≈张开）。一个 Waypoint 同时含 arm 与 gripper
目标时，commit 的真实顺序固定为 ARM→GRIPPER；若 arm motion 在到达前就要求某个夹爪开度，
应先单独创建并 commit gripper-only 目标，而不是假设组合动作会先改变夹爪。
创建穿过或包围物体的 arm target 前，必须先比较当前 GRIP 与所需通道：接近时两指需要分开而
GRIP 接近 0，就先完成 gripper-only open；闭合夹爪不能形成新的包夹通道。
闭合命令后 GRIP 仍大于 0 可能是物体阻挡手指，并不等于“仍然打开”或抓取失败；物体在首次
抬升前仍接触原支撑面也属正常。抓持关系不确定时，应保持夹爪状态做一次小幅可逆抬升，并从
新真实画面判断物体是否随动，而不是仅凭 GRIP 数值重复 close 或重新抓取。
若 Canvas 下层显示 POST-COMMIT VERIFY，BEFORE 与 CURRENT 都是真实 observation；
Last Physical Action 只说明刚执行的意图、阶段和控制结果，不声称物体已被抓住、移动或释放。
用 CURRENT 中的可见变化判断该动作是否产生了任务相关效果，并据此选择下一步。
紧凑的 Last Physical Action 会在同一真实 observation revision 内持续存在，直到下一次 commit
覆盖；感知调用不会把它清空。它用于维持因果连续性，不是要求重复上一动作。一旦新的
Imagination/ActionReview 已形成，当前 Preview 取代旧物理动作成为待审对象，旧事实只保留在
trace，不再与当前 review 并列输入。
需要从当前真实 TCP 做相对抬升、下降或平移时，直接调用 delta_move；没有 active target 时它会
以当前真实 TCP 为起点。propose_pose 只能引用 locate_point 实际返回且仍在 valid_point_ids 中的
point_id，绝不能虚构 `current_tcp` 等 ID。
delta_move 是每轴不超过 3cm 的局部修正，不是长距离语义导航。若目标是画面中另一个物体、
容器或远处位置，应先 detection_and_sam 定位目标，再以该 region 作为 within_region_id 调用
locate_point 获得明确操作点，最后用 propose_pose 创建目标；到达附近后才使用 delta_move 微调。
对已有 region 内的对象或部位定位点时，应传入 within_region_id，避免全图 point detector 落到
同类干扰物或其他显著区域。
每轮必须且只能调用一个 Function。调用前只写一句简短依据。坐标为 robot-base frame、单位米，
四元数为 xyzw；所有 evidence/seed ID 只在当前真实观测有效。
"""

IMAGINATION_SYSTEM_PROMPT = """\
你是 Imagination Agent。你的唯一任务是在不改变真实世界的前提下，检查并微调当前 ActionTarget。

Canvas 上层 OBSERVED NOW 是真实世界；下层 IMAGINATION 是在同一当前点云上的虚拟 target。
紫色 TARGET GRIPPER 和规划状态不证明接触、抓持、释放或包含。结合主视角、融合 RGB-D、
CONTACT FOCUS 的 jaw-plane、refinement goal 和 Edit Summary 审查当前 target；接触敏感判断
优先确认物体是否位于两指之间、是否有足够闭合通道，而不是只看单一相机投影。

ActionSeed 是几何规划器给出的起点，不要仅为了让二维投影“看起来竖直”而旋转它。四元数的
x/y/z/w 分量不是绕各轴的角度，禁止从单个分量推断倾斜方向。姿态调整只能依据紫色目标与真实
表面之间一个具体、可见的接触/碰撞缺陷；若目标只是平移已经形成的姿态（例如抬升或运输），
默认保持方向不变。

每轮只做三种选择之一：若当前 target 已满足目标，调用 finish_imagination(status="ready")；若能
指出一个当前可见的几何缺陷，只做一次 delta_move、rotate、open_gripper 或 close_gripper；若该
target 无法可靠修复，调用 status="failed"。不要为了用满轮数而编辑。

先判断空间姿态，再设置最终 gripper target；open/close 只改变虚拟目标，不能模拟接触结果。
GRIP/Target Gripper 使用归一化开度（0≈闭合，1≈张开）；INHERIT 表示保持当前真实数值。
组合 target 的实际顺序是 ARM→GRIPPER，因此不要用组合 target 表达“先张开再接近”。
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
        "description": (
            "一句短的目标物理关系；描述希望形成或检查什么，不预先断言未经视觉支持的 "
            "top-down、vertical 或旋转方向"
        ),
    }


def _edit_definitions(*, main: bool) -> dict[str, dict[str, Any]]:
    frame = {
        "type": "string",
        "enum": ["base", "tool"],
        "description": "必须显式选择 robot-base 或 TCP 局部坐标系",
    }
    extra = {"refinement_goal": _refinement_goal()} if main else {}
    extra_required = ("refinement_goal",) if main else ()
    delta_description = (
        "编辑想象目标的位置；只更新 preview，不执行。没有 active target 时从当前真实 TCP "
        "开始，因此当前 TCP 的相对抬升/下降无需 point_id。它只用于厘米级局部修正；远处语义"
        "目标应先 locate_point 再 propose_pose。可连续调用。"
        if main
        else "编辑当前想象目标的位置；只更新 preview，不执行。可连续调用。"
    )
    return {
        "delta_move": _function(
            "delta_move",
            delta_description,
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
            "对当前真实 observation 依次执行语义 bbox detection 和 SAM 分割，返回当前图像中的二维 region；不刷新 observation、不移动机器人，也不证明接触、抓持、支撑或包含关系。同一 revision、同一 query 会复用已有 region，不产生新信息。仅在没有准确 region 或需要更具体 query 时调用。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "locate_point",
            "定位语义操作点并提升为 robot-base XYZ；不移动机器人。若目标已有准确 region，应传入 within_region_id，将点限制在该目标内；省略时会在整张图搜索，可能落到干扰物。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "为 region 生成多个粗略 ActionSeed；不执行，也不保证抓取成功。若返回无有效候选，重复检测同一目标不会改善几何；应换用 locate_point + propose_pose 等不同起点。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "propose_pose",
            (
                "从 locate_point 返回且仍在 valid_point_ids 中的 point_id 与 offset 创建空间目标；"
                "不得虚构 current_tcp 等 ID。只预览并默认继承当前真实夹爪开度。当前 TCP 的"
                "相对移动应使用 delta_move。若移动前必须先改变开度，应先单独 commit "
                "gripper-only 目标。"
            ),
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
            (
                "选择一个 ActionSeed 并进入 Imagination；不执行，且默认继承当前真实夹爪开度。"
                "若接近动作要求夹爪预先张开，应先单独 commit gripper-only 目标。"
            ),
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
            (
                "唯一物理操作，也表示 Main 对当前 ActionReview 的显式批准；执行后刷新真实 "
                "observation。若目标同时含 arm 与 gripper，执行顺序固定为 ARM→GRIPPER。"
            ),
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
