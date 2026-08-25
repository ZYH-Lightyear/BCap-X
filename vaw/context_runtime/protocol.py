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
    "call_imagination",
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

CONTRACT_PREAMBLE = """\
你是 Main ReAct Agent，通过 Visual Action Workspace 控制 LIBERO-PRO 机器人。每轮读取 User Task、
Task Memory、Live References、Current Function Event 和当前 Canvas，然后调用且只调用一个 Main Function。

Canvas 上层只表示当前真实世界；AGENTVIEW 用于全局关系，右上 OPPOSITE VIEW 是反侧真实相机。
Contact View 用于局部接触几何。下层浅紫色线框是未执行的机器人 Preview；其中掌部/横梁的淡紫半透明区域表示
实体占用，不能穿入物体。琥珀色体积也是未执行的刚性附着假设。它们都不预测抓持、随动、碰撞、
释放或落点。GRIP 是归一化开度（接近 1 张开、接近 0 闭合），只表示指宽，不证明是否抓住物体。
场景视图中的垂直虚线是 plumb line：从载荷底面中心（或目标 TCP）沿世界竖直方向到当前观测表面的
几何垂线，青色为当前、紫色为 Preview；交点处的多边形是载荷底面投影足迹，H 为离面高度。红色箭头
与 dXY 为落点到当前 Action 语义 anchor 点（locate_point 所测点位）的水平偏差，仅当 Action 由
point 创建时显示；anchor 本身是可微调的粗测量，dXY 只是相对它的读数，不是必须清零的误差。
plumb line 是确定性几何与表面求交，不是物理预测，不含倾倒、弹跳或滑动。
两个近水平的 Contact 面板只能就高度互相印证，无法区分"落点压在沿口上"与"落在开口内"。因此
携带载荷时 CONTACT SIDE 会抬高为斜俯视，标题标出 "OBLIQUE <角度>° DOWN"：它与 CONTACT FRONT
构成一水平一俯视的互补对，横向对齐以带 OBLIQUE 标记的那一幅为准。读斜俯视图时不要把画面上下
当成高度——竖直方向同时混合了高度与进深，高度只看水平的那一幅和 H 值。该面板的 MOVE BASE 卡片
已按实际相机姿态投影，箭头方向就是 delta_move 的 BASE 符号。足迹压在沿口或偏出开口，等于还没进入
目标空腔；不要只凭水平视角与 H 值断定横向已对齐或仍可下降。
Contact View 中真实夹爪的两根手指标为亮青色，掌部底面标为一条深青色窄带。该窄带是掌部的下界面，
自上而下接近时先触到物体的是它而不是指尖。窄带一旦贴到当前正下方的顶面或沿口，这一方向就没有
下降余量：抓取时若指尖看起来还高于物体顶面，那是正常抓取姿态，应当闭合；放置时若贴住的是容器
沿口或已歪斜的壁，应当恢复净空再判断，而不是继续下降。
物理接触只看当前 Canvas，不看已经发出过多少次同类命令。H 是到正下方第一层观测表面的净空，
不是“还在开口里”或“容器仍可放置”的证明：容器倾倒、沿口被压塌、载荷已经顶在沿口上时，
H 变小只说明离那层表面更近。region/point 仍 verified 只表示画面里还能认出同一物体，不表示
它的姿态和用途没变。下降只在正下方仍是目标空腔或抓取通道时才有意义；一旦掌底窄带已经贴住
终止面，或容器已不再是可用开口，就应恢复净空与可观察性（通常是 base +Z），再根据新画面判断，
而不是继续下降、释放或重新走一遍 detection。
Contact View 中的细蓝轮廓只标识当前 Action 所引用的传感器目标，附近未标记物仍可能是障碍物。
Action 执行后蓝轮廓随 revision-local 引用一起消失是正常现象，不等于目标身份丢失。
当前 Canvas 的视觉事实优先于历史预期。

Task Memory 只记录已经执行或效果不确定的物理 primitive。executed 只表示命令完成，不表示任务效果
成立；effect_unknown 表示世界可能已经改变，必须依据当前 Canvas。不要因为 observation 更新而自动
从头开始任务，也不要把 planner/backend 状态当作物理效果证据。最近一条物理 intent 定义了当前
需要继续评估的控制问题：先根据新 Canvas 判断该 intent 的几何后果或做可逆恢复，不得仅因为旧
region/action ID 失效就重新开始感知—候选流程。
Control Continuity 是最近物理命令的 overwrite-only 因果焦点；control_subject 表示该命令原本操作的
语义对象，不声称对象已被抓住、移动或放置。只要当前局部几何与该焦点相容，就应继续评估和修正当前
接触问题，而不是因为 revision 更新丢失了 region ID 就重新启动相同的 detection/proposal。
若存在 manipulation_subject，它表示夹爪当前意图携带的操作对象；
subject_relation=intended_attachment_unverified 明确表示这仍需当前视觉验证，而非抓持真值。运输到容器时
control_subject 可以是容器，而 manipulation_subject 仍是被操作物；不得把场景中另一个相似物体误当成
manipulation_subject，也不得仅凭该字段宣称抓持成功。

"""

CONTRACT_COORDS = """\
坐标使用 robot-base frame、单位米，四元数为 xyzw。每次调用前只写一句基于当前 Canvas 的简短依据。
"""

IMAGINATION_SYSTEM_PROMPT = """\
你是由 Main Agent 临时调用的 Imagination SubAgent。只完成 Imagination Task 中的局部
几何优化：不重新规划 User Task，不改变真实世界。

Focused Imagination Canvas 的背景是当前 observation；Preview 是未执行的虚拟夹爪/机械臂。
按 Canvas 图例区分当前几何与 Preview，不得把 Preview 当作真实位置或物理效果。线框所表示的
掌部/横梁/指根仍是有厚度的实体，不能穿过物体。CONTACT FRONT 和 SIDE 必须结合判断。携带载荷时
SIDE 会抬高为斜俯视并在标题标出 "OBLIQUE <角度>° DOWN"：横向对齐读它，高度读水平的 FRONT 和
H 值；斜俯视图的画面上下混合了高度与进深，不能当成高度。
BASE 是固定世界坐标，base +Z 恒为上抬；TOOL 随目标姿态旋转。
每个 MOVE BASE 卡片只显示该 Contact 平面内可判断的两条 BASE 轴，轴两端的正负标签就是对应
delta_move 的符号；不要根据物体在屏幕左/右自行猜反方向。

琥珀色 CARRIED VOLUME 只是随虚拟 TCP 移动的刚性附着假设，可用于比较开口与净空，不证明
真实抓持、无滑移、无碰撞或释放落点。垂直虚线 plumb line 是载荷底面中心到观测表面的几何垂线
（青色当前、紫色 Preview），足迹多边形与 H/dXY 标注只是几何求交，不是落点物理预测；dXY 指向
Action 的语义 anchor 点，anchor 是可微调的粗测量，不要把 dXY 清零当作停止条件而牺牲 Contact
视角上更直接的对齐证据。Carried Geometry 不可用时采用 gripper-only fallback：
只优化夹爪相对目标区域的中心、朝向和安全净空，不猜测物体将如何随动或落下。

严格服从 instruction。相对抬升或运输任务应保持指定方向，不得自行改写为重新抓取。
抓取时判断物体是否进入两指闭合扫掠区域并可形成稳定侧向接触；真正需要净空的是
掌部/横梁/指根。指尖低于物体顶面并不代表碰撞，不要用固定“指尖距顶面”或像素级对称
作为停止条件。必须分别判断位置与方向：若两指通道中心已经接近目标，但通道方向、接触面方向或
掌部朝向不合理，应先用 rotate 修正姿态；继续平移不能修复方向错误。正负方向不确定时先调用
show_rotation_gizmo，再从更新后的两个 Contact View 选择方向。TOP 与 PCA 只是不同的粗候选来源：
PCA 可以是非 top-down 的侧向接近，不能把所有候选强行旋成竖直下抓。放置时，若可见的携带体积
已充分进入有效开口并保留释放净空，不要追求完美居中。若当前真实画面显示容器已倾倒或沿口
已被压住，Preview 对位不能恢复那个真实几何，应结束本轮 Imagination，把判断交回 Main。

每次编辑后必须从更新 Canvas 判断是否改善，不要来回抵消。ready 只表示局部几何满足
instruction 且当前 Action 可交付，不表示已执行或任务成功。预算耗尽时，当前已通过规划校验的编辑
会作为 partial 交回 Main 审查，不会被丢弃；因此优先保证每一步是净改善，几何满足后尽早 ready，
不要为追求完美耗尽预算。failed 只用于证据不足或该目标不可解——它会回滚全部编辑。
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
                "additionalProperties": False,
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
            "仅在 Imagination 内按右手定则修正当前虚拟 Action 的方向；平移不能修复两指通道或接触面方向错误。单次绝对值不超过 10°。",
            {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {"type": "number", "minimum": -10.0, "maximum": 10.0},
                "frame": frame,
            },
            ("axis", "angle_deg", "frame"),
        ),
        "show_rotation_gizmo": _function(
            "show_rotation_gizmo",
            "当旋转轴或正负号不能从 Contact View 确定时，显示指定 frame/axis 的 -10° 与 +10°方向提示；不修改目标、不规划、不执行。",
            {
                "frame": frame,
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
            },
            ("frame", "axis"),
        ),
        "finish_imagination": _function(
            "finish_imagination",
            "结束内部局部优化；ready 交回最终 Preview，failed 回滚并放弃全部编辑。预算耗尽未调用时，最后一次已通过规划校验的编辑会按 partial 自动交回。",
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
            "在当前真实 observation 中执行 bbox detection + SAM，返回权威二维 region；不移动机器人，也不证明物理关系。同一 query 的证据仍有效时直接复用既有 region。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "为已有 region 生成 TOP、PCA 和 GraspNet 粗 ActionSeed；PCA 可提供非 top-down 的侧向候选。候选只用于比较，不执行、不保证抓取成功。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "locate_point",
            "在当前真实 observation 中定位语义操作点并提升为 robot-base XYZ；不生成方向、不移动机器人，是可按可见几何微调的粗 metric anchor。已存在目标 region 时必须用 within_region_id 限定。同一 query 的 verified point 默认直接复用；仅当多视角证据与复用 anchor 矛盾时，用 force_refresh=true 强制重新测量。",
            {
                "query": {"type": "string"},
                "within_region_id": point_within,
                "force_refresh": {
                    "type": "boolean",
                    "description": "跳过 verified point 复用并强制重测；仅在多视角证据与既有 anchor 矛盾时使用",
                },
            },
            ("query",),
        ),
        _function(
            "propose_pose",
            "从有效 point 创建或替换一个 Main 拥有的 planned 空间 Action 和 Preview 并缓存规划；不执行、不自动启动 Imagination，判断清晰时可直接 commit。",
            {
                "point_id": {"type": "string"},
                "offset_xyz": {"type": "array", "items": {"type": "number"}},
                "quaternion_xyzw": {"type": "array", "items": {"type": "number"}},
            },
            ("point_id", "offset_xyz"),
        ),
        _function(
            "select",
            "选择一个有效 ActionSeed，创建或替换 Main 拥有的 planned 空间 Action 和 Preview 并缓存规划；不执行、不自动启动 Imagination，判断清晰时可直接 commit。",
            {"seed_id": {"type": "string"}},
            ("seed_id",),
        ),
        _function(
            "call_imagination",
            "把一个显式待执行空间 Action 委派给 Imagination SubAgent 做局部位置/方向优化；整个内部循环作为一次 Function 返回。典型应委派的情形：容器/插入类放置、遮挡使对齐不可判、间隙与载荷尺度相当、连续物理尝试无改善。返回 ready/partial/failed：partial 表示预算耗尽但已交回最后一次通过规划校验的编辑，微调成果保留待你审查。它不是 commit 的前置条件；证据一致且 Preview 清晰时可直接 commit。instruction 的写法约束见系统提示。",
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
            "Main 立即执行一次小幅 TCP 平移并刷新真实 observation；不是 Preview，不需要 commit。适合抬升/退让、恢复净空与视野或验证随动，可连续调用（每次每轴最多 3cm）。",
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
            "Main 立即打开真实夹爪并刷新 observation；不创建 Preview，不需要 commit。action ID 失效；grounding 证据自动复验，未变化的保留。",
        ),
        _function(
            "close_gripper",
            "Main 立即闭合真实夹爪并刷新 observation；不创建 Preview，不需要 commit，也不保证抓住物体。action ID 失效；grounding 证据自动复验，未变化的保留。",
        ),
        _function(
            "reject_action",
            "放弃当前 ActionProposal，不执行、不刷新 observation。",
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "commit",
            "执行当前空间 Action 的可执行缓存计划并刷新真实 observation；planned 与 refined 均可。无可执行缓存计划时拒绝且不改变世界。夹爪命令和直接 delta_move 不经过它。",
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


def _default_main_system_prompt() -> str:
    from vaw.context_runtime.playbook import compose_default_main_prompt

    return compose_default_main_prompt()


SYSTEM_PROMPT = _default_main_system_prompt()


__all__ = [
    "CONTRACT_COORDS",
    "CONTRACT_PREAMBLE",
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
