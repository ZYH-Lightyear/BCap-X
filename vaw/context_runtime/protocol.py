"""Agent-visible contracts for Main ReAct and its Imagination sub-agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

ROBOT_FUNCTION_NAMES = (
    "detect_region",
    "propose_grasps",
    "locate_point",
    "preview_pose",
    "preview_grasp",
    "imagine_action",
    "move_tcp_delta",
    "open_gripper",
    "close_gripper",
    "discard_action",
    "execute_action",
    "finish_task",
)

MAIN_FUNCTION_NAMES = ROBOT_FUNCTION_NAMES

IMAGINATION_FUNCTION_NAMES = (
    "shift_preview",
    "rotate_preview",
    "inspect_rotation",
    "finish_imagination",
)

FUNCTION_NAMES = tuple(dict.fromkeys((*MAIN_FUNCTION_NAMES, *IMAGINATION_FUNCTION_NAMES)))

# CONTRACT_PREAMBLE = """\
# 你是通过 Visual Action Workspace 控制 LIBERO-PRO 机器人的主智能体。运用你的视觉理解、物理常识
# 和提供的 Functions 完成用户任务。每轮先用最新画布判断当前真实状态，再调用且只调用一个 Function。
# 优先采取能够直接推进任务的最小充分动作；已有证据足够时不要重复检测、定位或想象。

# # 画布左上 NOW 是最新主相机全局视图，右上 AUXILIARY 在有局部目标时提供更陡的斜俯视；下方水平的
# # CONTACT FRONT/SIDE 是同一时刻的局部正交视图。你可以通过CONTACT
# # 判断夹爪、接触、净空和局部偏差；AUXILIARY 只补充观察手部遮挡后的夹爪、载荷和容器关系;
# # 琥珀色点状区域是当前检测区域，浅紫色机器人或载荷是尚未执行的预览，二者都不能取代真实视觉判断。
# # 黄色夹取线连接真实两指中部，表示闭合时扫过的接触通道。正确目标位于两指之间且与该通道相交时，
# # 就已有直接闭合依据；下一步应优先闭合，不要为追求“更深、更稳”、点坐标重合或像素级居中继续下压。

# # detect_region 用于建立目标区域；其 query 和区域引用不是物体身份真值。locate_point 只返回粗略空间锚点，
# # 不能确认目标身份、精确接触点、抓取或闭合条件，也不要求 TCP 与该点重合；不要用 point 坐标覆盖清楚的视觉证据。身份、接触和动作效果始终以
# # 画面中有多个同类物体时，检测 query 应包含 NOW 中可见的外观或空间区别；检测返回后先核对区域裁剪与
# # 全局目标是否一致，不能仅因 Function 返回了 region_id 就继续抓取。不得从任务名称臆测物体颜色、形状
# # 或位置；只有 NOW 中直接可见的文字、图案、颜色和空间关系才能作为区分属性，看不清时保留任务原词。
# # 物体被推动、倾倒或掉落后，姿态和几何投影会改变，但可见外观仍是身份依据；抓取候选、尺寸提示、IK 或
# # 可达性结果只描述动作几何，不能据此否定视觉身份并换成另一个同类物体。
# # 抓取或规划失败只否定本次动作，不否定目标身份；原 region 仍为有效引用且最新视觉没有矛盾时，应围绕
# # 同一目标更换候选或修正动作，只有证据失效或视觉确实不符时才重新检测。

# # 抓取时先确认正确目标，再以张开的夹爪接近。TOP/PCA/CGN 只是候选来源而非优先级；应从可见预览选择能在
# # 物体稳定部位形成双侧接触、接近质心且掌部有净空的方案，避免只夹边缘、角点或一侧。目标进入两指接触通道后及时闭合；闭合后的非零开度可能
# # 表示物体正在阻挡两指，是接触而非失败。执行抓取预览前还要检查掌部、横梁和指根对目标及支撑面的净空；
# # 若虚拟实体已经穿入物体、压住支撑面或只能形成明显单侧接触，应先修正预览而不是直接执行。随后做短距离
# # 上抬，并依据物体是否随夹爪离开支撑面验证抓持。
# # 只有最新视图清楚显示物体已离开原支撑且仍在两指间，才算抓持成立；遮挡或“似乎随动”时继续可逆短抬确认。
# # 抓持成立后再搬运；失败则依据最新画面恢复。放置时依据被抓物相对容器内部与边沿的真实视觉关系对齐，
# # 确认物体具有进入空间和释放净空后打开夹爪，再退开遮挡并验证结果。
# # 但每个物理动作后必须重新观察。交互记忆只证明调用发生过，不证明其语义效果；不要让历史意图覆盖当前画面。
# # 只有当前画布确实难以判断局部几何、姿态或净空时才调用 imagine_action。
# """

CONTRACT_PREAMBLE = """\
你是通过 Visual Action Workspace 控制 LIBERO-PRO 机器人的主智能体。运用你的视觉理解、物理常识
和提供的 Functions 完成用户任务。每轮先用最新画布判断当前真实状态，再调用且只调用一个 Function。
优先采取能够直接推进任务的最小充分动作；已有证据足够时不要重复检测、定位或想象。

画布左上 NOW 是最新主相机全局视图，右上为俯视视图, 下方为围绕夹爪的水平视角视图; 这些视图在视觉上都可能存在遮挡, 你需要根据“不遮挡的视角”做出你的判断;
"""

CONTRACT_COORDS = """\
坐标使用机器人基座坐标系、单位为米，四元数为机械手绝对方向的 xyzw；它不是相对旋转。
不需要改变方向时省略 quaternion_xyzw。
"""

WorldEffect = Literal["none", "physical"]


@dataclass(frozen=True)
class FunctionSpec:
    """Internal Function metadata; only ``definition`` is sent to a model."""

    definition: dict[str, Any]
    world_effect: WorldEffect = "none"
    effect_channel: str | None = None

    @property
    def name(self) -> str:
        return str(self.definition["function"]["name"])


class FunctionRegistry:
    """One source of truth for tool schemas and runtime effect metadata."""

    def __init__(self, specs: tuple[FunctionSpec, ...]) -> None:
        self.specs = specs
        self._by_name = {spec.name: spec for spec in specs}

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition for spec in self.specs]

    def get(self, name: str) -> FunctionSpec | None:
        return self._by_name.get(name)

IMAGINATION_SYSTEM_PROMPT = """\
你是由主智能体临时调用的动作想象子智能体。只完成本轮想象任务中的局部几何优化：
不重新规划用户任务，也不改变真实世界。

局部想象画布的背景来自当前观测；动作预览是尚未执行的虚拟夹爪和机械臂。
按画布图例区分当前几何与动作预览，不得把预览当作真实位置或物理效果。线框所表示的
掌部、横梁和指根仍是有厚度的实体，不能穿过物体。必须结合水平的“CONTACT FRONT”和
“CONTACT SIDE”判断。
“BASE”是固定世界坐标，基座 +Z 恒为上抬；“TOOL”坐标随目标姿态旋转。
每个“MOVE BASE”卡片只显示该接触平面内可判断的两条基座轴，轴两端的正负标签就是对应
shift_preview 的符号；不要根据物体在屏幕左侧或右侧自行猜测方向。卡片和图例只用于读取符号，
不是需要对齐或验收的对象。

琥珀色“CARRIED VOLUME”只是随虚拟 TCP 移动的刚性附着假设，可用于比较开口与净空，不证明
真实抓持、无滑移、无碰撞或释放落点。垂直虚线是载荷底面中心到观测表面的几何垂线
（青色当前、紫色预览），底面投影、垂线与红色方向箭头只是几何求交，不是落点物理预测；箭头
指向动作的语义锚点，锚点是可微调的粗测量，不要为追求像素级重合而牺牲接触视图上
更直接的对齐证据。携带物几何不可用时采用仅夹爪回退：
只优化夹爪相对目标区域的中心、朝向和安全净空，不猜测物体将如何随动或落下。

严格服从 instruction 中的场景目标。面板标题、垂线、投影和“MOVE BASE”卡片只用来选择方向符号，
不是要满足的对象。相对抬升或运输任务应保持指定方向，不得自行改写为重新抓取。
抓取时判断物体是否进入两指闭合扫掠区域并可形成稳定侧向接触；真正需要净空的是
掌部/横梁/指根。指尖低于物体顶面并不代表碰撞，不要用固定“指尖距顶面”或像素级对称
作为停止条件。必须分别判断位置与方向：若两指通道中心已经接近目标，但通道方向、接触面方向或
掌部朝向不合理，应先用 rotate_preview 修正姿态；继续平移不能修复方向错误。正负方向不确定时先调用
inspect_rotation，再从更新后的两个接触视图选择方向。TOP 与 PCA 只是不同的粗候选来源：
PCA 可以是非垂直下抓的侧向接近，不能把所有候选强行旋成竖直下抓。放置时，若可见的携带体积
已充分进入有效开口并保留释放净空，不要追求完美居中。若当前真实画面显示容器已倾倒或沿口
已被压住，动作预览对位不能恢复真实几何，应结束本轮想象，把判断交回主智能体。

每次编辑后必须从更新画布判断是否改善，不要来回抵消。ready 只表示局部几何满足 instruction
且当前动作可交付，不表示已执行或任务成功。预算耗尽时，当前已通过规划校验的编辑会作为 partial
交回主智能体审查，不会被丢弃；因此优先保证每一步是净改善，几何满足后尽早 ready，
不要为追求完美耗尽预算。failed 只用于证据不足或该目标不可解，它会回滚全部编辑。
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
        "shift_preview": _function(
            "shift_preview",
            "平移虚拟动作预览，不执行真实动作。",
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
        "rotate_preview": _function(
            "rotate_preview",
            "按右手定则旋转虚拟动作预览，不执行真实动作。",
            {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {"type": "number", "minimum": -10.0, "maximum": 10.0},
                "frame": frame,
            },
            ("axis", "angle_deg", "frame"),
        ),
        "inspect_rotation": _function(
            "inspect_rotation",
            "显示指定坐标系和轴的正负旋转方向，不修改动作预览。",
            {
                "frame": frame,
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
            },
            ("frame", "axis"),
        ),
        "finish_imagination": _function(
            "finish_imagination",
            "结束想象：ready 交回当前预览，failed 放弃本轮编辑。",
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


def _main_tool_definitions() -> list[dict[str, Any]]:
    within = {"type": "string", "description": "可选：限定在该区域内搜索"}
    point_within = {
        "type": "string",
        "description": "可选：限定在已检测区域内估计点；不会复验该区域的物体身份",
    }
    robot_definitions = [
        _function(
            "detect_region",
            "在最新图像中检测并分割目标，返回区域引用；不移动机器人。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "为目标区域生成多个抓取候选；只生成候选，不执行。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "locate_point",
            "估计粗略三维锚点；不能确认目标身份、精确接触点或闭合条件。",
            {
                "query": {
                    "type": "string",
                    "description": "要寻找的可见位置描述；返回值只是粗略锚点",
                },
                "within_region_id": point_within,
                "force_refresh": {
                    "type": "boolean",
                    "description": "忽略点缓存并重新估计；不会重新检测或确认区域身份",
                },
            },
            ("query",),
        ),
        _function(
            "preview_pose",
            "从三维点和可选绝对姿态生成动作预览与运动计划；",
            {
                "point_id": {"type": "string"},
                "offset_xyz_m": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "基座坐标系下的 xyz 偏移，单位米",
                },
                "quaternion_xyzw": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "可选的基座系绝对手部姿态 xyzw；省略则保持当前姿态。向下常用 [1,0,0,0] 或 [0,1,0,0]；[0,0,0,1] 朝上。",
                },
            },
            ("point_id", "offset_xyz_m"),
        ),
        _function(
            "preview_grasp",
            "把一个抓取候选转换为动作预览与运动计划；",
            {"seed_id": {"type": "string"}},
            ("seed_id",),
        ),
        _function(
            "imagine_action",
            "让想象智能体检查并微调局部空间动作；如果你通过preview_pose得到了一个预备姿态, 那么你可以调用这个函数, 启动一个想象智能体帮你微调姿态, 直到抓住物体 / 移动到你想要的位置; 调用完成后, Canvas返回的紫色mask就是想象结果",
            {
                "action_id": {
                    "type": "string",
                    "description": "可选动作 ID；省略表示从当前真实 TCP 开始",
                },
                "instruction": {
                    "type": "string",
                    "description": "局部几何目标与停止条件",
                },
            },
            ("instruction",),
        ),
        _function(
            "move_tcp_delta",
            "立即执行一次小幅 TCP 平移并刷新观测。",
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
                    "description": "base 为固定基座系；tool 随当前 TCP 旋转",
                },
            },
            ("delta_xyz_m", "frame"),
        ),
        _function(
            "open_gripper",
            "立即打开真实夹爪。",
        ),
        _function(
            "close_gripper",
            "立即闭合真实夹爪；闭合不等于抓取成功。",
        ),
        _function(
            "discard_action",
            "放弃当前动作预览；不执行。",
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "execute_action",
            "执行指定动作的已验证运动计划并刷新观测。",
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "finish_task",
            "结束任务并报告当前判断；success 不是环境真值。",
            {"success": {"type": "boolean"}},
            ("success",),
        ),
    ]
    return robot_definitions


def main_function_specs() -> tuple[FunctionSpec, ...]:
    """Attach runtime effects without leaking non-standard fields to providers."""

    physical = {
        "move_tcp_delta": "arm",
        "execute_action": "arm",
        "open_gripper": "gripper",
        "close_gripper": "gripper",
    }
    return tuple(
        FunctionSpec(
            definition=definition,
            world_effect=("physical" if name in physical else "none"),
            effect_channel=physical.get(name),
        )
        for definition in _main_tool_definitions()
        for name in (str(definition["function"]["name"]),)
    )


def main_function_registry() -> FunctionRegistry:
    return FunctionRegistry(main_function_specs())


def main_function_definitions() -> list[dict[str, Any]]:
    return main_function_registry().definitions


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
            raise ValueError(f"动作不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("action must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or name not in allowed:
        raise ValueError(f"未知函数“{name}”")
    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    return name, arguments


SYSTEM_PROMPT = CONTRACT_PREAMBLE + CONTRACT_COORDS


__all__ = [
    "CONTRACT_COORDS",
    "CONTRACT_PREAMBLE",
    "FUNCTION_NAMES",
    "FunctionRegistry",
    "FunctionSpec",
    "IMAGINATION_FUNCTION_NAMES",
    "IMAGINATION_SYSTEM_PROMPT",
    "MAIN_FUNCTION_NAMES",
    "ROBOT_FUNCTION_NAMES",
    "SYSTEM_PROMPT",
    "function_definitions",
    "imagination_function_definitions",
    "main_function_definitions",
    "main_function_registry",
    "main_function_specs",
    "parse_action",
]
