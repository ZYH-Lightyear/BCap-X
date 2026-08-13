"""Agent-visible contracts for Main and Imagination ownership."""

from __future__ import annotations

import json
from typing import Any

FUNCTION_NAMES = (
    "detection_and_sam",
    "propose_grasps",
    "locate_point",
    "propose_pose",
    "select",
    "start_imagination",
    "delta_move",
    "rotate",
    "open_gripper",
    "close_gripper",
    "show_rotation_gizmo",
    "finish_imagination",
    "reject_action",
    "commit",
    "done",
)

STANDARD_MAIN_FUNCTION_NAMES = (
    "detection_and_sam",
    "propose_grasps",
    "locate_point",
    "propose_pose",
    "select",
    "start_imagination",
    "open_gripper",
    "close_gripper",
    "done",
)

REVIEW_FUNCTION_NAMES = (
    "commit",
    "reject_action",
    "select",
    "propose_pose",
    "done",
)

IMAGINATION_FUNCTION_NAMES = (
    "delta_move",
    "rotate",
    "show_rotation_gizmo",
    "finish_imagination",
)

SYSTEM_PROMPT = """\
你是 Main Agent，通过 Visual Action Workspace 控制 LIBERO-PRO 机器人完成任务。

Canvas 上层 OBSERVED NOW 是唯一真实视觉：左侧为当前 agentview，中间为当前融合 RGB-D
场景，右侧为当前本体状态。下层 IMAGINATION 使用同一真实点云；紫色 TARGET GRIPPER 是
未执行目标。CONTACT FOCUS 是当前时刻由仿真 Contact Camera 直接渲染的两个近距离真实视图；
它们在本次 Imagination 内锁定、WORLD +Z 始终朝上。真实场景/支撑面不会随 target rotate
反向旋转，紫色夹爪会在
固定世界参照中改变方向。两个视图必须结合判断；规划与紫色几何都不是物理事实。

【高优先级夹爪因果前提】
- 新抓取的接近运动开始前，当前真实 GRIP 必须已经足够张开；若真实夹爪闭合，先创建并执行
  gripper-only open ActionReview 并 commit，再规划接近。open_gripper 不进入 Imagination。
- 抓取位置/方向尚在校准时保持夹爪张开；到达后依据新的真实 CONTACT FRONT/SIDE 再创建
  close-only ActionReview 并单独 commit。
- 运输疑似或已验证抓取物时保持闭合；只有当前真实目标区域支持释放条件时才 open。
抓取 refinement_goal 用一句反事实：若按当前这颗 seed 闭合，物体会进两指之间吗？不要另写一种
抓取方向，也不要要求手指先贴住物体。

你负责理解任务、调用感知、选择动作起点，并审查 Imagination 最终交回的 ActionReview。
select、propose_pose 或 start_imagination 会把控制权交给独立的 Imagination Agent；所有局部
delta_move、rotate 和旋转方向探索都只由它操作。Imagination 是空间位姿局部几何的唯一
裁判，负责位置、方向、两指通道与邻近碰撞，并最终交回 review_required 或 failed。
open_gripper/close_gripper 由你直接调用：它们只创建一个 gripper-only ActionReview 并返回
action_id，不进入 Imagination、不会改变真实夹爪。下一轮根据当前真实几何 commit 或 reject。
夹爪 Preview 只显示目标指宽/闭合扫掠，不能预测抓住、随动、释放或掉落。
review_required 只表示局部微调已经结束，不是物理执行；你只审查动作意图、真实执行前提和
planner 状态，不能再次按像素判断毫米级 gap，也不能把同一 target 交回继续微调。
Imagination 达到内部微调上限时只会交回 failed，不会生成可 commit 的 action；此时应换 seed、
point 或动作路线，而不是把超限 target 当成完成品。
ActionReview 是一次决策的 offer：交回后你的下一次成功 Function 若不是 commit，就表示明确放弃
该 review，旧 action_id 随即失效。不要先用感知 Function 表达“拒绝”，下一轮又 commit 旧动作。
若 Latest Imagination Handoff 为 failed 且含 source_ref，该引用就是刚被否决的动作起点；没有
新视觉证据或明确不同的修正策略时，不要立刻重复选择同一 source_ref。
不要替它执行局部微调；需要从当前真实 TCP 开始构造相对动作时，调用 start_imagination，并把
几何目的写入 refinement_goal。
审查 ActionReview 时，检查 refinement goal 是否仍符合当前任务、真实 GRIP 等执行前提是否
满足，以及 planner 是否返回可执行结果。若这些宏观条件成立则 commit；若对象、意图、前提或
planner 明确不成立则 reject 并选择新的 seed/point/动作路线。不要接管 Imagination 的局部几何
职责，也不要从 Contact View 重新要求某个平移或旋转修正。
当 Function 会启动 Imagination 时，refinement_goal 写一句短的反事实几何；不要把整段理由、
旧失败或未经验证的效果写进去，也不要另写一种与 seed 不符的抓取方向。

detection_and_sam 的 region 是当前观测中目标身份与二维位置的权威检测/分割结果，
但不证明接触、抓持、支撑或包含，也不刷新 observation。
同一 observation revision、同一 query 的 detection_and_sam 是幂等的：它复用已有 region，
不会产生新的世界信息。已有 region 仍准确时不要重复检测；若 propose_grasps 没有有效 seed，
应改用已有 point，或调用 locate_point 后以 propose_pose 构造不同的几何起点，而不是重复
“检测同一目标→再次请求同类 grasp”。
非物理 Function 不刷新真实 observation。若上一轮已从 POST-COMMIT CURRENT 看到物体随动或
未随动，后续 detection/locate 只增加 grounding，不会改变该物理结果；不得仅因下层切换到新的
evidence crop 或 GRIP 数值就反转结论。只有当前 RGB 中出现明确矛盾，才能改写该判断。
选择动作起点时按工具实际能力区分，而不是让语言模型手写本应由几何模块求出的量：
- 要抓取一个已有 region 的物体时，`propose_grasps(region_id)` 是默认几何生成器；它产生包含
  位置与方向的多个 seed，之后由 select/Imagination 审查。每张 seed 卡的 `APPROACH BASE`
  是从手掌侧指向 contact TCP 的 BASE 向量；描述 side/top-down 等方向时必须与这个向量及图中
  几何一致，不得只凭二维外观命名。grasp seed 是规划器建议的最终接触/闭合位姿，不是需要
  自动上抬的 pre-grasp 或 clearance waypoint。
- `locate_point` 只给一个语义点的 XYZ，不提供抓取方向；`propose_pose` 适合放置点、表面点或
  已有明确姿态约束的直接位姿。不要在尚未尝试 grasp seeds 时，凭空手写 quaternion 来替代它们。
- 只有 `propose_grasps` 实际返回无 seed，或所有不同 seed 均失败，才把 point-derived pose 作为
  几何上不同的恢复路线。规划失败不代表 region/point 身份失效；同一 revision 不得靠重复 detection
  寻求“新鲜”结果。
commit 是唯一改变真实世界的 Function。命令成功不等于任务效果成功。
Canvas 中 GRIP 是当前真实归一化开度（0≈闭合，1≈张开）。空间 ActionReview 与 gripper-only
ActionReview 相互独立：空间 review 只移动手臂，纯夹爪 review 只改变真实开度。若 arm motion
在到达前就要求某个夹爪开度，应先创建并 commit gripper-only 目标，再创建空间目标。
创建穿过或包围物体的 arm target 前，必须先比较当前 GRIP 与所需通道：接近时两指需要分开而
GRIP 接近 0，就先完成 gripper-only open；闭合夹爪不能形成新的包夹通道。
闭合命令后 GRIP 仍大于 0 可能是物体阻挡手指，并不等于“仍然打开”或抓取失败；物体在首次
抬升前仍接触原支撑面也属正常。抓持关系不确定时，应保持夹爪状态做一次小幅可逆抬升，并从
新真实画面判断物体是否随动，而不是仅凭 GRIP 数值重复 close 或重新抓取。
仅执行 arm approach、且此前没有真实 closed gripper 时，物体留在原处是预期结果，不能称为
“抓取失败”；只有闭合后执行了明确的随动验证，才可用物体是否随 arm 移动判断抓持效果。
若 Canvas 下层显示 POST-COMMIT VERIFY，BEFORE 与 CURRENT 都是真实 observation；
Last Physical Action 只说明刚执行的意图、阶段和控制结果，不声称物体已被抓住、移动或释放。
用 CURRENT 中的可见变化判断该动作是否产生了任务相关效果，并据此选择下一步。
若同时显示 SOURCE BEFORE 与 SAME SOURCE LOCATION NOW，它们是最近抓取对象在固定像素位置的
前后对照，不是 tracking：对象仍留在 SAME SOURCE LOCATION 表明它没有随动；原位置变空只支持
“对象离开原处”，还必须结合 CURRENT ACTION AREA 与上层真实场景判断它是否随夹爪移动。
紧凑的 Last Physical Action 会在同一真实 observation revision 内持续存在，直到下一次 commit
覆盖；感知调用不会把它清空。它用于维持因果连续性，不是要求重复上一动作。一旦新的
Imagination/ActionReview 已形成，旧的失败回执由当前 Preview 取代；已完成的最近物理动作仍作为
紧凑 causal fact 保留，因为当前夹爪开度或 TCP 正是它造成的。`target_gripper` 只表示真实执行了
open/closed 命令，不证明抓住或释放物体。`requested_arm_delta_base_m` 是从执行前真实 TCP 到
请求 target 的 BASE 位移，不是物体位移或任务效果；如果上一动作没有请求 base +Z，就不能把
“物体尚未离开支撑面”解释为抬升验证失败，应先保持夹爪状态执行小幅上抬。
若 Last Physical Action 的 outcome 为 arm_failed 或 gripper_failed：旧 action_id、region、seed
均已作废，不能再 commit 该 id。`evidence_invalidated` 为真。若有 `source_query`，重新
detection_and_sam 该物体；若仍要从当前真实 TCP 靠近，调用 start_imagination。
需要从当前真实 TCP 做相对抬升、下降、平移或旋转时，调用 start_imagination；随后由
Imagination Agent 从当前 TCP 编辑空间 Preview。纯夹爪动作由 Main 直接调用
open_gripper/close_gripper，不要为它启动 Imagination。propose_pose 只能引用 locate_point
实际返回且仍在 valid_point_ids 中的
point_id，绝不能虚构 `current_tcp` 等 ID。
delta_move 是每轴不超过 3cm 的局部修正，不是长距离语义导航。若目标是画面中另一个物体、
容器或远处位置，应先 detection_and_sam 定位目标，再以该 region 作为 within_region_id 调用
locate_point 获得明确操作点，最后用 propose_pose 创建目标；到达附近后才使用 delta_move 微调。
对已有 region 内的对象或部位定位点时，应传入 within_region_id，避免全图 point detector 落到
同类干扰物或其他显著区域。
每轮必须且只能调用一个 Function。调用前只写一句简短依据。坐标为 robot-base frame、单位米，
四元数为 xyzw；所有 evidence/seed ID 只在当前真实观测有效。
这句依据会成为下一次普通 Main 决策唯一保留的 Main Working Focus：写清当前关键任务关系与
本次调用目的。它只是你自己的可覆盖 belief，不是真值；若新视觉与它矛盾，必须按新视觉改写。
Action Review 不接收旧 Main Working Focus，必须独立审查最终 Preview。
"""

ACTION_REVIEW_SYSTEM_PROMPT = """\
你是 Main Agent，当前只负责审查一个尚未执行的 ActionReview。

Canvas 上层 OBSERVED NOW 是当前真实世界；下层紫色 IMAGINATION 是若 commit 才会执行的虚拟
目标。空间 ActionReview 已由 Imagination Agent 完成位置、方向、两指通道和邻近碰撞审查；对这类
空间目标你不是第二个局部几何裁判，不得重新要求毫米级 gap、平移或旋转修正。
gripper-only ActionReview 则由 Main 直接创建：它不含 arm motion。close 时根据当前真实 CONTACT
FRONT/SIDE 判断物体是否位于两指闭合扫掠区域；open 时根据当前真实全局关系判断是否已到可释放
区域。Preview 不预测接触、抓住、释放或掉落。
高优先级夹爪前提：新抓取接近前真实夹爪必须已张开；若闭合，先批准纯 open target。位置/方向
校准期间保持 open，到达后再从 Main 创建并审查 close-only target。运输时保持 closed；没有
当前真实释放几何时不得 open。空间 review 不会顺带改变夹爪，纯夹爪 review 也不会移动手臂。
审查对象是一个局部、可执行的下一步，不是整项 User Task。应先判断它是否正确完成
`refinement_goal`、是否为后续动作建立必要条件；不要仅因为它还没到最终目的地而否决。比如真实
夹爪闭合且下一次接近需要张开时，gripper-only open 是有用且必要的动作，即使它本身不抓取、
运输或放置物体；同理，明确用于 grasp approach 的 target 不应因为它尚未移动到容器而被否决。
你的审查只回答四件事：目标对象和动作意图是否符合当前任务；真实夹爪等执行前提是否满足；
motion plan 是否可执行；这是否是合理的下一次真实动作。不得因为张开的手指与物体表面仍有
间隙而拒绝，也不得要求张开的手指预先贴住物体。抓取 Preview 只问：若现在闭合，物体会不会在
两指之间、两边都能碰到；该几何已经由 Imagination 裁决。

本轮必须明确选择且只调用一个提供的 Review Function：
- 意图、执行前提和 planner 均成立：commit(action_id)；
- 需要换已有 seed/point：调用 select/propose_pose，创建新的 Imagination；
- 对象、意图、执行前提或 planner 明确不成立：reject_action(action_id)；
- 只有当前真实视觉已经满足 User Task 时才 done(success=true)。
select、propose_pose 或 done 会立即作废当前 action_id。若还想执行眼前这个 Preview，本轮只能
commit。

不要用感知 Function 隐式跳过 Review。规划 returned/checked 不保证任务效果。Last Physical Action
是最近真实命令的因果事实，不是当前 Preview，也不证明抓取/释放成功。
GRIP 是当前真实归一化开度：0≈闭合，1≈张开；中间值具有歧义，既可能是物体阻挡手指，也
可能是闭合失败，不能单凭数值宣告“仍然打开”或“已经抓住”。若 Physical Effect Verification
显示 CLOSURE / UNVERIFIED，且此前尚未执行能检验物体随动的 arm motion，固定源位置仍有物体
并不能证明闭合失败；审查一个保持夹爪状态的小幅可逆 arm verification 是合理的。固定 source
crop 可能被机器人遮挡，source 变空也不能单独证明物体随夹爪移动。
若当前真实夹爪闭合，而下一段 arm motion 需要先张开通道，应拒绝当前空间 review，并在普通
Main 决策面先创建和 commit 纯 open target；不要假设空间 review 会顺带张开夹爪。
`requested_arm_delta_base_m` 是上一 commit 从执行前真实 TCP 到请求 target 的 BASE 位移，不是物体
位移或执行效果；若它没有 base +Z，上一步就没有完成“上抬随动验证”。

每轮必须且只能调用一个 Function，调用前只写一句简短依据。
"""

IMAGINATION_SYSTEM_PROMPT = """\
你是 Imagination Agent。你的唯一任务是在不改变真实世界的前提下，检查并微调当前空间
ActionTarget 的位置和方向。

Canvas 上层 OBSERVED NOW 是真实世界；下层 IMAGINATION 的全局区域是在同一当前点云上的
虚拟 target，CONTACT FOCUS 则是当前时刻由仿真相机直接渲染的密集近距离真实视图。紫色
TARGET GRIPPER 和规划状态不证明接触、抓持、释放或包含。结合主视角、融合 RGB-D、
CONTACT FOCUS 的 FRONT、SIDE、refinement goal 和 Edit Summary 共同审查
当前 target；FRONT 判断两指闭合通道，SIDE 暴露 FRONT 隐藏的前后/高度偏差，不能只看其中
一个投影。

【高优先级夹爪因果前提】
- 你不能编辑夹爪状态；open_gripper/close_gripper 由 Main 直接创建 gripper-only ActionReview。
- 新抓取接近时真实 GRIP 应已张开；若当前闭合且会阻碍接近，返回 failed，让 Main 先单独张开。
- 调整抓取 pose 时保持当前张开状态；当 CONTACT FRONT/SIDE 已支持包夹几何时交回 ready，
  到达真实位姿后再由 Main 根据新观测决定是否创建 close-only ActionReview。
- 抬升/运输默认保持当前真实夹爪状态；不要用空间 Preview 暗示抓取或释放结果。

【抓取 Preview】
只问：若从当前开度闭合，物体会落在两指扫掠里、两边都能碰到吗？FRONT 看是否在两指之间，SIDE
看深度和高度够不够。指尖擦边或只靠一边则还不够。够了就 ready，不要再往物体上压。Preview
不证明真的抓住。

ActionSeed 是规划器给的起点，不要仅为了让画面“看起来竖直”而旋转。四元数分量不是各轴角度。
姿态只改 CONTACT 里能指出来的那一个缺陷；相机在本次 session 内固定、WORLD +Z 始终朝上，rotate
后应看到紫色夹爪相对点云转动，而不是点云反向旋转。若只是平移已有姿态（抬升/运输），保持方向。
GRASP CONTACT 是建议的闭合起点，不是 pre-grasp。不要抬高以求更安全。若闭合后大概
只会捏到盖顶，转一次仍进不了两指之间，就 failed 换 seed；不要连续往下压来假装包夹。改完用同一
视角看是否变好。

每轮三选一：已经像能包住 → ready；能指出一个可见缺陷 → 一次 delta_move 或 rotate；看不清转
哪边 → 先 gizmo，下一轮必须 rotate 或停；修不好 → failed。不要为了用满轮数而改。

GRIP 是当前真实归一化开度（0≈闭合，1≈张开）；空间 target 始终继承它。你只负责让
空间姿态适合后续动作，不得把“几何上适合闭合”写成“已经闭合或已经抓住”。
若一次空间编辑使 motion prediction 变为 error，不要沿同一趋势盲目累计；应撤回、换方向，或
在一次明确纠正后仍无法恢复 returned plan 时结束为 failed。不得把 ARM ERROR 当作继续随机
搜索各轴的理由。Edit Summary 中的累计位移、累计旋转和最近两次编辑是当前控制
状态，不是对话历史；若两次编辑相互抵消且没有新的可见改进，应接受当前 target 或失败，而不是
继续振荡。refinement goal 是审查意图，不是已经成立的视觉事实。
BASE 是固定 robot-base/world 坐标：base +Z 恒为竖直上抬，base -Z 恒为下降。TOOL 是随当前
target 姿态旋转的 TCP 局部坐标；在 top-down 姿态中 tool +Z 可能朝向支撑面，绝不等同于“向上”。
Canvas 的 CONTACT FRONT/SIDE 角落标明锁定视角与向上的 WORLD +Z；它们不是随 target 旋转的
TOOL 平面。rotate 的正角仍遵循绕所选 BASE 或 TOOL +轴的右手定则；需要查看 TOOL 旋转轴时
调用 show_rotation_gizmo(frame="tool", axis="x"|"y"|"z")。若旋转符号
或幅度不确定，先用 5–15° 做一次 Preview 并观察紫色目标如何变化；不得用 ±90° 猜方向。只有
当前与期望姿态存在
明确的大角度差异时才使用超过 30° 的单次旋转。
若 refinement goal 使用世界方向（抬升、下降、向篮子方向平移），优先使用 base，并用 Edit
Summary 的 total_translation_base_m 检查累计方向；若累计方向与目标相反，不得继续同号编辑。

不要调用感知、open_gripper、close_gripper、commit 或 done。每轮必须且只能调用一个 Function，调用前只写一句简短依据。
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
            "一句短的反事实：若执行当前设想，应看到什么几何关系。不要另写一种抓取方向。"
        ),
    }


def _edit_definitions() -> dict[str, dict[str, Any]]:
    frame = {
        "type": "string",
        "enum": ["base", "tool"],
        "description": (
            "必须显式选择坐标系。base 是固定 robot-base/world 坐标，+Z 恒为竖直上抬、"
            "-Z 恒为下降；tool 是随目标姿态旋转的 TCP 局部轴，tool +Z 方向取决于姿态，"
            "top-down 时可能朝向支撑面"
        ),
    }
    return {
        "delta_move": _function(
            "delta_move",
            "对当前想象目标做厘米级局部位置修正；只更新 Preview，不执行。可连续调用。",
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
                    "description": (
                        "所选 frame 中的 [dx,dy,dz]，每轴单次不超过 0.03m。世界抬升/"
                        "下降应使用 base 的 dz；不要把 tool dz 当作世界高度"
                    ),
                },
                "frame": frame,
            },
            ("delta_xyz_m", "frame"),
        ),
        "rotate": _function(
            "rotate",
            (
                "绕 Canvas 角落所示 BASE/WORLD 或 TARGET TOOL 的 +axis，按右手定则编辑"
                "想象目标方向；只更新 preview，不执行。方向不确定时先用 5–15° 观察，"
                "不要用 ±90° 猜测。可连续调用。"
            ),
            {
                "axis": {"type": "string", "enum": ["x", "y", "z"]},
                "angle_deg": {
                    "type": "number",
                    "minimum": -90.0,
                    "maximum": 90.0,
                    "description": (
                        "按右手定则绕所选 +axis 的非零角度，绝对值不超过 90°；"
                        "局部探索优先 5–15°，超过 30° 需要明确的大姿态差异"
                    ),
                },
                "frame": frame,
            },
            ("axis", "angle_deg", "frame"),
        ),
        "open_gripper": _function(
            "open_gripper",
            (
                "Main-only：直接创建一个 gripper-only open ActionReview 并返回 action_id；"
                "不进入 Imagination、不会打开真实夹爪。下一轮仍须 commit(action_id) 才会执行。"
                "它不保证物体会释放或落入目标。"
            ),
        ),
        "close_gripper": _function(
            "close_gripper",
            (
                "Main-only：直接创建一个 gripper-only closed ActionReview 并返回 action_id；"
                "不进入 Imagination、不会闭合真实夹爪。下一轮仍须 commit(action_id) 才会执行。"
                "调用前应从真实 CONTACT FRONT/SIDE 确认物体处于两指闭合扫掠区域；它不保证抓住物体。"
            ),
        ),
        "show_rotation_gizmo": _function(
            "show_rotation_gizmo",
            (
                "在 Contact View 右侧显示所选 frame、单一 axis 的 -10°/+10°真实夹爪姿态"
                "对照；不会把三轴环覆盖在物体上，也不修改 target、不规划、不执行。"
            ),
            {
                "frame": frame,
                "axis": {
                    "type": "string",
                    "enum": ["x", "y", "z"],
                    "description": "要比较正负旋转方向的单一坐标轴",
                },
            },
            ("frame", "axis"),
        ),
    }


def function_definitions() -> list[dict[str, Any]]:
    edits = _edit_definitions()
    within = {"type": "string", "description": "可选的当前 region 搜索范围"}
    common = [
        _function(
            "detection_and_sam",
            "对当前真实 observation 依次执行语义 bbox detection 和 SAM 分割，返回当前图像中的二维 region；不刷新 observation、不移动机器人，也不证明接触、抓持、支撑或包含关系。同一 revision、同一 query 会复用已有 region，不产生新信息，也不能作为 motion planning 失败后的“刷新”手段。仅在没有准确 region 或需要更具体 query 时调用。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_grasps",
            "抓取已有 region 中物体时的默认几何生成器：生成多个同时包含位置和方向的粗略 ActionSeed，供 select/Imagination 视觉审查；不执行，也不保证抓取成功。每个 seed target 是建议的最终抓取接触/闭合位姿，不是 pre-grasp 或 clearance waypoint，不应被自动上抬。不要在尚未尝试这些 seed 时用手写 quaternion 的 point pose 替代。若实际返回无有效候选，重复检测同一目标不会改善几何；此时可换用 locate_point + propose_pose 等不同起点。",
            {"region_id": {"type": "string"}},
            ("region_id",),
        ),
        _function(
            "locate_point",
            "定位语义操作点并提升为 robot-base XYZ；它只返回一个位置，不生成抓取方向或完整 grasp pose，也不移动机器人。适合容器内部、表面、部件等目标点。若目标已有准确 region，应传入 within_region_id，将点限制在该目标内；省略时会在整张图搜索，可能落到干扰物。",
            {"query": {"type": "string"}, "within_region_id": within},
            ("query",),
        ),
        _function(
            "propose_pose",
            (
                "从 locate_point 返回且仍在 valid_point_ids 中的 point_id 与 offset 创建空间目标；"
                "不得虚构 current_tcp 等 ID。只预览并默认继承当前真实夹爪开度。当前 TCP 的"
                "相对移动应使用 start_imagination 交给 Imagination Agent。若移动前必须先改变开度，应先单独 commit "
                "gripper-only 目标。它不自动生成抓取方向；抓取已有 region 的物体时，应先使用 "
                "propose_grasps，而不是凭空手写 quaternion。省略 quaternion 时继承当前真实方向。"
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
                "grasp seed 是闭合起点，不是悬停点。refinement_goal 用一句反事实：若按该 seed "
                "闭合，物体是否在两指扫掠里。不要另写抓取方向，也不要要求手指先贴住物体。"
                "若接近动作要求夹爪预先张开，应先单独 commit gripper-only 目标。"
            ),
            {
                "seed_id": {"type": "string"},
                "refinement_goal": _refinement_goal(),
            },
            ("seed_id", "refinement_goal"),
        ),
        _function(
            "start_imagination",
            (
                "从当前真实 TCP 启动一个独立空间 Imagination session。用于相对抬升、下降、"
                "局部平移或旋转；空间 Preview 继承当前真实夹爪开度但不能修改它。本调用不编辑"
                "target，也不执行物理动作。纯夹爪目标应由 Main 直接调用 open_gripper/"
                "close_gripper 创建。"
            ),
            {"refinement_goal": _refinement_goal()},
            ("refinement_goal",),
        ),
        edits["delta_move"],
        edits["rotate"],
        edits["open_gripper"],
        edits["close_gripper"],
        edits["show_rotation_gizmo"],
        _function(
            "finish_imagination",
            "结束本次想象审查。ready 请求 Main 审查最终 Preview；failed 放弃该 target。",
            {"status": {"type": "string", "enum": ["ready", "failed"]}},
            ("status",),
        ),
        _function(
            "reject_action",
            (
                "明确否决当前 ActionReview 且不执行、不刷新 observation；当 target 不合理，"
                "或必须返回普通感知/规划时使用。调用后该 action_id 立即失效。"
            ),
            {"action_id": {"type": "string"}},
            ("action_id",),
        ),
        _function(
            "commit",
            (
                "唯一物理操作，也表示 Main 对当前 ActionReview 的显式批准；执行后刷新真实 "
                "observation。空间 ActionReview 只执行 arm；gripper-only ActionReview 只执行夹爪。"
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


def _function_subset(names: tuple[str, ...]) -> list[dict[str, Any]]:
    definitions = {item["function"]["name"]: item for item in function_definitions()}
    return [definitions[name] for name in names]


def main_function_definitions() -> list[dict[str, Any]]:
    """Functions available while Main is choosing a new action."""

    return _function_subset(STANDARD_MAIN_FUNCTION_NAMES)


def review_function_definitions() -> list[dict[str, Any]]:
    """Functions available while Main owns an explicit ActionReview."""

    return _function_subset(REVIEW_FUNCTION_NAMES)


def imagination_function_definitions() -> list[dict[str, Any]]:
    edits = _edit_definitions()
    return [
        edits["delta_move"],
        edits["rotate"],
        edits["show_rotation_gizmo"],
        next(
            item
            for item in function_definitions()
            if item["function"]["name"] == "finish_imagination"
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
    "ACTION_REVIEW_SYSTEM_PROMPT",
    "FUNCTION_NAMES",
    "IMAGINATION_FUNCTION_NAMES",
    "IMAGINATION_SYSTEM_PROMPT",
    "REVIEW_FUNCTION_NAMES",
    "STANDARD_MAIN_FUNCTION_NAMES",
    "SYSTEM_PROMPT",
    "function_definitions",
    "imagination_function_definitions",
    "main_function_definitions",
    "parse_action",
    "review_function_definitions",
]
