# VAW M1.5 — Agentic System Completion

> **现行 Context OS 更新（v37）**：当前实现已经删除旧 `post_commit` 页面、one-shot physical
> continuity 文本和原始 Function outcome 回灌，改为四个正交输入层：Current Canvas、最多六条
> 物理 Task Memory、revision-local Live References 和覆盖更新的 Current Function Event。现行规范见
> [`AGENTIC_CONTEXT_OS.md`](AGENTIC_CONTEXT_OS.md)；下文的 v1–v36 内容保留为实验演进记录，不再
> 定义当前接口。

## 1. 目标

M1.5 的目标不是继续优化一张好看的 Canvas，而是完成一个可验证的 Visual Action
Workspace Agentic System：

> Main Agent 依据当前真实视觉决定任务级动作，Imagination Agent 在清晰、可控的局部视觉中
> 收敛一个虚拟空间动作，Main 审查后 commit；简单局部移动与夹爪控制可由 Main 直接执行；
> 物理执行后，Main 根据新的真实状态继续当前目标或
> 进入下一个动作，最终在真实 LIBERO-PRO 中完成基本 pick-and-place。

当前编排已收敛为 Main-owned ReAct：Imagination 是 Main 通过 `refine_action` 同步调用的局部
SubAgent，不再作为与 Main 并列、逐 turn 切换 ownership 的顶层循环。权威接口以
[`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md) 为准；下文旧 `ActionReview/ownership`
措辞是历史实验记录，不再定义当前代码结构。

Canvas、Function 和 Prompt 都只是该闭环的组成部分。任何单独截图、IK 成功、轨迹生成、
夹爪闭合、Agent 声称成功或 scripted smoke 都不能证明 M1.5 完成。

本文是 M1.5 的权威设计与验收文档。`archive/M1_4_2_DUAL_AGENT_RUNTIME.md` 和
`archive/M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md` 保留为历史基线与失败记录；若与本文冲突，以本文为准。
当前已实现的 Runtime、Function ownership、Context Builder 和 Canvas 契约集中记录在
[`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)，本文继续承担研究目标、版本演进与验收
记录，不再要求读者从全部历史版本段落反推当前实现。

## 2. 完成定义

M1.5 只有同时满足以下条件才算完成。

### 2.1 真实任务门槛

- 主任务固定为 `libero_object_swap:0`，不得在运行中人工接管；
- 使用同一冻结的 Main/Imagination 模型、Prompt、Function schema 和 renderer，运行 seeds
  `0, 1, 2`，至少 `2/3` episode 获得真实 `env_success=true`；
- 至少一条成功 trace 完整包含：语义定位、动作 seed、局部微调、Main review、arm/gripper
  commit、抓持后的真实视觉核验、运输、释放和最终真实任务成功；
- 再选择一个未用于实现调试的基础 pick-place task，至少完成一个 seed，且不得加入任务名、
  物体名、固定坐标或 pick/place phase 的代码分支。

### 2.2 Agentic 闭环门槛

- `commit(action_id)` 只执行 Imagination 返回 ready 的空间动作；Main `delta_move/open/close`
  是无需 commit 的直接物理 Function；
- Imagination 不直接 commit，Main 不承担逐步局部优化；
- Imagination 正常结束、达到编辑上限或失败都必须在同一次 `refine_action` 内返回 Main；主动
  `ready` 只更新 Main-owned pending action，编辑上限按 `failed` 返回并回滚，更不能自动批准；
- Main 必须看到最终 Preview，并能够选择 commit、重新进入 Imagination、换 seed 或放弃；粗
  `select/propose_pose` 不得直接 commit；
- commit 后的新请求必须包含当前真实视觉和一条最小动作连续性信息，使 Main 知道刚执行的是
  arm、gripper 或二者，以及原始意图；
- revision-local region/point/seed 不跨物理动作复用；最近一次物理意图在当前真实 observation
  revision 内 overwrite-only 保留，直到下一次 commit 覆盖；
- Agent-visible Context 不包含 reward、environment success、privileged object pose、raw depth、
  calibration、raw mask/cloud 或 planner trajectory。

### 2.3 视觉可控性门槛

- 固定输出 `2048×1280`、DPR=1、无滚动、动画或响应式重排；
- `OBSERVED NOW` 始终只显示当前真实 observation，不叠加虚拟结果；
- `select/propose_grasps` 时最多五个 seed 必须同时完整可见，ID 与图像一一对应，不被裁切、
  遮挡或缩小到无法比较；
- Imagination 的主要面积必须是 target-centric Contact Focus，而不是再次展示全局场景；
- 在默认相机和常见工作距离下，目标物体与 target gripper 的联合包围框应占 Contact Focus
  短边的 `30%–75%`；超出范围时 compiler 必须自适应，不依赖任务专用坐标；
- 单次 `3 cm` 位移和 `10°` 旋转必须在相邻 Preview 中产生清晰可辨的几何变化；
- target 处不绘制遮挡物体的局部三轴。BASE/WORLD 或 orientation 提示只能出现在固定边角或
  独立 widget；
- 紫色实体只表示未执行目标；当前真实机器人仅使用白色轮廓标记，不覆盖当前视觉证据。两者都
  不预测或暗示物体会被抓住、移动、释放或进入容器。

## 3. 当前基线与已证实缺陷

基线 commit：`91f9d17 feat(vaw): checkpoint VIA-style dense canvas`。

基线优势：

- agentview 与 camera-aligned dense RGB-D surface 清晰、稳定；
- 当前机器人与紫色目标机器人来自统一 FK/相机标定；
- 当前真实世界与未执行 Imagination 已经视觉分层；
- 只有 commit 会刷新真实 observation。

真实 trace `via_canvas_qwen35plus_t0_s1` 暴露出以下系统缺陷：

1. 三次 Imagination session 均没有主动调用 `finish_imagination`；每次连续编辑到六轮上限；
2. 前两次 handoff 以 `budget_exhausted` 暴露给 Main，Main 将其误解为不可 commit 的失败；
3. Imagination 在 `+Y/-Y`、`+Z/-Z` 间反复抵消，因为请求中没有初始 target、累计变换和最小
   编辑摘要；
4. 当前 target-focused crop 仍覆盖约 84% 原始图像宽度，2–3 cm 修正的视觉差异过小；
5. runtime 把 Main 的完整自然语言 response 直接作为 `refinement_goal`，历史失败和预算描述会
   污染局部几何任务；
6. commit 后 revision 正确清除旧 evidence，但也没有向下一轮 Main 保留刚执行动作的意图与
   实际阶段，导致任务推进重新起步；
7. 当前 previous-observed 图片只有视觉差异，没有说明刚才执行的是 arm、gripper 或二者，不能
   独立承担因果连续性。

这些问题共同说明：当前失败主要来自 Agent harness 和 Context Builder，而不是 VAW 的核心
视觉上下文思路。

## 4. 目标架构

```text
Current real observation
        │
        ▼
Main Agent
  ├─ perception / evidence
  ├─ create ActionSeed + concise refinement goal
  ├─ create gripper-only open/close ActionReview
  └─ review final ActionReview
        │
        ▼
Imagination Agent
  ├─ inspect Contact Focus
  ├─ delta_move / rotate
  ├─ minimal edit memory
  └─ handoff: review_required | failed
        │
        ▼
Main: commit | revise | replace | abandon
        │
        ▼
commit (only physics)
        │
        ▼
Fresh observation + one LastPhysicalAction
        │
        └─ continue same focus or choose next task action
```

该架构不是 pick/place 状态机。Main 仍然依据当前视觉和任务自由选择动作，runtime 只维护所有权、
revision 生命周期与最小因果连续性。

## 5. Context Builder 设计

### 5.1 Main 请求

```text
Main System Prompt
User Task
Current semantic manifest
Optional ActionReview (spatial handoff or Main-created gripper-only preview)
Optional LastPhysicalAction (one item, overwrite-only)
Current 2048×1280 Canvas
```

不加入 Function transcript、旧 reasoning、旧 evidence ID 或多轮图片历史。

### 5.2 Imagination 请求

```text
Imagination System Prompt
Concise refinement goal
Current ActionTarget
EditSummary
Current 2048×1280 Canvas
```

`EditSummary` 是命令状态，不是对话历史：

```text
EditSummary
├── initial_target
├── current_target
├── total_translation_from_seed
├── total_rotation_from_seed
├── last_edit
└── previous_edit
```

它只活在当前 Imagination session。用途是识别累计位移、方向反转和是否已经没有明确改进。

### 5.3 Refinement goal

禁止继续使用 Main 的完整 rationale 作为 refinement goal。启动 Imagination 的 Main Function
必须显式携带一句短目标，例如：

```text
使两指在罐体中部形成对称包夹，并避免触碰地面。
```

该文本描述期望几何，不描述旧失败、预算、未经验证的抓持结果或任务阶段。

### 5.4 LastPhysicalAction

```text
LastPhysicalAction
├── intent
├── executed_stages: arm | gripper
└── outcome: completed | arm_failed | gripper_failed
```

- overwrite-only，不形成列表；
- 不含 receipt ID、revision、旧 action ID 或 reasoning；
- 在当前真实 observation revision 内持续存在，由下一次 commit 覆盖；
- 不判断 task effect，不替代当前真实视觉；
- 可同时携带一次 before/current 视觉对照，但两张大图只在 commit 后的第一个 Main 请求中呈现；
  后续感知调用只保留紧凑动作事实，避免既丢失因果又长期占据视觉面积。

## 6. Canvas 设计

### 6.1 OBSERVED NOW

- 大幅干净 agentview；
- 与 agentview 互补的 episode-locked 反侧 MuJoCo RGB 视角；
- 紧凑 robot state；
- 不显示 target、region、seed 或 planner 状态。

### 6.2 Seed Selection

- 下层切换为五张等宽、完整候选卡；
- 每张只显示一个 seed 的 target gripper、局部目标几何和 ID；
- 五张卡使用统一物理尺度与视角，禁止每张独立 auto-zoom；
- 必须保留足够大的对象/手指轮廓，不显示 score 和原始 planner 数值；
- 全局整臂结果只保留一个小型 inset，不能挤占 candidate comparison。

### 6.3 Contact Focus

当前 Imagination 下层采用固定结构：

```text
┌──────── camera-aligned global Preview ───────┬── CONTACT FRONT ─┐
│ current dense RGB-D surface                  │ direct MuJoCo RGB│
│ current RGB geometry + purple target         ├── CONTACT SIDE ──┤
│                                              │ direct MuJoCo RGB│
└──────────────────────────────────────────────┴───────────────────┘
```

- Contact Front/Side 由 episode-private MuJoCo Camera 直接渲染，不再由稀疏点云 novel-view
  重投影；相机在一次 Imagination session 内锁定且 WORLD +Z 保持竖直；
- 当前真实夹爪只显示白色轮廓；青色表示 previous Preview，紫色表示 current Preview；
- `delta_move`：根据每张 Contact Camera 的真实标定显示两个屏幕内 BASE 正轴；最接近视线方向
  的第三轴放入独立 `DEPTH IN/OUT` 子卡，避免三轴在同一原点重叠；
- Contact Camera 在目标深度显示 0–3 cm 透视标尺；平移 Preview 同时显示参考点到目标点的
  标定投影箭头与实际长度，辅助模型把像素变化映射到厘米动作；
- `rotate`：左上固定 `ROTATE BASE` 三维右手正向图例不覆盖物体；需要确认具体 frame/axis 的
  正负结果时，`show_rotation_gizmo` 在独立侧栏显示 `−10°/+10°` 真实夹爪姿态对照；
- gripper-only preview：由 Main 直接创建，只改变虚拟指宽；不进入 Imagination，并明确物体点
  保持当前 observation、不模拟动力学；
- planner error 只作为紧凑状态，不替代几何判断。

### 6.4 Post-Commit

- 上层全部更新为新的真实 observation；
- 下层短暂显示 `BEFORE → CURRENT` 目标附近视觉对照和 LastPhysicalAction；
- 不显示大块 receipt、JSON、revision 或 action ID；
- 下一轮 Main 可以直接从当前 TCP 启动新的 Imagination，例如累计四次编辑形成 `+10 cm` lift，
  也可以重新感知或开始下一任务动作。

## 7. Handoff 与停止语义

Agent-visible handoff 只保留：

```text
review_required(action_id, source_ref?)
failed(source_ref?)
```

- Imagination 主动认为足够合理时产生 `review_required`；
- 达到 turn limit 时产生 `failed`，不创建 `ActionReview` 或 action ID；
- `termination_reason=turn_limit` 只写 trace，不进入 Agent-visible handoff；
- Main 只审查动作意图、真实执行前提、planner 可执行性和下一物理动作是否合理；局部位置、
  方向、两指通道与邻近碰撞由 Imagination 单独裁决；
- Review 不提供返回同一 target 继续微调的 Function；失败后必须换 seed、point 或动作路线；
- ActionReview 是一次 Main 决策的 offer：下一次成功 Function 若不是 `commit`，即视为放弃，
  action 与私有 plan 同步销毁；无效调用不消费 review；
- 若 target 没有可执行 motion plan，必须产生 `failed` 或显式 planner error，不创建可 commit
  review。
- `source_ref` 只携带当前被审查的 seed/point 引用，不是历史；它使 Main 知道刚被否决的动作
  起点，避免在没有新视觉证据或不同修正策略时立即重复同一个 seed。

Imagination 每轮的概念决策是：

```text
accept current target
OR make one visually justified edit
OR fail this target
```

只有能够指出一个当前可见几何缺陷时才继续编辑。连续反向编辑不是自动 gate，但必须通过
`EditSummary` 对模型可见，促使其停止振荡或换策略。

## 8. Milestones

### M1.5.0 — Baseline Freeze and Audit

- 冻结 `91f9d17` 为 dense Canvas baseline；
- 保存 `via_canvas_qwen35plus_t0_s1` 为 handoff/oscillation 失败样例；
- 本文档冻结完成定义、非目标与实验表。

验收：当前实现、失败原因和后续每项修改都能映射到一个明确 gate。

### M1.5.1 — Control-Readable Canvas

- 重构 seed selection 比例，确保五候选完整可读；
- 加入 metric target-centric Contact Focus；
- 加入 source surface emphasis、previous target silhouette、move/rotate edit cues；
- 保持 agentview 与 dense observed world 的现有清晰度。

验收：固定 fixtures 覆盖 seed、delta、rotate、gripper-only、planner-error；截图确定性通过；
静态 VLM probe 在无选项条件下能描述主要几何缺陷并给出方向一致的修正。

### M1.5.2 — Imagination Convergence and Review

- 引入最小 `EditSummary`；
- refinement goal 改为显式短文本，不再复用完整 rationale；
- 对 Main 隐藏内部预算原因；主动 ready 才产生 review，超限产生无 action 的 failed handoff；
- 更新 Imagination Prompt 和 Function 描述，使 accept/edit/fail 对称。

验收：fake provider 覆盖主动 ready、turn-limit failed、agent failed 和 planner error；真实静态
session 不再持续出现无信息的正负方向抵消；超限 target 不可被 Main commit。

### M1.5.3 — Post-Commit Continuity

- 加入 overwrite-only `LastPhysicalAction`；
- 将 before/current 对照编译进同一 Canvas；
- 保留一次自然语言 focus，不保留旧 evidence ID；
- 删除额外 previous image message，避免输入格式变化。

验收：arm-only、gripper-only、arm+gripper commit 后，Main 能区分刚执行的内容，并分别选择
继续微调、改变夹爪、验证 lift 或进入下一动作；无 Function history 泄漏。

状态：离线修订完成，待真实复测。真实模型 trace 中，Main 在 arm commit 后直接继续创建 close-gripper preview，
没有重新从 detection/propose 启动任务；在 arm+gripper commit 后，也能够从当前真实画面提出
小幅 lift 来核验抓持。后续 trace 又证明“一次 Main 决策后删除物理意图”仍然过短：目的地感知
会擦除正在运输的因果状态。修订后，Function 参数错误不会消费因果或 review；成功的非 commit
调用只消费大幅视觉对照，紧凑 LastPhysicalAction 保留到下一次 commit。

### M1.5.4 — End-to-End Pick-and-Place

- 在 `libero_object_swap:0` 完成 pick、lift verification、transport、place 和 release；
- 修复只由真实 trace 证明的 Function/backend/Context 缺陷；
- 每次真实 run 记录冻结配置、commit、trace 和第一失败原因。

当前第一失败原因审计：`m153_qwen35plus_post_commit_t0_s1` 的初次
`detection_and_sam("alphabet soup can")` 把前景红绿罐误识别为任务目标。SAM mask、对象点云、
camera-to-base 变换与 grasp target 都与该错误 box 内部一致，因此不是坐标转换错误。根因是底层
detector 被要求强制返回单个 box，却没有被要求在多个同类容器间做精确语义消歧。

第一次只增强 forced-single prompt 的尝试并不充分：固定帧和 scripted smoke 能返回正确目标，但
`m154_qwen35plus_grounding_t0_s1` 在抓取失败后的再次检测中仍选择了红绿罐。这证明对同一个模型
重复确认会产生自洽式错误，不能作为 verifier。

最终修复保持 Function schema 和 CaP-X 不变，在 VAW 私有 detector 边界执行：

```text
最多三个 distinct semantic candidates
→ full scene + 每个候选的放大 crop
→ 独立选择一个候选，或明确返回 ambiguous
→ 仅对通过复核的 box 调用 SAM
```

Agent 仍只看到原始简短 query、`region_id` 和最终 bbox；candidate evidence、复核回答和坐标约定
只进入 trace。若复核无法区分则返回 error，不注册看似权威的 region。坐标解析复用 CaP-X 已配置
的模型约定：GPT 使用 pixel，Qwen 使用 norm1000；真实测试曾捕获并修正强制 norm1000 导致 GPT
box 上移的 wiring bug。

真实 trace `m154_candidate_review_pixel_scripted_t0_s1` 中，生成器同时提出红绿罐
`[397,303,447,371]` 和蓝色罐 `[339,203,372,251]`，放大复核选择第二项；随后 SAM、point lift、
五个 grasp seed 与 CuRobo 链路均围绕正确目标完成。该 smoke 只证明 grounding wiring，不计入任务
成功门槛。

随后引入同一 observation 的 2× agentview semantic render。真实 scripted trace
`m154_hires_grounding_scripted_t0_s1` 在高分辨率语义图上选择蓝色 alphabet soup can，并把 bbox
无损映射回 `800×512` 观测后完成 SAM、point lift、五 seed 与 CuRobo 链路；高分辨率 RGB 仅活在
private episode context，不进入 Canvas、manifest 或 Function result。

真实 Agent trace `m154_qwen35plus_hires_grounding_t0_s1` 与
`m154_qwen35plus_control_semantics_t0_s1` 进一步证明：多个 revision 的目标身份已经稳定正确，
但任务仍未成功。前者将四元数分量误当旋转角度并反复旋转；删除该控制歧义后，后者不再出现
无依据旋转，并首次让真实夹爪从 `0.968` 收到 `0.768`，说明发生了实体接触。然而下层单一
camera-aligned Preview 被紫色 hand 遮挡，无法显示物体是否位于两指闭合通道；Imagination 因而
在 Z 方向振荡，甚至将 TCP `z=-0.014m` 误判为合理，最终 CuRobo 执行不收敛。

因此当前第一失败原因已经从 semantic grounding 转移为 **contact observability**。M1.5.4 的下一
实现版本在全局 Preview 旁增加由当前 agentview+wrist RGB-D 编译的正交 `JAW PLANE`，并将失败
handoff 的当前 `source_ref` 交回 Main。它不增加 phase、动作 gate、任务专用坐标或预测物体运动。

真实 Agent trace `m154_qwen35plus_contact_focus_t0_s1` 证明 Contact Focus 本身有效：第一次抓取
Imagination 只做一次 `+2 cm Z` 即主动 ready；close session 只做一次 `-2.5 cm Z` 即 ready；
arm+gripper commit 后真实 `GRIP=0.246`，CURRENT 图中蓝色罐已经离开原支撑并位于两指间。Main
也在下一轮明确判断“已经抓起”，随后正确检测 basket。

同一 trace 同时暴露三个新的 harness 缺陷，而不是 Canvas idea 失败：

1. basket detection 成功后，runtime 把 LastPhysicalAction 连同大图一起清除，下一轮 Main 失去
   “刚抓起目标、正在找目的地”的因果状态，重新从检测并抓取罐头开始；
2. Main 在后续 review 中用 detection 表达放弃，但旧 ActionReview 仍留在 manifest，下一轮又被
   commit；
3. 被夹持/遮挡状态下 Contact-GraspNet 偶发输出 target `[0.641,-0.102,0.145]`，而 source mask
   点云中心为 `[0.407,-0.094,0.073]`、最近距离 `0.221 m`，该 scene-scale outlier 仍被注册成 seed。

当前修复将“紧凑因果事实”和“一次性视觉对照”拆开；将 ActionReview 定义为一次决策 offer；
并用 source 点云自身的 robust 3-D extent 做 candidate/source 一致性检查。该检查不排序 seed、
不规定抓取方向，也不含任务或物体类别分支。

真实 scripted trace `m154c_source_guard_scripted_t0_s1` 证明正常的单视角候选仍能通过上述检查。
真实 Agent trace `m154c_qwen35plus_causal_t0_s1` 则在被遮挡 revision 中拒绝了唯一异常 seed：其
target 到 source 点云最近距离为 `0.217 m`，同时保留其他 revision 的正常候选。该 run 还暴露
gripper-only visual edit 没有 reference pose 时的 Canvas 崩溃，修复后已加入确定性回归。

真实 Agent trace `m154d_qwen35plus_settle_t0_s1` 进一步把第一失败位置推进到 place：

- 首次 CuRobo approach 曾因终点 joint residual `0.039 rad` 被误报失败。Reduced API 不返回
  waypoint 状态，因此 adapter 现在只对同一个缓存终点做一次有界 settle；不放宽 `0.02 rad`
  验收阈值、不重规划也不换目标。复测中 arm commit 达到 `11.1 mm` TCP error，后续 grasp/lift
  commit 分别达到 `19.3 mm` 与 `9.6 mm`；罐头在 CURRENT 图中明确离开支撑面并随夹爪上移。
- 闭合后 `GRIP≈0.77` 实际来自罐体阻挡，而不是“仍然打开”。Main 一度因此重复 close，说明
  Context 可见但因果提示仍不完整；Prompt 现明确要求以小幅随动验证抓持，不能只按开度判定。
- 抓起后 Main 没有 grounding basket interior，而把 `delta_move` 当成长距离导航；累计约 15cm
  后过早 open，罐头落在篮子左侧。下一修订明确区分：远处语义目标必须
  `detection_and_sam → locate_point(within_region_id) → propose_pose`，`delta_move` 只负责附近
  厘米级修正。
- 同一 revision、同一 query 的 detection 现在幂等复用 region，防止失败恢复时生成一串等价
  ID 和重复 Canvas evidence。全图 point 曾错误落到 basket（`x=0.751m`）；Function 描述现要求
  已有 region 时显式传 `within_region_id`。

因此 M1.5.4 仍未验收，但真实闭环已经证明 identity grounding、candidate/source 对齐、arm
execution、contact close 与 lift-follow verification 可连续工作。当前首要失败已收敛为
**destination grounding 与 transport/place action selection**，而不是 Canvas 分辨率或 pick 执行。

真实 Agent trace `m154e_qwen35plus_destination_t0_s1` 验证了上述 destination 语义修订：当
`propose_grasps` 返回空候选后，Main 不再循环检测，而是正确执行
`locate_point(within_region_id) → propose_pose`。但旧的 `LastPhysicalAction=arm_failed` 同时出现在
新的 ActionReview 旁，Main 连续把已经重新规划的动作误当成旧失败。因果事实本身没有错，错误在
Context Builder 把“上一动作结果”和“当前待审动作”并列成了两个竞争焦点。修订后底层
LastPhysicalAction 仍保留在 trace/state，但只要存在 Imagination 或 ActionReview，Agent-visible
packet 与文本就隐藏旧物理事实；当前 Preview 成为唯一待审对象。

真实 Agent trace `m154f_qwen35plus_review_scope_t0_s1` 证明该作用域修正有效：旧 arm failure 没有
阻止 Main 审查新的 point-based Waypoint，链路能够完成
`locate_point → propose_pose → delta_move → review → commit`。该 run 随后暴露出独立的执行层缺陷：
LIBERO 视频恰好记录 999 帧，而采样率为每 4 个 simulation step 一帧，对应默认 4000-step
horizon；第二次 commit 在已经 terminated 的 episode 上继续运行并返回
`executing action in terminated episode`。根因是 VAW 把 Reduced API 的逐 waypoint settle 配成
`0.01 rad / 120 steps`，一条轨迹即可消耗数千 simulation steps，同时 Runtime 没有传播底层
episode termination。

同一 `libero_object_swap:0` reset 上用安全的 `base +3 cm Z` 做了隔离 A/B 诊断：

- 原 VAW 默认值执行 31-waypoint CuRobo 轨迹消耗 81 sim steps，TCP error `8.43 mm`；
- 对齐 CaP-X low-level trajectory helper 的 `0.025 rad / 15 steps`，并保留 VAW 的严格
  `0.02 rad` 最终 joint residual gate，只消耗 43 sim steps，TCP error `6.13 mm`、最终 residual
  `0.0172 rad`，环境未终止。

因此执行修订不放宽最终验收、不重规划、不修改 CaP-X：只缩短 intermediate waypoint 的阻塞预算，
仍允许对同一缓存终点做一次最多 120-step settle。Runtime 另接入 trace-only environment terminal
check；任何 commit 导致底层 episode 结束后都立即以 `env_terminated` 停止，不能继续向死环境发出
感知或动作。

真实 Agent trace `m154g_qwen35plus_bounded_exec_t0_s1` 验证了执行预算修订：7 次 physical
commit 共记录 328 个每 4 simulation step 采样的视频帧，环境没有触及 4000-step horizon；首个
approach、close 与后续移动的 TCP error 均约 `9.5–12.6 mm`。因此 executor/horizon 已不再是当前
首要失败。

该 trace 把下一失败精确定位到 Imagination→Main 的控制语义断点：Main 要求“抬升”，却用
`tool +Z` 启动编辑；top-down grasp 下该局部轴朝向支撑面。Imagination 连续编辑后最终 target
相对初始 target 在 base Z 方向下降约 `11 cm`，但 handoff 时 `ActionReviewArtifacts` 丢弃了
`EditSummary`，Main Canvas 显示 `MOVE ΔXYZ —`，随后把明显低于当前 TCP 的 target 误判为合理并
commit。修订采用两个通用机制，而非任务 phase/gate：

1. schema 和 Function 描述明确 `base +Z` 恒为世界上抬，tool 轴随 target 姿态旋转，世界方向动作
   不得把 tool Z 当作高度；
2. handoff 原子冻结 `initial target / current target / cumulative base translation/rotation`，并同时
   编译进 Main 的 ActionReview Canvas 与最小控制文本，使 Main 能逐轴核对 refinement goal。

新版本为 Web schema 16 / `vaw-context-v15-review-edit` / renderer
`context-web-v15-review-edit`。它只传递当前命令状态，不恢复 transcript history，也不暴露 trajectory、
相机参数或环境真值。

真实 Agent trace `m154h_qwen35plus_review_edit_t0_s1` 对 frame 修订给出正向证据：所有“抬升/下降”
编辑均显式使用 base Z；不可执行 pose 在 handoff 边界被判为 failed，没有形成可被 Main 错误 commit
的 ActionReview。该 run 同时暴露了更早的 capability-routing 失败：Main 在准确 `region1` 已存在时
完全跳过 `propose_grasps`，三次用同一个 point 配合手写 offset/quaternion 构造侧抓，所有规划均为
IK error；每次失败后又调用幂等 detection，误以为能获得“fresh”几何。

这不是要增加 pick phase，而是 Function affordance 描述不充分。修订保持十一项 Function 和 runtime
调度不变，只明确工具能力边界并调整展示顺序：对象 region 的抓取默认由 `propose_grasps` 生成完整
位置+方向 seed；`locate_point` 只提供 XYZ，`propose_pose` 用于放置/表面点或已有明确方向约束的直接
pose；motion failure 不会让同 revision 的 perception 变旧，重复 detection 不能恢复 IK。

真实 Agent trace `m154q_qwen35plus_cumulative_review_t0_s0` 证明 gripper-only baseline 与局部 Review
修订有效，但进一步暴露了 grasp seed 的控制语义缺口：GraspNet seed 已是最终接触/闭合目标，Main 与
Imagination 却把它当作需要先抬高的 pre-grasp。首个 seed 被累计上抬约 `5 cm` 后执行，夹爪悬在罐头
上方；后续 point pose 又把绝对 target Z 误读成“离物体表面的高度”，生成过低目标并在真实执行中推走
物体。终点验收正确拒绝了这些大误差，因此不能靠放宽 executor gate 掩盖该问题。

修订不加入 pick 流程或接触真值，而是在当前视觉证据上补齐通用语义：grasp 来源的 target 标记为
`GRASP CONTACT`，明确它是最终接触位姿而非 pre-grasp；compiler 从当前 revision 的 region RGB-D
表面计算 `TCP→SOURCE` 最近距离，作为是否仍明显悬空的度量提示。该标量不是抓持判定，也不预测
动力学。Prompt 同时要求持续 `ARM ERROR` 在一次有依据修正后交回失败，禁止用随机轴旋转搜索 IK。
版本更新为 Web schema 18 / `vaw-context-v17-contact-semantics` / renderer
`context-web-v17-contact-semantics`。

冻结静态对照进一步分离了 Prompt 与 Context 的作用：仅把新 Prompt 加到缺少距离标量的旧
`context_0003.png` 上，Qwen3.5-Plus 仍以“安全 approach”为由执行 `base +Z 2.5 cm`；在同一冻结
任务/seed 重新生成的 v17 无物理 contact probe 中，所选 grasp target 的
`TCP→SOURCE = 1.6 mm`，模型不再上抬，而是提出一次 `base-Z 15°` 姿态修正。该结果证明距离证据
消除了当前已知的 pre-grasp 误读，但旋转是否改善接触仍必须由连续 Preview 和真实闭环验证，不能
作为任务成功证据。

真实 Agent trace `m154r_qwen35plus_contact_semantics_t0_s0` 验证了 contact 修订的真实作用：Main
使用首个 grasp seed，Imagination 没有上抬或随机旋转；arm+close 后 `GRIP=0.796`，随后 base +Z
`3 cm` 的 POST-COMMIT 画面清晰显示 alphabet soup can 随夹爪离开支撑面。Main 当轮正确判断
“grasp verified”并转向检测 basket。但 basket detection 是非物理调用，Context Builder 却在下一轮
完全移除了 before→current 核验画面，只留下 basket grounding 与紧凑文本；Main 随即在同一真实
observation 上反转为“罐头仍在桌面”，开始重新抓取手中物体。

因此下一修订不是 object tracking，也不是把“已抓住”写成真值，而是让当前 observation 的物理因果
视觉在非物理调用间持续：完整 POST-COMMIT 对照仍只显示一轮；之后 grounding/idle/error 模式在
下层保留小型 `LAST COMMIT · CURRENT OBSERVED` 当前 crop，与新 evidence 并列。该 crop 与当前
RGB 属于同一 revision，不是旧图或历史截图；进入新 Imagination/Review 时隐藏，下一次 commit
自动替换。Prompt 只补充一条因果约束：非物理 Function 不改变 observation，不能仅因 evidence
面板切换或 GRIP 数值反转刚从当前 RGB 得出的随动结论。版本更新为 Web schema 19 /
`vaw-context-v18-physical-continuity` / renderer `context-web-v18-physical-continuity`。

真实 Agent trace `m154s_qwen35plus_physical_continuity_t0_s0` 证明 v18 已能在 grounding 后保留
“抓取失败并重试”的因果状态，也完成了从目标检测、grasp seed、抬升、basket point 到 release 的
完整自主决策链；但环境最终仍为失败。审计发现两个确定性 Context Builder 问题：第一，保留的
`post_commit` 页面优先级高于新 `ActionReview`，导致 Main 在审查后续动作时实际看见旧回执而不是
将要 commit 的紫色目标；第二，抬升核验只围绕 TCP 裁剪，遮挡下无法明确判断同一目标是否仍留在
原支撑位置。Main 因此把未随动的罐头误判为已抓住，并把这个错误 belief 一致带到 basket。

v19 修订不加入 object tracker 或任务 phase。`ActionReview` 现在始终优先占据审查画面；episode
私有状态只保留最近 grasp source 的 query 与固定 agentview bbox。每次后续 commit 编译三块真实
视觉：`SOURCE BEFORE`、`SAME SOURCE LOCATION NOW` 和 `CURRENT ACTION AREA`。固定 source
位置仍出现对象是“未随动”的直接反证；原位置变空只支持“离开原处”，仍需结合当前动作区域和
全局 RGB 判断是否附着。该 artifact 在新 grasp source 出现时覆盖，不产生公共 object ID、不做
跨相机 tracking，也不注入环境真值。版本更新为 Web schema 20 /
`vaw-context-v19-causal-verification` / renderer `context-web-v19-causal-verification`。

真实 trace `m154t_qwen35plus_causal_verification_t0_s0` 证明 v19 的 Review 优先级与固定 source
对照均已生效，Agent 也能自主完成 grasp、lift、basket grounding、placement Preview 与 release
commit；但环境仍为失败。完整轨迹暴露的不是新的视觉缺口，而是所有权交接时的语义缺口：普通
Main 已知“部分开度可能来自物体阻挡、需要随动测试”，Action Review 却看不到发起动作的
`Main Working Focus`，其独立 Prompt 也缺少同一开度语义。模型于是把同一个 `GRIP 0.325/0.785`
先解释为可能接触，下一轮又解释成仍然张开；同时把固定 source crop 的遮挡变化误当成物体移动。

v20 把这类物理因果边界从长 Prompt 中提升为当前 Context 的一等但非真值状态。每次成功的
close/open/arm commit 分别编译为 `closure/release/arm_motion · UNVERIFIED`，并给出仍需的视觉
证据；它不判断抓取、释放或运输成功。Review 继续接收 Main 发起本次 Preview 的 overwrite-only
短意图，右侧 Waypoint 栏显示最近效果与证据缺口；固定 source 图明确标注 occlusion possible。
这不是 phase machine、自动动作建议或 privileged verifier，只是让 Main/Imagination 在同一个
当前 observation 上共享一致的证据语义。版本更新为 Web schema 21 /
`vaw-context-v20-physical-verification` / renderer `context-web-v20-physical-verification`。

冻结 seed 0 的 v20 trace `m154u_qwen35plus_physical_verification_t0_s0` 证明该语义修正确实改变了
决策：一次 close 后 `GRIP=0.388` 时，Main 不再把中间开度直接解释成成功或失败，而是创建并
commit `base +Z 3 cm` 的可逆随动核验；新 RGB 证明罐头未随动后才恢复。与此同时，该 trace
暴露了剩余的通用控制缺口：Imagination 在调用 rotate 前只能看到固定 BASE 轴，TARGET TOOL 轴
和正角方向只有编辑后才部分可见，Function 描述又允许 ±90°，导致一次无证据的 `base-X -90°`
猜测并破坏原本可规划的 grasp seed。v21 因此不改变 Function 或控制语义，只把 `LOCAL 3/4` 的
固定 `BASE / WORLD` +轴和 `JAW PLANE` 的随目标旋转 `TARGET TOOL` +轴同时放入角落，并规定
rotate 为所选 +轴的右手定则；不确定时先用 5–15° Preview，超过 30° 必须有明确的大姿态差异。
版本更新为 Web schema 22 / `vaw-context-v21-rotate-guide` / renderer
`context-web-v21-rotate-guide`。

v21 frozen seed 0 `m154r_v21_rotate_guide_qwen35plus_t0_s0` 中，Imagination 的首次无依据
大旋转从旧版 `-90°` 收敛为 `base-Y -15°`，证明小角度 Rotate 引导有效；但 Main 把
`approach_z≈-1` 的 seed 错写成 side-grasp goal，后续微调忠实执行了错误语义，最终
`env_success=false`。v22 因此在每张 seed 卡直接显示精确 `APPROACH BASE [x,y,z]`，同时明确
仅 arm approach 后物体留在原处是预期结果，不能在尚未真实闭合时叙述为“抓取失败”。版本更新
为 Web schema 23 / `vaw-context-v22-seed-approach` / renderer
`context-web-v22-seed-approach`。

v22 seed 1 的前 14 turns 证明 approach 标签生效：Main 正确把 `approach_z≈-1` 的 s3 解释为
top-down；但 scalar `TCP→SOURCE` 从 3 mm 增至 18/27 mm 时，Main 与 Imagination 仍凭相机
“上下”连续执行 base -Z，直至 collision error。v23 删除这个无方向标量，Packet 只携带从 target
TCP 指向最近当前 source surface 的 `TCP→SOURCE BASE [dx,dy,dz]`；Canvas 由该向量派生距离，
Prompt 要求沿同号 BASE 分量小步试探，并逐轮确认向量范数缩小。版本更新为 Web schema 24 /
`vaw-context-v23-source-vector` / renderer `context-web-v23-source-vector`。

v23 的真实短 probe 进一步证明，仅给当前 source vector 仍不足以支持无 transcript 的逐步比较：
下一轮看不到上一版 target，模型会在向量翻转后继续沿旧方向编辑。v24 因此收紧 Agent 所有权：
Main 只能通过 `select/propose_pose/start_imagination` 创建空间控制会话，Review 只能
`commit/reject` 或换起点，`delta_move/rotate` 仅属于 Imagination；`open/close` 后续收敛为
Main 直接创建的 gripper-only Review，不再进入空间 Imagination。进一步删除
把同一 target 退回局部编辑的路径，避免 Main 与 Imagination 成为重复几何裁判。
每次空间编辑后的 Contact Focus 同时显示青色 previous Preview 与紫色 current Preview；当时按需
显示的三轴旋转环现已由 v29 的单轴正负对照替代。Main、Review、Imagination 的 Prompt 顶部共享
抓取接近前 open、校准时保持 open、运输时 closed、释放前检查当前真实目标区域的因果前提。
active target 与 near-field presentation 选择逻辑从巨型 `packet.py` 拆入独立 presenter。版本更新
为 Web schema 25 / `vaw-context-v24-imagination-agent` / renderer
`context-web-v24-imagination-agent`。

v24 的单个正交 Contact Focus 仍会把沿观察方向的偏移压成同一投影，且原始 point-splat 在
novel view 中产生明显孔洞。v25 将右侧局部证据改为与 target tool 对齐的双正交视图：
`CONTACT FRONT · TOOL Y-Z` 观察两指闭合通道，`CONTACT SIDE · TOOL X-Z` 暴露 FRONT 隐藏的
前后与高度偏差。两张图共享同一当前 agentview+wrist RGB-D、目标坐标系、尺度与紫色 Preview。
RGB-D presenter 同时改为按相邻 sensor pixel 构造纹理三角面，并在深度断层和异常长边处断开，
以得到连续但不虚构遮挡面的 surface。Canvas 仍固定 `1920×1080`，右侧 rail 经视觉复核扩为
768 px；坐标提示只绘制当前正交平面中的两根轴，观察方向独立写为 `LOOK +axis`，避免三个轴
投影到同一点产生标签堆叠。版本
更新为 Web schema 26 / `vaw-context-v25-dual-contact-surface` / renderer
`context-web-v25-dual-contact-surface`。

v26 不再把 16:9 当作约束，固定 viewport 改为 `2048×1280`。上层使用 540 px，下层使用
733 px；imagination scene 与双 Contact raster 分别原生输出 `1200×720` 和 `832×720`，避免
仅放大 DOM 后产生 letterbox。Canvas 外边距、上下层间隔、卡片间隔和主要边框全部收敛到
`1–4px`，新增像素只服务当前真实世界与动作想象。版本更新为 Web schema 27 /
`vaw-context-v28-visual-contact-only` / renderer
`context-web-v28-visual-contact-only`。Contact Camera 在每次 Imagination 开始时锁定水平朝向，
WORLD +Z 始终朝上；后续 rotate 只改变虚拟夹爪，不反向旋转当前 RGB-D 点云或支撑面。

v29 将 Contact View 从 agentview/wrist RGB-D 的 novel-view 重投影改为两张 episode-private
MuJoCo Contact Camera 直接渲染。相机以初始 Imagination target 为中心、WORLD +Z 朝上且在
session 内锁定；每次截图只临时修改 `frontview/sideview` 相机并在 `finally` 恢复，不推进仿真。
这消除了深度断层和遮挡造成的重投影孔洞，使当前场景达到 agentview 同源的密集 raster 质量；
同时它是 simulation-only active sensor，不得在实验中描述为普通 Canvas 重排。旧的物体中心
三轴旋转环被删除，`show_rotation_gizmo(frame, axis)` 改为 Contact View 独立右栏中的单轴
`−10° / +10°` 真实夹爪姿态对照，不遮挡接触证据。版本更新为 Web schema 30 /
`vaw-context-v29-direct-contact-camera` / renderer `context-web-v29-direct-contact-camera`。

v30 将夹爪控制从空间 Imagination 中彻底分离。`open_gripper/close_gripper` 现在只出现在普通
Main Function 面，直接创建一个 pose-free、gripper-only ActionReview；它不调用 controller、
不刷新 observation，也不进入 Imagination。下一轮 Main 从当前真实 Contact Front/Side 与全局
关系审查后显式 `commit/reject`。Imagination 只保留 `delta_move/rotate/rotation gizmo/finish`，
空间 Preview 始终继承当前真实开度，避免局部姿态优化与接触动作在同一会话互相污染。Context
中的纯夹爪 Review 只公开目标开度、`NOT EXECUTED` 和接触/动力学未知边界，不伪造 arm motion
或抓取/释放效果。版本更新为 Web schema 31 / `vaw-context-v30-main-gripper-review` / renderer
`context-web-v30-main-gripper-review`。

v31 收敛 Contact View 的控制提示。此前三个 BASE 平移轴被投影到同一个二维原点，当其中一轴
接近相机光轴时会出现箭头、标签重叠和伪透视歧义。新版本根据每张 Direct Contact Camera 的
标定保留两个最具屏幕可见性的 BASE 正轴，并将最接近视线方向的第三轴独立显示为
`DEPTH +axis IN/OUT`；`ROTATE BASE` 保持为固定斜视的三维右手正向控制图例，与 scene projection
明确分离。该修改改变了策略可见 raster 语义但不改变 Function 或 Context 字段，因此版本更新为
Web schema 32 / `vaw-context-v31-separated-control-guides` / renderer
`context-web-v31-separated-control-guides`。

v33 在 Direct Contact Camera 上加入 visibility-aware session selection。每个新的
`refine_action` 私下渲染四组候选正交 RGB-D，相机比目标高 12 cm 并朝目标俯视；presenter 用
当前 region 的 sensed point cloud 投影与候选 depth 判断被更近表面遮挡的比例，优先选择两张图
中较差者仍清晰的方位。最终只把选中的两张 RGB 送入 Canvas，depth、point cloud 与选择分数
均不进入策略上下文。同一 refinement 及其 Main review 复用固定相机对，连续 `delta_move` /
`rotate` 不再触发视角变化；新的 refinement、Action 替换或物理 revision 才重选。该修改不改变
公开 Function、Context schema 或 renderer 名称。

验收：主任务 seeds `0,1,2` 至少 `2/3` env success。

### M1.5.5 — Basic Generalization and Freeze

- 在另一个未调试的基础 pick-place task 上测试；
- 冻结 schema、Prompt、Function、renderer 和模型配置；
- 更新 README、Milestone 表和最终测试报告。

验收：新增任务至少一个 seed 成功；无任务专用分支；完整离线回归、Web build 和真实 trace
可复现。

## 9. 测试矩阵

| 层级 | 必须证明的内容 | 证据 |
|---|---|---|
| Model | revision 生命周期、target/edit/review/last-action 状态 | unit tests |
| Function | preview 无物理副作用、commit-only physics、cached plan 一致 | fake backend |
| Context | 无隐私泄漏、请求形状稳定、无 transcript history | message snapshot tests |
| Canvas | 2048×1280、五 seed 完整、metric focus、delta/rotate 可辨识 | deterministic PNG tests |
| Agent | Imagination 能 accept/edit/fail，Main 能 commit/revise | scripted/fake provider tests |
| Static VLM | 无选项判断局部缺陷和下一动作 | frozen diagnostic set |
| Real | 基本 pick-place env success | LIBERO-PRO traces + evaluator |

## 10. 版本实验记录

每个实现版本必须在改代码前写清假设，完成后补充 commit 与证据。一次只验证一个主要系统假设。

| 版本 | 假设 | Commit | 离线证据 | 真实 Trace | 结果/下一失败 |
|---|---|---|---|---|---|
| M1.5.0 | dense world view 可作为清晰视觉基线 | `91f9d17` | 12 tests + Web build | `via_canvas_qwen35plus_t0_s1` | world 清晰；local control、handoff、continuity 失败 |
| M1.5.1 | metric Contact Focus 能让 2–3 cm/小角度修正可读 | `5d056f0` | 12 packet/agent tests + Ruff + Web build | `m151_v11_control_focus_scripted_t0_s1` | 五 seed 完整；3 cm 蓝紫分离和 5° 新旧轮廓可读；开放式静态 VLM probe 待完成 |
| M1.5.2 | 最小 edit memory + neutral review 能结束振荡并促成 Main review | `07f0512` | 相关 runtime/packet/agent 回归 + Ruff + Web build | `m152_qwen35plus_review_t0_s1` | 首次 Imagination 一次修正后主动 ready，Main 审查并 commit；随后暴露 post-commit 因果丢失，进入 M1.5.3 |
| M1.5.3 | one-shot physical continuity 能避免 commit 后任务重启 | `07f0512`, `ed590bf` | 40 full VAW tests + Ruff + Web build；post-commit fixture 为 `1920×1080` | `m153_post_commit_scripted_t0_s1`, `m153_qwen35plus_post_commit_t0_s1` | arm commit 后 Main 直接进入 close preview；arm+gripper 后提出 lift 核验；被拒绝的 Function 不再提前消费因果画面。下一首要失败是 detection 的语义错配，而非 post-commit 任务重启 |
| M1.5.4-a | 2× semantic render 能稳定区分相似容器 | in progress | high-resolution mapping/cache/revision tests | `m154_hires_grounding_scripted_t0_s1`, `m154_qwen35plus_hires_grounding_t0_s1` | identity grounding 稳定正确；下一失败转为错误姿态解释与接触不可观测 |
| M1.5.4-b | 去除 quaternion 控制歧义后 Imagination 不再无依据旋转 | in progress | 49 full VAW tests + Ruff + Web build | `m154_qwen35plus_control_semantics_t0_s1` | 旋转错误消失并产生真实接触；单投影仍造成 Z 振荡，加入正交 JAW PLANE 与 rejected source handoff 后待真实复测 |
| M1.5.4-c | 正交 Contact Focus 能收敛局部接触；revision-local cause + one-decision review 防止任务重启与 stale commit | in progress | 52 full VAW tests + Ruff + Web build | `m154_qwen35plus_contact_focus_t0_s1`, `m154c_source_guard_scripted_t0_s1`, `m154c_qwen35plus_causal_t0_s1` | 局部调整明显收敛且首次真实抓起；因果、stale review、source outlier 与 gripper-only Canvas crash 均已修复并回归 |
| M1.5.4-d | 精确终点 settle + 幂等 grounding + local/semantic motion 分工能把闭环推进到可靠 place | in progress | 56 full VAW tests + Ruff + Web build | `m154d_qwen35plus_settle_t0_s1` | pick 与 3cm lift 真实成功；首次 place 因未 grounding basket、把 delta 当长距离导航而落在篮外；策略语义已修订，待复测 |
| M1.5.4-e | 当前 Review 应覆盖旧物理失败，避免两个因果焦点竞争 | in progress | packet/message scope tests + Ruff | `m154e_qwen35plus_destination_t0_s1`, `m154f_qwen35plus_review_scope_t0_s1` | destination recovery 已使用 region-scoped point；隐藏旧 failure 后 Main 能审查新 point-based action。下一失败来自 executor 耗尽 LIBERO horizon |
| M1.5.4-f | 有界 waypoint tracking + environment termination 传播能保留真实闭环预算 | `6e7313d` | safe `+3 cm Z` A/B：81 steps/8.43 mm → 43 steps/6.13 mm；terminal callback regression | `m154g_qwen35plus_bounded_exec_t0_s1` | 7 次 physical commit 未耗尽 horizon，executor 修复成立；下一失败转为 tool/base 语义与 review edit 丢失 |
| M1.5.4-g | frame 因果语义 + review command-state continuity 能阻止方向相反的 target 被批准 | in progress | 58 full VAW tests + Ruff + Web build；handoff cumulative-edit packet/message tests | `m154h_qwen35plus_review_edit_t0_s1` | base/world 方向语义生效，不可执行 target 未被交回；该 run 未产生可执行 review，累计 edit 的真实 Main 审查仍待覆盖 |
| M1.5.4-h | 工具 capability routing 能优先使用完整 grasp seeds，并停止用幂等 perception 恢复 IK | in progress | Function order/description contract tests + targeted 37 tests + Ruff | pending agent rerun | 不增加 phase/gate；待验证 Main 使用 `region → propose_grasps` 而非未尝试 seed 就手写 quaternion |
| M1.5.4-i | Main 只在独立 Review 决策面批准动作，普通决策不会遗留 stale review | in progress | Review tool-surface、reject、packet 与 runtime 回归 | `m154k_qwen35plus_review_contract_t0_s1` | Main 能 commit/reject/revise，未再用 detection 隐式跳过 review；发现 7-D joint L2 对小分量误差的重复放大 |
| M1.5.4-j | overwrite-only Main Working Focus 能跨一次非物理调用维持失败结论 | in progress | 单 focus 覆盖/隔离测试；无 transcript/history | `m154l_qwen35plus_per_joint_t0_s1`, `m154m_qwen35plus_working_focus_t0_s1` | 抓持随动失败后，下一轮保留“失败并重试”而没有转向 basket；发现 gripper-only review 加空间编辑时丢失 gripper target，已修复 |
| M1.5.4-k | Review 编辑必须保留完整 arm+gripper target，终点验收不应随关节维数人为收紧 | in progress | 63 full VAW tests + Ruff + Web build | `m154n_qwen35plus_preserve_target_t0_s1` | target 保真修复通过；真实失败 target 的 achieved TCP 仍偏差约 3.8 cm，正确拒绝并跳过 gripper。恢复随后陷入无可行 seed/手工侧抓 IK error，主任务仍未成功 |
| M1.5.4-l | Action Review 应审查局部 refinement goal，而不是要求每个动作直接完成 User Task | in progress | Review prompt contract regression | `m154o_qwen35plus_review_focus_t0_s0` | pick/lift 失败被正确识别；Review 却以“没有抓住/没去篮子”为由否决必要的纯 open 和 side-grasp approach。补充通用 prerequisite/局部动作审查原则后待复测 |
| M1.5.4-m | gripper-only Preview 之后的空间编辑必须以启动时真实 TCP 为累计位移基线 | in progress | private baseline / cumulative EditSummary regression | `m154p_qwen35plus_local_review_t0_s0` | Imagination 实际累计 base Z `-9.5 cm`，但 Review 只看到最后一步 `-2 cm` 后错误 commit；根因是 gripper-only target 无 pose 时私有 baseline 缺失，修复不改变公共 target 的 gripper-only 语义 |
| M1.5.4-n | grasp seed 必须被解释为最终接触目标，并提供当前 source surface 的可视距离证据 | in progress | 64 full VAW tests + Ruff + Web build；同任务无物理 contact probe：`TCP→SOURCE=1.6 mm` 后不再自动上抬 | `m154q_qwen35plus_cumulative_review_t0_s0` | 旧图仅换 Prompt 仍上抬 2.5 cm；v17 probe 改为一次姿态微调，证明 contact metric 有效但尚未证明真实闭环成功，进入 frozen seed 复测 |
| M1.5.4-o | 非物理 grounding 不得抹去同一 observation 中刚验证的物理因果视觉 | in progress | revision-local post-commit raster persistence / grounding precedence / Web regression | `m154r_qwen35plus_contact_semantics_t0_s0` | 首次真实抓持与 +3cm 随动成功；basket detection 后核验画面消失，Main 在同一 RGB 上反转结论并重抓手中物体。新增紧凑 current-observed continuity inset，待 frozen seed 复测 |
| M1.5.4-p | Main 必须看见当前 ActionReview；抓持核验必须显式比较同一 source 原位置 | in progress | 65 full VAW tests + Ruff + Web build；review-priority、causal-source persistence、固定 ROI 与 `1920×1080` fixture | `m154s_qwen35plus_physical_continuity_t0_s0`, `m154t_qwen35plus_causal_verification_t0_s0` | v19 能走到 release，但 Action Review 丢失 Main 短意图、固定 crop 受遮挡、部分开度解释反转，最终 `env_success=false` |
| M1.5.4-q | 物理命令效果必须以一致的 UNVERIFIED 状态跨 Main/Review 所有权传播 | in progress | 65 full VAW tests + Ruff + Web build；closure/release/arm-motion cue、Review focus 与 `1920×1080` fixture | pending v20 frozen seeds | 不加入真值或 phase；待证明 arm+close 后不会因开度歧义跳过随动测试 |
| M1.5.4-r | rotate 必须在调用前具有可读的 BASE/TOOL frame、轴与符号依据 | in progress | pending v21 unit/snapshot/static diagnostics | `m154u_qwen35plus_physical_verification_t0_s0` | v20 已产生正确随动核验，但一次无证据 `base-X -90°` 破坏 grasp；v21 只增加坐标指南和小角度探索语义，不加入任务分支或角度 clamp |
| M1.5.4-s | Main 必须从精确 BASE approach 解释 seed，且不能把纯 arm approach 叙述为抓取失败 | in progress | pending v22 unit/Web/frozen seed diagnostics | `m154r_v21_rotate_guide_qwen35plus_t0_s0` | v21 首次 rotate 收敛为 15°，但 Main 将 approach_z≈-1 的 seed 误称 side grasp；point-pose approach 后又在未闭合时错误宣称抓取失败，主任务仍为 `env_success=false` |
| M1.5.4-t | 接触微调必须获得带 BASE 方向的 source surface correction，而非孤立距离 | in progress | 67 full VAW tests + Ruff + Web build + deterministic direction fixture | pending v23 seed-1 control probe | v22 正确识别 top-down seed，但 `TCP→SOURCE` 从 3→18→27 mm 时仍连续 base -Z 并撞入 collision；v23 用唯一 BASE delta 向量替换标量，不加入任务分支 |
| M1.5.4-u | Main/Imagination ownership、Direct Contact Camera 与 gripper-only Review 应形成统一当前规范 | in progress | current architecture/code contract + VAW regression | `m154u_v24_imagination_agent_qwen35plus_t0_s1` 及后续 v29/v30 traces | 双 Agent ownership、history-free Context、Direct Contact Camera 与 Main-only gripper Review 已落地；真实任务成功率门槛仍未达到 |
| M1.5.4-v | Contact 控制提示必须在二维相机中无歧义表达三个 BASE 平移轴和三维旋转正向 | in progress | schema 32 packet/Web contract + deterministic Contact guide tests | pending frozen-seed rerun | 两个屏幕内平移轴与独立 DEPTH IN/OUT 已分离；renderer 元数据升级，避免与 v30 trace 混淆 |
| M1.5.4 | 完整闭环可达到基本 pick-place 成功 | in progress | 65 full VAW tests + Ruff + Web build | pending frozen seeds 0/1/2 | 尚未达到 `2/3 env_success`，不得宣称完成 |

## 11. 非目标

M1.5 不加入：

- pick/place phase machine 或任务专用动作序列；
- privileged success、object pose、PDDL 或 reward 注入；
- 跨 episode 或无限场景历史；
- 通用 object tracking/Scene Memory；
- 新的 grasp network、VLA policy 或 RL trainer；
- GUI 点击、按钮 action space 或 DOM 操作；
- 以 planner success、TCP error、夹爪 opening 或文字声明替代视觉任务效果。

Scene Memory、训练数据采集和 RL 只在 M1.5 的基本 Agentic 闭环通过后进入下一阶段。
