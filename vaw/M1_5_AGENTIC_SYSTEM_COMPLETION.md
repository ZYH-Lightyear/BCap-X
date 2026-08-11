# VAW M1.5 — Agentic System Completion

## 1. 目标

M1.5 的目标不是继续优化一张好看的 Canvas，而是完成一个可验证的 Visual Action
Workspace Agentic System：

> Main Agent 依据当前真实视觉决定任务级动作，Imagination Agent 在清晰、可控的局部视觉中
> 收敛一个虚拟动作，Main 审查后 commit；物理执行后，Main 根据新的真实状态继续当前目标或
> 进入下一个动作，最终在真实 LIBERO-PRO 中完成基本 pick-and-place。

Canvas、Function 和 Prompt 都只是该闭环的组成部分。任何单独截图、IK 成功、轨迹生成、
夹爪闭合、Agent 声称成功或 scripted smoke 都不能证明 M1.5 完成。

本文是 M1.5 的权威设计与验收文档。`M1_4_2_DUAL_AGENT_RUNTIME.md` 和
`M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md` 保留为历史基线与失败记录；若与本文冲突，以本文为准。

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

- `commit(action_id)` 是唯一物理 Function；
- Imagination 不直接 commit，Main 不承担逐步局部优化；
- Imagination 正常结束、达到编辑上限或失败都必须明确把控制权交还 Main；编辑上限不能被
  表达成动作失败或自动批准；
- Main 必须看到最终 Preview，并能够选择 commit、重新进入 Imagination、换 seed 或放弃；
- commit 后的新请求必须包含当前真实视觉和一条最小动作连续性信息，使 Main 知道刚执行的是
  arm、gripper 或二者，以及原始意图；
- revision-local region/point/seed 不跨物理动作复用；最近一次物理意图在当前真实 observation
  revision 内 overwrite-only 保留，直到下一次 commit 覆盖；
- Agent-visible Context 不包含 reward、environment success、privileged object pose、raw depth、
  calibration、raw mask/cloud 或 planner trajectory。

### 2.3 视觉可控性门槛

- 固定输出 `1920×1080`、DPR=1、无滚动、动画或响应式重排；
- `OBSERVED NOW` 始终只显示当前真实 observation，不叠加虚拟结果；
- `select/propose_grasps` 时最多五个 seed 必须同时完整可见，ID 与图像一一对应，不被裁切、
  遮挡或缩小到无法比较；
- Imagination 的主要面积必须是 target-centric Contact Focus，而不是再次展示全局场景；
- 在默认相机和常见工作距离下，目标物体与 target gripper 的联合包围框应占 Contact Focus
  短边的 `30%–75%`；超出范围时 compiler 必须自适应，不依赖任务专用坐标；
- 单次 `3 cm` 位移和 `10°` 旋转必须在相邻 Preview 中产生清晰可辨的几何变化；
- target 处不绘制遮挡物体的局部三轴。BASE/WORLD 或 orientation 提示只能出现在固定边角或
  独立 widget；
- 紫色只表示未执行目标，蓝色只表示当前真实机器人；不预测或暗示物体会被抓住、移动、释放
  或进入容器。

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
  └─ review final ActionReview
        │
        ▼
Imagination Agent
  ├─ inspect Contact Focus
  ├─ delta_move / rotate / gripper preview
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
Optional ActionReview
Optional LastPhysicalAction (one item, overwrite-only)
Current 1920×1080 Canvas
```

不加入 Function transcript、旧 reasoning、旧 evidence ID 或多轮图片历史。

### 5.2 Imagination 请求

```text
Imagination System Prompt
Concise refinement goal
Current ActionTarget
EditSummary
Current 1920×1080 Canvas
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
├── executed_stages: arm | gripper | arm+gripper
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
- camera-aligned dense RGB-D world surface；
- 紧凑 robot state；
- 不显示 target、region、seed 或 planner 状态。

### 6.2 Seed Selection

- 下层切换为五张等宽、完整候选卡；
- 每张只显示一个 seed 的 target gripper、局部目标几何和 ID；
- 五张卡使用统一物理尺度与视角，禁止每张独立 auto-zoom；
- 必须保留足够大的对象/手指轮廓，不显示 score 和原始 planner 数值；
- 全局整臂结果只保留一个小型 inset，不能挤占 candidate comparison。

### 6.3 Contact Focus

Imagination 下层采用固定结构：

```text
┌──────────────── Contact Focus ───────────────┬── Global inset ─┐
│ target 周围固定物理范围                       │ 整臂/障碍关系    │
│ current object surface + current/target hand │                │
│ previous target outline + current target     │                │
│ contact corridor / jaw clearance evidence    │                │
└──────────────────────────────────────────────┴────────────────┘
```

- compiler 根据 target depth 和相机标定，将默认 `20 cm` 物理范围转换为图像 crop；
- source region 对应的当前 RGB-D surface 可高亮，其他点降低亮度；
- `delta_move`：显示 previous→current target 的外置箭头、last delta 和 cumulative delta；
- `rotate`：显示新旧 silhouette 差异和独立 rotation widget，不在物体上叠加三轴；
- gripper edit：只改变虚拟指宽，并明确物体点保持当前 observation，不模拟动力学；
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
- 达到 turn limit 时也产生 `review_required`，但 `termination_reason=turn_limit` 仅写 trace；
- Main 不能仅根据 handoff 原因 commit，必须看最终 Preview；
- Main 也不能把达到编辑上限自动解释为失败；
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
- 对 Main 隐藏 `budget_exhausted`，统一为 `review_required`；
- 更新 Imagination Prompt 和 Function 描述，使 accept/edit/fail 对称。

验收：fake provider 覆盖主动 ready、turn-limit review、failed 和 planner error；真实静态 session
不再持续出现无信息的正负方向抵消；Main 能审查 limit 交回的最终动作。

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
| Canvas | 1920×1080、五 seed 完整、metric focus、delta/rotate 可辨识 | deterministic PNG tests |
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
| M1.5.4 | 完整闭环可达到基本 pick-place 成功 | in progress | 63 full VAW tests + Ruff + Web build | pending frozen seeds 0/1/2 | 尚未达到 `2/3 env_success`，不得宣称完成 |

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
