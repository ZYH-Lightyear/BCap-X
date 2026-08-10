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
- revision-local region/point/seed 不跨物理动作复用，但自然语言 focus 与最近一次物理意图可以
  跨一次 commit 保留；
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
- 在下一次 Main 决策后消费；
- 不判断 task effect，不替代当前真实视觉；
- 可同时携带一次 before/current 视觉对照，但两张图必须在同一 Canvas 中呈现。

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
review_required(action_id)
failed()
```

- Imagination 主动认为足够合理时产生 `review_required`；
- 达到 turn limit 时也产生 `review_required`，但 `termination_reason=turn_limit` 仅写 trace；
- Main 不能仅根据 handoff 原因 commit，必须看最终 Preview；
- Main 也不能把达到编辑上限自动解释为失败；
- 若 target 没有可执行 motion plan，必须产生 `failed` 或显式 planner error，不创建可 commit
  review。

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

状态：已完成。真实模型 trace 中，Main 在 arm commit 后直接继续创建 close-gripper preview，
没有重新从 detection/propose 启动任务；在 arm+gripper commit 后，也能够从当前真实画面提出
小幅 lift 来核验抓持。Function 参数被 runtime 拒绝时不会消费这条一次性因果上下文，只有
成功的 Main Function 才会消费；commit 则原子地用新物理动作替换旧记录。

### M1.5.4 — End-to-End Pick-and-Place

- 在 `libero_object_swap:0` 完成 pick、lift verification、transport、place 和 release；
- 修复只由真实 trace 证明的 Function/backend/Context 缺陷；
- 每次真实 run 记录冻结配置、commit、trace 和第一失败原因。

当前第一失败原因审计：`m153_qwen35plus_post_commit_t0_s1` 的初次
`detection_and_sam("alphabet soup can")` 把前景红绿罐误识别为任务目标。SAM mask、对象点云、
camera-to-base 变换与 grasp target 都与该错误 box 内部一致，因此不是坐标转换错误。根因是底层
detector 被要求强制返回单个 box，却没有被要求在多个同类容器间做精确语义消歧。

修复保持 Function schema 和 CaP-X 不变：VAW 在 detector 边界私下将简短 query 扩展为“精确
语义目标；依据可见属性与关系区分其他对象/区域；不得因为更近或更大而选择 generic match”。
Region 继续保存并向 Agent 展示原始简短 query。固定真实帧 probe 中，旧提示返回错误 box
`[392,302,444,372]`，增强提示返回正确 box `[342,201,376,253]`；真实 scripted trace
`m154_semantic_grounding_scripted_t0_s1` 使用原始无冠词 query 返回 `[342,201,375,252]`，且完成
SAM、point lift、grasp seed 与 CuRobo 链路。该 smoke 只证明 grounding wiring，不计入任务成功门槛。

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
| M1.5.4 | 完整闭环可达到基本 pick-place 成功 | in progress | semantic grounding regression + full VAW regression | `m154_semantic_grounding_scripted_t0_s1` | 首个真实失败源已从“错误目标 box”修复为可复现的精确语义 grounding；下一步重新运行真实 Agent，定位新的第一失败点 |

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
