# ARCHIVED — VAW M1.3.2 Visual Density 设计稿

> **历史设计冻结，未按本文形态实现，已被 M1.5 架构取代。** 当前 Canvas、Function ownership
> 和 Context Builder 见 [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md)。以下
> `1440×1080`、wrist RGB、K=8 和旧 renderer 内容只用于解释设计演进。
> 状态：设计冻结，尚未实现  
> 目标 renderer：`context-web-v2-dense`  
> 输出：固定 `1440×1080`、DPR=1

## 1. 目标

M1.3.2 将 Dynamic Context Canvas 从“同时展示视觉证据和协议说明”收敛为真正的 Visual Action Workspace：绝大多数像素用于当前真实 RGB、当前 evidence、Action imagination 和必要的机器人本体状态。

本轮重点解决两个问题：

1. Task、manifest、Function History 和 receipt JSON 已通过文本消息提供，却又被重复绘制进 raster，挤占视觉空间。
2. 当前 receipt 页面以绿色、大面积卡片和低密度文字强调“命令执行成功”，容易诱导 VLM 将 command execution 错误理解为 task effect verified。

M1.3.2 不通过环境真值替 Agent 判断任务是否完成。它只让动作后的当前视觉后果更容易观察，并始终区分：

```text
COMMAND EXECUTED ≠ TASK EFFECT VERIFIED
```

## 2. 信息分工

### 2.1 Raster 保留的信息

- 当前 revision 的 `agentview` RGB。
- 当前 revision 的 wrist RGB。
- region、point、candidate 等当前视觉 evidence。
- Action Proposal 的局部交互 focus、真实 gripper geometry 与 IK/FK imagination；TCP
  只用于内部投影和 trace，不在默认 policy raster 中绘制孤立十字。
- 紧凑的机器人本体状态。
- 动作后当前目标区域的视觉核验 crop。
- 一行不可替代的 physical event 摘要。

### 2.2 不再绘制进 Raster 的信息

- VAW logo、产品名称和全局标题栏。
- Task Prompt；它已经作为独立 User 文本传给策略。
- `vaw-context-v2`、viewport、schema version 等协议元数据。
- `01 / 02` 编号和双语章节说明。
- `RESULT-DRIVEN · NOT A PHASE GATE` 等 System Prompt 内容。
- footer，包括“不是 GUI action space”等项目说明。
- `VALID IDS`；有效引用已经存在于 minimal manifest。
- 完整 receipt JSON、完整错误 JSON 和 revision 生命周期说明。
- Agent 后续不会作为参数引用的 `receipt_id`。

Function result 和完整错误仍保留在 K=8 Function History 与 trace 中；receipt ID 仅保留在 trace。

## 3. 固定 Canvas 与动态布局

Canvas 始终为 `1440×1080`，无滚动、动画、hover-only 信息或响应式重排。布局只根据 `decision.mode` 在少量固定模板之间切换。

### 3.1 Evidence-heavy 模式

适用于 `grounding / candidates / proposal`：

```text
┌──────────────── Persistent World Context：约 52% ────────────────┐
│              clean agentview              │ clean wrist          │
├──────────────── Dynamic Decision Workspace：约 48% ─────────────┤
│ robot state + grounding / focused candidates / imagination       │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 Observation-heavy 模式

适用于 `receipt / idle / terminal`：

```text
┌──────────────── Persistent World Context：约 78% ────────────────┐
│              clean agentview              │ clean wrist          │
├──────────────── Post-action / Quiet Workspace：约 22% ──────────┤
│ robot state + current verification crop + compact event strip    │
└───────────────────────────────────────────────────────────────────┘
```

`error` 使用独立的固定恢复模板，保留当前 RGB 和简短错误摘要；错误不会被渲染为 receipt success。

### 3.3 Persistent World Context

- agentview 始终占最大视觉面积，用于判断物体、容器、支撑面和全局空间关系。
- agentview 与 wrist 都是无 region、point、self-mask 或 imagination overlay 的当前原始 RGB。
- wrist 只用于补充局部接触、夹持和释放；本体状态移到 Dynamic Decision Workspace 顶部。
- 视角标签只保留紧凑的 `AGENTVIEW` 与 `WRIST`，不保留解释性副标题。
- 不恢复 `EE now` 图像标记。

本体状态固定为四行，不使用多个小卡片：

```text
GRIP  0.986
TCP   [0.681, 0.246, 0.376]
QUAT  [0.851, 0.524, -0.027, -0.006]
JOINT [0.227, 0.640, 0.166, -1.304, -0.102, 1.880, 0.079]
```

- 状态值使用约 `15–16px` monospace。
- gripper 只显示连续的 normalized opening 数值，不在 raster 中用阈值派生 `OPEN / CLOSED`。
- 数值允许合理减少显示精度，但不得省略七个关节角或改变单位和 quaternion 顺序。

### 3.4 Focused Action Preview

- candidate crop 由 `source region ∪ target gripper` 决定，整条机械臂 mask 不参与裁剪范围，
  避免把对象和手爪缩成难以比较的小目标。
- proposal crop 由 source、reference/target pose 和 target gripper 共同决定；局部 focus 为主，
  小 inset 保留整臂构型。
- predicted arm 使用约 `45%` 紫色混合，predicted hand/fingers 使用约 `80%` 紫色混合并保留
  实色轮廓，使 VLM 优先观察两指、对象和邻近障碍物关系。
- `delta_move` 使用前后 gripper 与绿色位移箭头；`rotate` 使用前后 orientation axes；不绘制
  孤立的 `TCP target` 十字或标签。

## 4. Receipt 重构

### 4.1 紧凑事件条

Receipt 不再占用两张大卡片。事件条只显示当前决策所需且未在本体状态中充分表达的信息。

`commit` 示例：

```text
COMMIT a2 · TCP error 6.5 mm · TASK EFFECT UNVERIFIED
```

gripper 示例：

```text
OPEN_GRIPPER · opening 0.986 · TASK EFFECT UNVERIFIED
```

规则：

- `position_error_m` 必须标记为 `TCP error`，不能显示成泛化的 success/error。
- receipt 使用中性灰或蓝色；`TASK EFFECT UNVERIFIED` 使用琥珀色。
- 绿色不用于表示物理命令已经完成。绿色仅保留给当前视觉证据支持的 selection/target 语义。
- 不显示完整 JSON、`receipt_id`、空的 valid-ID 条或 revision 生命周期说明。
- execution discrepancy 或 backend error 进入 `error` mode，而不是绿色 receipt。

### 4.2 动作后视觉核验

Receipt mode 的主要内容是最新 observation，而不是执行报告。

- `commit(action_id)` 前保存该 Action Proposal 的 target TCP，作为最近 spatial target 摘要。
- commit 刷新 observation 后，将该 target 投影到新 revision 的 agentview。
- 以投影点为中心，从当前 agentview 生成固定尺度的 post-action crop；投影点只决定裁剪位置，
  raster 不显示 `LAST TCP TARGET` 十字或标签。
- `open_gripper()` 或 `close_gripper()` 不覆盖最近 spatial target；gripper 动作刷新 observation 后，继续用最新 RGB 生成同一目标附近的核验 crop。
- 下一次 commit 替换最近 spatial target；episode reset 清空它。
- 最近 spatial target 是动作摘要，不是 object tracking、永久 object identity 或世界真值。

核验 crop 必须：

- 只来自当前 revision RGB，不拼接旧 observation。
- 不恢复已经失效的 region、point、candidate 或 action ID。
- 不包含 depth、intrinsics、camera pose、raw mask/cloud、reward 或 privileged success。
- 在 target 无法投影、位于相机后方或 crop 无有效面积时，回退为当前完整 agentview。
- 没有最近 spatial target 的初始 gripper receipt 也回退为当前 agentview，不虚构目标区域。

`DecisionWorkspaceSpec.primaryRasterId` 复用为 post-action raster 引用，例如 `decision:post_action`；不复制 raster 数据到 receipt metadata。

## 5. 数据与兼容边界

- Agent-visible Function 保持 M1.4 当前十一个，名称、参数与返回 schema 均不变化。
- `solve_ik`、motion backend、revision invalidation、K=8 History 和 provider loop 均不变化。
- 保持 `vaw-context-v2` 和 Web `schemaVersion: 3`。
- ContextPacket 继续通过现有 `primaryRasterId` 引用 post-action raster，不增加策略必读字段。
- Runtime/Compiler 只增加最近 spatial target 的 episode-local 摘要，用于当前 RGB 投影。
- Renderer 名称更新为 `context-web-v2-focus`，确保 trace 可以区分旧版密度布局与新的
  clean-world / focused-action 图像。
- 既有 trace 作为实验产物保留；未冻结的 legacy M1.2 renderer 已在后续代码收敛中删除，
  历史实现由 Git checkpoint `fd8d89a` 保存。

本轮不再加入：

- M1.4 十一个 Function 之外的新工具或 place macro；
- relation verifier 或 environment-truth gate；
- object tracking、Scene Memory、PDDL、OBB；
- trajectory/collision planner；
- 新的 phase、allowed-tools 或硬编码任务流程。

## 6. 测试与验收

### 6.1 视觉与确定性

- 输出严格为 `1080×1440×3`，DPR=1。
- 同一 ContextPacket 重复截图得到字节级一致 PNG。
- 页面无滚动、动画或布局溢出。
- 本体状态四行完整可读，七关节角不截断。
- 不再出现品牌栏、Task 重复文本、footer、`01/02`、`VALID IDS`、完整 receipt JSON 或协议说明。

### 6.2 Receipt 行为

- commit receipt 显示当前 target crop、action ID、TCP error 和 `TASK EFFECT UNVERIFIED`；
  revision 只存在于文本 manifest，不在 raster 重复显示。
- gripper receipt 使用最新 RGB，并复用最近 spatial target。
- 新 commit 正确替换 spatial target，reset 正确清空。
- 无 target、target 越界、target 在相机后方时稳定回退当前 agentview。
- physical error 进入 error mode，不显示为绿色 receipt。

### 6.3 隐私与协议回归

- post-action raster 与 packet summary 不泄露 depth、intrinsics、camera matrix、raw mask/cloud、reward 或 environment success。
- Function 列表保持 M1.4 十一个；Agent-visible result 不含 `receipt_id` 或重复 revision，manifest
  不含 renderer/schema 元数据。
- `decision_basis` 只进入 trace，K=8 History 只回放 structured call/result。
- K=8 History、revision 失效、active action 生命周期和 trace 记录保持兼容。
- 当前 schema-v3 `1440×1080` renderer 回归通过，且代码中不存在 schema-v1 分流。

### 6.4 真实案例验收

使用 `m131_dynamic_qwen35plus_visual_prompt_20260803_v2` 中失败的 `context_0011.png` 与 `context_0012.png` 状态生成 M1.3.1/M1.3.2 对照：

- 篮子与 alphabet soup can 的当前关系获得显著更大的有效像素面积。
- receipt 不再以绿色卡片暗示任务成功。
- 打开夹爪后的画面明确保留目标区域核验，并显示 `TASK EFFECT UNVERIFIED`。
- 人工检查能够直接看出 can 位于篮子外侧，且无需读取完整 receipt JSON。

## 7. 完成定义

当 VLM 接收到的 raster 中，协议性文字只保留一行物理事件摘要，而当前双相机画面、紧凑本体状态和动作后目标区域成为主要视觉内容时，M1.3.2 视觉密度重构完成。

该改造只改善证据组织与视觉核验条件，不宣称消除 VLM hallucination，也不把 command receipt 升格为 task success 证明。
