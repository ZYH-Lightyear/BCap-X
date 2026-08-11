# VAW — Visual Action Workspace

`vaw` 是面向 LIBERO-PRO 的双 Agent Visual Context Runtime。它把当前真实观测与动作想象
编译为固定 `1920×1080` Canvas，但模型不点击页面：动作通道始终是 structured Function
call。

当前架构不再回放 Function history，也不维护 Persistent Waypoint。Main Agent 负责语义决策
和物理提交；Imagination Agent 在独立、无物理副作用的会话中连续检查和微调一个
`ActionTarget`。

当前权威目标、完成定义、Agentic Context 设计和逐版本验收路线见
[`M1_5_AGENTIC_SYSTEM_COMPLETION.md`](M1_5_AGENTIC_SYSTEM_COMPLETION.md)。双 Agent 基线见
[`M1_4_2_DUAL_AGENT_RUNTIME.md`](M1_4_2_DUAL_AGENT_RUNTIME.md)；早期 20-turn 抓取审计见
[`M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md`](M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md)，二者只作为
历史失败证据，不再定义当前完成标准。

## Runtime

```text
CURRENT OBSERVED
      │
      ▼
Main Agent ── perception / ActionSeed ──► Imagination Agent
   ▲                                      │
   │                  delta / rotate / gripper preview
   │                                      │
   └──────── ActionReview / failed ────────────────┘
   │
   └── Main reviews ── commit(ActionReview) ──► physical world ──► fresh observation
```

每次 provider 请求都从零构造。Main 只看到 task、当前 Canvas、minimal policy state、最近一次
handoff，以及当前 observation revision 的一条 overwrite-only `LastPhysicalAction`；大幅
previous/current 对照只在 commit 后出现一次。Imagination 只看到 task、
refinement goal、当前 `ActionTarget` 和当前 Canvas。两边都看不到对方或自己的历史
call/result/rationale。

感知与运动有明确的信息边界。同一 observation revision 中重复相同的
`detection_and_sam` query 会复用已有 region，不会制造新的世界状态；已有准确 region 时，
`locate_point` 应显式传 `within_region_id`。`delta_move` 是每轴最多 3cm 的当前/想象 TCP
局部修正；远处语义目标应先定位 region 和 point，再通过 `propose_pose` 进入局部 Preview。

## Function space

Main Agent（普通决策面）：

```text
detection_and_sam(query, within_region_id?)
locate_point(query, within_region_id?)
propose_grasps(region_id) -> seed_ids
propose_pose(point_id, offset_xyz, refinement_goal, quaternion_xyzw?)
select(seed_id, refinement_goal)
delta_move(delta_xyz_m, frame, refinement_goal)
rotate(axis, angle_deg, frame, refinement_goal)
open_gripper(refinement_goal)
close_gripper(refinement_goal)
done(success)
```

Action Review 决策面只在 Imagination 交回一个待审动作时出现：

```text
commit(action_id)
reject_action(action_id)
delta_move / rotate / open_gripper / close_gripper
select / propose_pose
done(success)
```

普通 Main 看不到 `commit`；因此只有在最终 Preview 已进入 Action Review 后才可能执行。审查时
若调用 editor，会把同一完整 target 交回 Imagination；若调用 `reject_action`，则显式销毁该
offer，不刷新真实 observation。

Imagination Agent：

```text
delta_move(delta_xyz_m, frame)
rotate(axis, angle_deg, frame)
open_gripper()
close_gripper()
finish_imagination(status="ready" | "failed")
```

除 `commit` 外，所有 Function 都不会改变真实世界。`select/propose_pose` 以及 Main 直接调用
空间或夹爪 editor 时进入 Imagination；结束后产生等待 Main 判断的 `ActionReview`。
Main 启动 Imagination 时必须显式提供一句短的 `refinement_goal`，不能把整段 rationale 当成
局部控制目标。Imagination 每次请求只收到当前 Canvas、目标几何和累计 `EditSummary`，不收到
Function transcript。普通 Main 只额外收到一条 overwrite-only `Main Working Focus`：上一轮
Main 自己的一句依据，用于在感知调用后保留“抓取失败，正在重试”这类短期任务关系；它不是
环境真值，也不会进入 Imagination 或 Action Review。默认最多连续想象 6 轮；主动完成或达到上限都以中性的
`review_required` 交回 Main，`turn_limit` 只写 trace。只有 Main 查看最终 Preview 后调用
`commit` 才构成批准。`ActionReview` 是一次决策的 offer：Main 的下一次成功调用若不是
`commit`，旧 review 会被明确丢弃，不能在后续回合被误提交。

## Canvas

Web schema 18 / `vaw-context-v17-contact-semantics` / renderer
`context-web-v17-contact-semantics`：

- 上层 `OBSERVED NOW · REAL WORLD`：干净 agentview、与 agentview 标定透视一致的稠密
  RGB-D surface 和四行本体状态；
- `ACTION SEEDS`：最多五个候选以固定五列占满下层，统一尺度并完整显示；
- active target 时下层为 `IMAGINATION · NOT EXECUTED`：同一当前 RGB-D surface 的
  camera-aligned 全局 Preview，
  以及由当前 agentview+wrist RGB-D 编译的正交 `JAW PLANE` Contact Focus；后者用于观察目标
  物体是否真正位于两指通道，紫色 target 始终表示未执行；
- 没有 active target 时，下层明确标成 `CURRENT EVIDENCE · OBSERVED` 或
  `CURRENT GEOMETRY · OBSERVED`，不再把当前蓝色机器人误标成未执行想象；
- commit 后下层短暂切换为同一 Canvas 内的 `BEFORE COMMIT → CURRENT OBSERVED` 目标区
  对照；大图只显示一次，紧凑 `LastPhysicalAction` 在下一次 commit 前持续提供因果连续性，
  不额外发送旧图或声明任务效果；
- BASE/WORLD 坐标提示由 robot-base 几何投影产生，并固定在角落以避免遮挡 target；
- grounding、ActionSeed 与 refinement 信息只占用下层固定 overlay，不改变双层版式；
- 紫色几何只存在于下层，并始终表示未执行的预测；
- Imagination 交回 Main 后仍保留精确的初始/最终 target 与累计 base-frame 位移/旋转，避免
  ActionReview 丢失局部编辑方向；
- 无 receipt 页面、旧 Function history、Task 重复文本或 privileged state。

Depth、相机参数、raw mask/cloud、planner trajectory 和环境 success 只存在于 private context
或 trace，不进入策略消息。

## 代码结构

```text
vaw/context_runtime/
  model.py          # evidence、ActionTarget/Seed、ImaginationState、ActionReview
  private.py        # sensor、planner、source provenance、visual edit artifacts
  functions.py      # Main/Imagination 共用的无物理 editor 与唯一 commit
  workspace.py      # revision 生命周期与 dispatch
  protocol.py       # 两个独立 System Prompt 和工具视图
  runtime.py        # history-free Main/Imagination ownership loop
  packet.py         # private state → policy-visible ContextPacket
  scene_view.py     # camera-aligned dense RGB-D surface + 当前/目标实体 FK silhouette
  web_renderer.py   # fixed Playwright screenshot
  trace.py          # 完整审计记录；不回灌策略
```

## 构建与运行

```bash
cd /mnt/data/zyh/BCap-X/vaw-ui
npm run build

cd /mnt/data/zyh/BCap-X
source .venv-libero/bin/activate
python -m vaw.scripts.run_context_agent \
  --mode agent \
  --suite libero_object_swap \
  --task-id 0 \
  --seed 1 \
  --model vapi/gpt-5.5 \
  --imagination-model vapi/qwen3.5-plus \
  --protocol text \
  --max-turns 32 \
  --max-imagination-turns 6 \
  --motion-backend curobo \
  --record-video
```

省略 `--imagination-model` 时两个角色复用 `--model`，但 provider 请求和 Context 仍完全
隔离。

## 测试

```bash
source /mnt/data/zyh/BCap-X/.venv-libero/bin/activate
python -m pytest -q \
  tests/test_vaw_context_runtime.py \
  tests/test_vaw_context_agent.py \
  tests/test_vaw_context_packet.py \
  tests/test_vaw_near_field.py \
  tests/test_vaw_gripper_fk.py
```

当前实现不修改 CaP-X、RoboMEx、`capx_skill_rl` 或已有输出 trace。
