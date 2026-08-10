# VAW — Visual Action Workspace

`vaw` 是面向 LIBERO-PRO 的双 Agent Visual Context Runtime。它把当前真实观测与动作想象
编译为固定 `1920×1440` Canvas，但模型不点击页面：动作通道始终是 structured Function
call。

当前架构不再回放 Function history，也不维护 Persistent Waypoint。Main Agent 负责语义决策
和物理提交；Imagination Agent 在独立、无物理副作用的会话中连续检查和微调一个
`ActionTarget`。

双 Agent 基线见 [`M1_4_2_DUAL_AGENT_RUNTIME.md`](M1_4_2_DUAL_AGENT_RUNTIME.md)。当前的
20-turn 真实抓取优化、Context Builder 审计和逐版本验收记录见
[`M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md`](M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md)。

## Runtime

```text
CURRENT OBSERVED
      │
      ▼
Main Agent ── perception / ActionSeed ──► Imagination Agent
   ▲                                      │
   │                  delta / rotate / gripper preview
   │                                      │
   └──── ActionReview / failed / budget exhausted ┘
   │
   └── Main reviews ── commit(ActionReview) ──► physical world ──► fresh observation
```

每次 provider 请求都从零构造。Main 只看到 task、当前 Canvas、minimal policy state、最近一次
handoff，以及 commit 后仅出现一次的 previous-observed 图。Imagination 只看到 task、
refinement goal、当前 `ActionTarget` 和当前 Canvas。两边都看不到对方或自己的历史
call/result/rationale。

## Function space

Main Agent：

```text
detection_and_sam(query, within_region_id?)
locate_point(query, within_region_id?)
propose_grasps(region_id) -> seed_ids
propose_pose(point_id, offset_xyz, quaternion_xyzw?)
select(seed_id)
delta_move(delta_xyz_m, frame)
rotate(axis, angle_deg, frame)
open_gripper()
close_gripper()
commit(action_id)
done(success)
```

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
默认最多连续想象 6 轮，达到上限时以 `budget_exhausted` 原因交回 Main。无论显式完成还是
预算耗尽都不代表动作获批；只有 Main 查看最终 Preview 后调用 `commit` 才构成批准。

## Canvas

Web schema 8 / `vaw-context-v7-dual-agent-review` / renderer
`context-web-v7-dual-agent-review`：

- 上层 `CURRENT OBSERVED`：agentview 始终为真实 RGB；非想象时 gripper-local 为真实 RGB-D，
  Imagination 期间在同一份当前点云上叠加紫色虚拟机器人并提高近场采样密度；
- 下层 `IMAGINATION WORKSPACE`：grounding、ActionSeed、editing、reviewed、error 或 idle；
- 紫色机器人只存在于下层，并始终表示未执行的预测；
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
