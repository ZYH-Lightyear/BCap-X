# VAW 当前架构契约

> 当前基线：Web schema 41 / `vaw-context-v41-evidence-lifecycle` /
> renderer `context-web-v41-evidence-lifecycle` / 固定 `2048×1280`。

Context 与 Memory 的完整规范见 [`AGENTIC_CONTEXT_OS.md`](AGENTIC_CONTEXT_OS.md)。本文只描述
当前控制架构与代码映射；历史 Milestone 文档不再作为接口依据。

## 1. 顶层流程

```text
current LIBERO-PRO observation
        │
        ▼
episode-private RGB-D / calibration / masks / plans
        │ trusted compiler
        ▼
Current Canvas + Task Memory + Live References + Current Event
        │
        ▼
Main ReAct Agent ── call_imagination(action_id, instruction)
        │                         │
        │                         ▼
        │                Imagination SubAgent
        │                delta / rotate / gizmo
        │                         │ ready / failed
        ◄─────────────────────────┘
        │
        ├─ commit/reject planned|refined spatial action
        ├─ direct delta/open/close
        └─ perception/proposal/done
```

Main 始终拥有任务级控制权。一次 `call_imagination` 可以包含多个内部 turns，但对 Main 只是一个同步
Function transaction；内部调用和 rationale 只写嵌套 trace。

## 2. Function surfaces

Main：

```text
detection_and_sam
propose_grasps
locate_point
propose_pose
select
call_imagination
delta_move
open_gripper
close_gripper
reject_action
commit
done
```

Imagination：

```text
delta_move
rotate
show_rotation_gizmo
finish_imagination
```

- `select/propose_pose` 创建 revision-local planned `ActionProposal` 并缓存规划；
- `call_imagination` 必须显式引用该 action ID，是可选的空间推理工具，不是 commit 的前置条件；
- Imagination 只能编辑空间 pose，不能感知、控制夹爪、commit 或 done；
- `ready` 只把 Action 标记为 refined（来源记录）；`failed`/limit 回滚到进入前的 ActionProposal
  与完整 artifacts，seeds 保持有效；策略级 `reason` 为
  `geometry_unresolved | plan_unavailable | turn_limit | subagent_error`；
- Main `delta_move/open_gripper/close_gripper` 立即执行并刷新真实 observation；
- `commit` 执行当前 Action 的可执行 cached spatial plan，planned 与 refined 均可；
  无可执行计划的 commit 是 pre-dispatch 拒绝，不改变世界。
- CuRobo 负责 `select/propose_pose` 的初始空间规划。Imagination 不按“最后一次 edit 大小”路由，而按
  当前真实 TCP 到完整 target 的差值路由：`<=6 cm` 且 `<=20°` 使用 PyRoki，否则使用
  CuRobo。Main 的立即 `delta_move` 仍固定使用 PyRoki。两条路径共享公共 TCP 坐标，并由 cached
  plan 记录实际执行 backend；不做隐式 fallback。

## 2.5 证据复验（世界变化验证）

每次物理刷新后运行确定性的证据复验（`evidence.py`）：对每个 region/point 的 grounding 存档
（RGB-D 窗口 + SAM mask），先用 URDF FK 全臂剪影排除机器人自身像素，再比较深度与 RGB。深度
区分“被遮挡”（更近表面在前）与“已离开”（存档表面后方的背景暴露）。未变化的续期为
`verified`，被挡住的保留为 `occluded`（之后仍对照原始存档复验），被证实变化的删除。删除需要
正面证据，遮挡只推迟检查。commit 的 grasp 目标 region 由动作本身宣告失效，不等像素投票。

由此，四层验证从感知层开始：**证据复验** → 计划可行性（cached plan executable）→ commit TCP
error → 环境成功判据。复验结果以一行 world change check 进入 Current Event
（`changed and dropped / occluded, kept unverified / verified unchanged`）；
`detection_and_sam`/`locate_point` 对仍 `verified` 的同 query 证据直接复用既有引用；重复闭合
夹爪附带 episode 级 advisory。以上都是可见性机制，不是门控。

## 3. Agent-visible Context

Main 每次 provider 请求都从当前状态重新构造：

```text
System Prompt
User Task
Task Memory                  # 最多 6 条已 dispatch 的物理 primitive
Live References              # 当前有效 region/point/seed/action；occluded 证据带 status 标签；非零时含 imagination_attempts
Current Function Event       # 最近一次最小结果投影
one current Main Canvas
Main Function definitions
```

不回放 transcript、旧 rationale、旧图片、receipt、revision、solver telemetry 或 backend error。
Imagination 只接收局部 instruction、edit summary、剩余预算和当前 Focused Canvas。

## 4. Canvas

- 上层：干净的当前 Agentview、Opposite View 与 header 内的紧凑 `GRIP` 数值；TCP/joints 不占视觉卡片；
- 下层：按需要显示 grounding、seeds、ActionProposal Preview 或最新真实 Contact Views；
- Main 的 Contact 模式删除重复全局图，Front/Side 两栏严格等宽；Focused Imagination 同样删除全局图，
  Front/Side 两行严格等高。两种 projection 都按最终槽位原生渲染，不使用 letterbox；
- 物理动作后的 detection/locate 只增加 evidence inset，不替换 Contact 连续性；
- 浅紫色三维细线表示未执行目标；当前机器人主体来自真实 RGB。Contact View 只用青色半透明 mask
  标出当前 FK 两根手指，并用琥珀色标出当前 revision 有效的操作对象表面；证据失效后不沿用旧 mask；
- 实体占用只帮助识别局部穿插，不声称路径或碰撞已经检查；
- attachment OBB 只是假设几何，不是抓持、滑移、碰撞或释放真值；它可来自 grasp seed 或 region
  内 point。OBB 缺失时 Imagination 降级为 gripper-only 对齐，不伪造物体体积，也不因缺失本身失败；
- Contact Front/Side 分别沿夹爪闭合方向的正交/平行水平视轴，动态选择无遮挡的正反侧；画面内不
  放位移箭头，只保留 5 cm 标尺；
- 不存在 post-commit 报告页、before/current 旧图或 Canvas error 报告。

## 5. Trace

```text
trace_dir/
├── steps.jsonl
├── context_XXXX.png
├── runtime_events.jsonl
├── meta.json
└── subagents/imagination_XXXX/
    ├── meta.json
    ├── steps.jsonl
    └── context_XXXX.png
```

完整 Function result、planner diagnostics、模型原始输出、reward/env success 都允许进入 trace，但不能
回灌 Agent Context。每个 Imagination session 的 `meta.json` 含 `instruction`、`status` 和策略级
`reason`。

## 6. 代码映射

```text
memory.py         TaskMemory / FunctionEvent / deterministic reducers
model.py          live evidence (三态生命周期) / ActionProposal / ImaginationSession / RobotState
evidence.py       grounding 存档 + FK 剪影 + 深度/像素复验 verdicts
private.py        sensors / plans / evidence archives / presentation artifacts / imagination checkpoint
functions.py      handlers and physical dispatch boundary
workspace.py      revision lifecycle + evidence revalidation + memory reduction
protocol.py       scoped prompts and Function definitions
runtime.py        Main ReAct + synchronous ImaginationRunner
packet.py         trusted Canvas compiler
trace.py          top-level and nested audit traces
vaw-ui/           deterministic Main/Focused renderer
```