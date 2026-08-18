# VAW 当前架构契约

> 当前基线：Web schema 38 / `vaw-context-v37-working-memory` /
> renderer `context-web-v37-working-memory` / 固定 `2048×1280`。

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
Main ReAct Agent ── refine_action(action_id, instruction)
        │                         │
        │                         ▼
        │                Imagination SubAgent
        │                delta / rotate / gizmo
        │                         │ ready / failed
        ◄─────────────────────────┘
        │
        ├─ commit/reject refined spatial action
        ├─ direct delta/open/close
        └─ perception/proposal/done
```

Main 始终拥有任务级控制权。一次 `refine_action` 可以包含多个内部 turns，但对 Main 只是一个同步
Function transaction；内部调用和 rationale 只写嵌套 trace。

## 2. Function surfaces

Main：

```text
detection_and_sam
propose_grasps
locate_point
propose_pose
select
refine_action
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

- `select/propose_pose` 创建 revision-local 粗 `PendingAction`；
- `refine_action` 必须显式引用该 action ID；
- Imagination 只能编辑空间 pose，不能感知、控制夹爪、commit 或 done；
- `ready` 只赋予 commit 资格，`failed`/limit 回滚为不可 commit 的粗 Action；
- Main `delta_move/open_gripper/close_gripper` 立即执行并刷新真实 observation；
- `commit` 只执行 ready 的 cached spatial plan。
- CuRobo 负责 `select/propose_pose` 的粗空间规划。Imagination 不按“最后一次 edit 大小”路由，而按
  当前真实 TCP 到完整 target 的差值路由：`<=6 cm` 且 `<=20°` 使用 PyRoki，否则使用
  CuRobo。Main 的立即 `delta_move` 仍固定使用 PyRoki。两条路径共享公共 TCP 坐标，并由 cached
  plan 记录实际执行 backend；不做隐式 fallback。

## 3. Agent-visible Context

Main 每次 provider 请求都从当前状态重新构造：

```text
System Prompt
User Task
Task Memory                  # 最多 6 条已 dispatch 的物理 primitive
Live References              # 当前有效 region/point/seed/action
Current Function Event       # 最近一次最小结果投影
one current Main Canvas
Main Function definitions
```

不回放 transcript、旧 rationale、旧图片、receipt、revision、solver telemetry 或 backend error。
Imagination 只接收局部 instruction、edit summary、剩余预算和当前 Focused Canvas。

## 4. Canvas

- 上层：干净的当前 Agentview、Opposite View 与紧凑 RobotState；
- 下层：按需要显示 grounding、seeds、PendingAction Preview 或最新真实 Contact Views；
- 物理动作后的 detection/locate 只增加 evidence inset，不替换 Contact 连续性；
- 浅紫色三维细线表示未执行目标；当前机器人只来自真实 RGB，不叠加容易错位的白色 FK 轮廓。
  掌部横梁/指根额外使用很淡的半透明实体占用提示，手指与整臂不填充；
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
回灌 Agent Context。

## 6. 代码映射

```text
memory.py         TaskMemory / FunctionEvent / deterministic reducers
model.py          live evidence / PendingAction / RobotState
private.py        sensors / plans / presentation artifacts
functions.py      handlers and physical dispatch boundary
workspace.py      revision lifecycle + memory reduction
protocol.py       scoped prompts and Function definitions
runtime.py        Main ReAct + synchronous ImaginationRunner
packet.py         trusted Canvas compiler
trace.py          top-level and nested audit traces
vaw-ui/           deterministic Main/Focused renderer
```
