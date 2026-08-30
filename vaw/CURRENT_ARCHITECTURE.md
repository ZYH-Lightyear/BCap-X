# VAW 当前架构契约

> 当前基线：Web schema 54 / `vaw-context-v54-closed-gripper-z-cue` /
> renderer `context-web-v54-closed-gripper-z-cue` / Main 与 Focused Imagination 均固定为 `2048×1280`。

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
Task + Current Canvas / Live References
        │           + Embodied State + Interaction Memory
        │
        ▼
Main ReAct Agent ── imagine_action(instruction, action_id?)
        │                         │
        │                         ▼
        │                Imagination SubAgent
        │                shift / rotate / direction inspection
        │                         │ ready / failed
        ◄─────────────────────────┘
        │
        ├─ execute/discard planned|refined spatial action
        ├─ direct delta/open/close
        └─ perception/proposal/finish
```

Main 始终拥有任务级控制权。一次 `imagine_action` 可以包含多个内部 turns，但对 Main 只是一个同步
Function transaction；内部调用和 rationale 只写嵌套 trace。

## 2. Function surfaces

Main：

```text
detect_region
propose_grasps
locate_point
preview_pose
preview_grasp
imagine_action
move_tcp_delta
open_gripper
close_gripper
discard_action
execute_action
finish_task
```

Imagination：

```text
shift_preview
rotate_preview
inspect_rotation
finish_imagination
```

- `preview_grasp/preview_pose` 创建 revision-local planned `ActionProposal` 并缓存规划；
- `imagine_action` 是可选的空间推理工具，不是执行的前置条件；已有 Action 时引用其 ID，
  否则从当前真实 TCP 懒创建局部 Action；
- Imagination 只能编辑空间 pose，不能感知、控制夹爪、执行动作或结束任务；
- `ready` 只把 Action 标记为 refined（来源记录）；`failed`/limit 回滚到进入前的 ActionProposal
  与完整 artifacts，seeds 保持有效；策略级 `reason` 为
  `geometry_unresolved | plan_unavailable | turn_limit | subagent_error`；
- Main `move_tcp_delta/open_gripper/close_gripper` 立即执行并刷新真实 observation；
- `execute_action` 执行当前 Action 的可执行 cached spatial plan，planned 与 refined 均可；
  无可执行计划的 `execute_action` 是 pre-dispatch 拒绝，不改变世界。
- CuRobo 负责 `preview_grasp/preview_pose` 的初始空间规划。Imagination 不按“最后一次 edit 大小”路由，而按
  当前真实 TCP 到完整 target 的差值路由：`<=6 cm` 且 `<=20°` 使用 PyRoki，否则使用
  CuRobo。Main 的立即 `move_tcp_delta` 仍固定使用 PyRoki。两条路径共享公共 TCP 坐标，并由 cached
  plan 记录实际执行 backend；不做隐式 fallback。
- Main 的 Robot Function surface 只包含当前列出的机器人与 Imagination Functions，不包含外部技能加载。

## 2.5 证据复验（世界变化验证）

每次物理刷新后运行确定性的证据复验（`evidence.py`）：对每个 region/point 的 grounding 存档
（RGB-D 窗口 + SAM mask），先用 URDF FK 全臂剪影排除机器人自身像素，再比较深度与 RGB。深度
区分“被遮挡”（更近表面在前）与“已离开”（存档表面后方的背景暴露）。未变化的续期为
`verified`，被挡住的保留为 `occluded`（之后仍对照原始存档复验），被证实变化的删除。删除需要
正面证据，遮挡只推迟检查。`execute_action` 的 grasp 目标 region 由动作本身宣告失效，不等像素投票。

由此，四层验证从感知层开始：**证据复验** → 计划可行性（cached plan executable）→ execute TCP
error → 环境成功判据。完整复验结果进入 Function result 和 trace；下一轮 Live References 只呈现
仍然有效的当前引用及其 `occluded` 状态；
`detect_region`/`locate_point` 对仍 `verified` 的同 query 证据直接复用既有引用；重复闭合
夹爪附带 episode 级 advisory。以上都是可见性机制，不是门控。

## 3. Agent-visible Context

Main 每次 provider 请求都从当前状态重新构造：

```text
System Prompt
Task
Live References              # 当前有效 region/point/seed/action；occluded 证据带 status 标签；可知时含 source_follow_through；非零时含 imagination_attempts
Embodied State Card          # 最新 TCP/夹爪测量、最近动作及其 recency
Short-Term Interaction Memory # call/action 事务时间线，不推断任务语义
Function definitions         # 通过 provider tool channel 完整可见
one current Main Canvas
```

不回放 transcript、旧 rationale、旧图片、solver telemetry 或任务语义 summary。Interaction Memory 中的
outcome 只来自 Function transaction/receipt，并明确保留 revision 变化。
Imagination 只接收局部 instruction（场景任务，不是图例验收清单）、edit summary、剩余预算和当前 Focused Canvas。

## 4. Canvas

- Main 的常规 Contact Canvas 为横向四格：上排左侧是同一 revision 的最新主相机 `NOW`，
  上排右侧是独立的 `CONTACT AUXILIARY` 68° 斜俯辅助视角；下排是水平的
  `CONTACT FRONT` 与 `CONTACT SIDE`。辅助视角不替换正交接触视图，也不投影历史画面；
- grounding、seeds 和 ActionProposal 仍按当前决策态显示 evidence 或候选，不能用过期视觉替换当前状态；
- Focused Imagination 保持 Front/Side 两行严格等高。Main 中的 Front/Side 则并排显示。
  两种 projection 都按最终槽位原生渲染，
  不使用 letterbox；
- 物理动作后的 detection/locate 只增加 evidence inset，不替换 Contact 连续性；
- 浅紫色三维细线表示未执行目标；当前机器人主体来自真实 RGB。Contact View 只用青色半透明 mask
  标出当前 FK 两根手指，并持续用琥珀色标出所有 `verified` detect region 的传感器表面；
  region 被复验为 `occluded` 或因变化而删除时立即停止绘制；
- 实体占用只帮助识别局部穿插，不声称路径或碰撞已经检查；
- attachment OBB 只是假设几何，不是抓持、滑移、碰撞或释放真值；它可来自 grasp seed 或 region
  内 point。OBB 缺失时 Imagination 降级为 gripper-only 对齐，不伪造物体体积，也不因缺失本身失败；
- Contact Front/Side 分别沿夹爪闭合方向的正交/平行视轴并始终保持水平，动态选择无遮挡的正反侧；
  Auxiliary 使用 Front/Side 之间的对角方位和 68° 斜俯角，以减轻 Panda 手部对夹爪、载荷和容器的遮挡；
  垂线、落点十字与红色方向箭头只表达当前观测下的定性几何关系，不显示易被缩放/OCR 误读的高度
  或 dXY 数值；
- 上一物理动作前后的原始观测继续保存在 trace/video，供 Observatory 和 Progress Critic 使用，
  但不进入 Main Canvas。

## 5. Trace

```text
trace_dir/
├── steps.jsonl
├── context_XXXX.png
├── runtime_events.jsonl
├── meta.json
├── contexts/turn_XXXX/
│   ├── context.json
│   └── canvas.png
├── actions/action_XXXX/
│   ├── manifest.json
│   ├── agentview.mp4
│   ├── wrist.mp4
│   └── poster.jpg
└── subagents/imagination_XXXX/
    ├── meta.json
    ├── steps.jsonl
    └── context_XXXX.png
```

完整 Function result、planner diagnostics、模型原始输出、reward/env success 都允许进入 trace，但不能
回灌 Agent Context。每个 Imagination session 的 `meta.json` 含 `instruction`、`status` 和策略级
`reason`。Main 的每条 step 额外记录 Embodied State Card、provider 调用前的
Interaction Memory、本轮 InteractionEvent 与调用后的完整 Memory、
Function effect kind。

## 6. 代码映射

```text
memory.py         InteractionEvent / InteractionMemory
context_projection.py Embodied State Card + Main text context projector
model.py          live evidence (三态生命周期) / ActionProposal / ImaginationSession / RobotState
evidence.py       grounding 存档 + FK 剪影 + 深度/像素复验 verdicts
private.py        sensors / plans / evidence archives / presentation artifacts / imagination checkpoint
functions.py      handlers and physical dispatch boundary
workspace.py      revision lifecycle + evidence revalidation + physical dispatch
protocol.py       Function Registry / Function definitions / contract prompt
runtime.py        Main ReAct + Agent memory owner + synchronous ImaginationRunner
packet.py         trusted Canvas compiler
trace.py          top-level and nested audit traces
vaw-ui/           deterministic Main/Focused renderer
```
