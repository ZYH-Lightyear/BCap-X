# VAW Agentic Context OS

本文是 VAW 当前 Context、Memory 与 Canvas 的唯一规范。历史 Milestone 文档只记录实验演进，
不得作为实现接口依据。

## 1. 目标

Main Agent 每轮只接收能够继续操控的当前事实：

```text
Current Canvas
+ Task Memory
+ Live References
+ Current Function Event
→ one Main ReAct decision
```

四层互不复制。系统不回放 Function transcript、旧 rationale、旧图片或 planner telemetry，也不
建立 object tracker、PDDL belief、任务 phase machine 或模型生成的自由文本 memory。

## 2. 四层 Context

### 2.1 Current Canvas

Canvas 负责视觉事实，不负责解释执行结果：

- 上层：当前 revision 的 Agentview、Opposite View 和紧凑 RobotState；
- 下层：当前 relevant grounding、ActionSeed、空间 Preview 或最新真实 Contact View；
- 浅紫细线表示未执行机器人几何；只有掌部横梁/指根使用淡紫半透明实体占用提示，手指与整臂
  继续保持细轮廓，避免遮挡真实 RGB；琥珀色体积只是假设几何；
- 淡紫实体占用用于提醒横梁和指根不能穿入目标，不等同于 collision checker；
- 不显示 Task、receipt、revision、Function JSON、backend error、TCP error 或协议说明。

### 2.2 Task Memory

Task Memory 是 runtime 写入、最大六条、overwrite/compact-only 的物理 primitive ledger：

```json
[
  {"op":"move_to","status":"executed","intent":"approach alphabet soup for grasp"},
  {"op":"close_gripper","status":"executed"},
  {"op":"delta_move","status":"executed","frame":"base","delta_xyz_m":[0,0,0.03]}
]
```

只允许四种 primitive：

```text
move_to
delta_move
open_gripper
close_gripper
```

状态只有：

- `executed`：控制命令完成；不声称抓住、随动、释放、包含或任务成功；
- `effect_unknown`：控制命令已经下发但失败，世界可能已改变。

Memory 不保存 region/point/seed/action ID，不保存感知或规划失败，不保存模型 rationale。感知调用
不能清除或覆盖它。连续同 frame 的成功 `delta_move` 合并为累计位移；相邻重复的相同成功夹爪
命令折叠；`effect_unknown` 永不与成功记录合并。

### 2.3 Live References

Live References 只列出当前 observation 中仍可合法引用的短期句柄：

```json
{
  "pending_action":{"action_id":"a2","intent":"approach can for grasp","state":"coarse"},
  "valid_region_ids":["region3"],
  "valid_point_ids":[],
  "valid_seed_ids":["s6","s7"]
}
```

region、point、seed 和 action 全部 revision-local。物理动作刷新 observation 后统一失效；Task
Memory 不失效。

### 2.4 Current Function Event

Current Event 是最近一次 Function result 的 handler-owned 最小投影，只保留状态和新引用：

```json
{"function":"select","status":"ok","references":{"action_id":"a2"}}
```

失败只保留一条可行动的通用信息：

```json
{"function":"commit","status":"failed","message":"the spatial command did not complete; its effect is uncertain"}
```

下一次 Function transaction 会完整覆盖 Current Event。CuRobo/PyRoki 状态、异常文本、IK 数值、
position error 和 trace 路径不得进入这里。

## 3. Function result 的唯一归档规则

| Function 结果 | Task Memory | Live References | Current Event | Trace |
|---|---:|---:|---:|---:|
| perception 成功 | 否 | 新 region/point | 最小引用 | 完整 |
| proposal 成功 | 否 | 新 seed/action | 最小引用 | 完整 |
| perception/planner 未 dispatch 失败 | 否 | 不变 | 一轮失败 | 完整 |
| physical dispatch 成功 | `executed` | revision 后清空 | ok | 完整 |
| physical dispatch 后失败 | `effect_unknown` | revision 后清空 | 一轮失败 | 完整 |
| protocol/no-call/multi-call | 否 | 不变 | 一轮协议失败 | 完整 |

关键边界是“是否已 dispatch 物理控制”，而不是 Function 名称。规划失败不能伪装成物理历史；
控制器失败也不能被丢弃成普通 error。

## 4. Main 与 Imagination

Main 每轮重新构造且只构造：

```text
System Prompt
User Task
Task Memory
Live References
Current Function Event (if any)
one current Main Canvas
Main Function definitions
```

Imagination 是 `refine_action(action_id, instruction)` 内同步调用的局部 SubAgent。`action_id` 必须
显式存在，禁止从当前 TCP 隐式创建动作。它只接收 instruction、当前 Focused Canvas、累计 edit
summary、最近两次 edit 和剩余预算；不接收 Task Memory 或 Main transcript。

`failed`/turn limit 回滚到进入 refinement 前的粗 Action，该 Action 仍不可 commit；`ready` 只赋予
Main 审核后的 commit 资格，不自动执行。

空间规划采用自适应双层路由：`select/propose_pose` 的粗目标由 CuRobo 规划；Imagination
每次编辑后根据“当前真实 TCP → 完整 target”重新选择 backend。仅当平移不超过 `6 cm`且姿态
差不超过 `20°` 时使用 PyRoki；远处粗目标即使只编辑了 `1 cm`，仍使用 CuRobo。Main 的立即
`delta_move` 因为始终从真实 TCP 产生单次厘米级 target，固定使用 PyRoki。两者经过 adapter
共享同一个 policy-visible TCP 定义。`commit` 必须使用生成当前 cached plan 的同一 backend；失败时
不进行静默 fallback。PyRoki 的 endpoint IK 不提供路径或碰撞保证，这些边界不得写成视觉真值。

## 5. Canvas 路由

Canvas schema 固定为：

```text
context:  vaw-context-v37-working-memory
web:      38
renderer: context-web-v37-working-memory
viewport: 2048×1280, DPR=1
```

下层按当前决策证据路由：

1. refinement active → Focused Preview；
2. 当前产生 seeds → 候选比较；
3. pending action → 空间 Preview；
4. 最近执行物理动作 → 最新真实 Contact Views；
5. 之后调用 detection/locate → Contact Views 保持，grounding 只作为小 inset；
6. 普通 grounding → 当前 region/point evidence；
7. 无相关 evidence → 扩大当前真实世界，不放低密度报告。

不存在 `post_commit` 页面、before/current 旧图对比或 error 报告页。

Focused Contact Camera 在一次 refinement 内锁定：一条水平视轴始终与夹爪闭合方向平行，另一条
始终与其正交；presenter 只在每条轴的正/反观察侧之间选择，使两张图都尽量看清目标。场景内不绘制
位移箭头，右下角只保留目标深度处的 `5 cm` 标尺；最近一次数值 edit 仍以紧凑文字条显示。

## 6. 私有状态与 Trace

以下数据只存在于 episode-private context 或 trace：RGB-D、相机标定、raw mask/cloud、motion plan、
returned joints、planner diagnostics、原始 Function result、模型 rationale、reward、env success 和
privileged state。

Trace 可以完整复现每次请求与控制，但 trace 数据不能反向回灌策略 Context。

## 7. 不变量

1. Main 每轮只调用一个 Main Function；Imagination 每轮只调用一个局部 Function。
2. Memory 只能由实际 handler transaction 写入，模型不能直接编辑。
3. 未 dispatch 的失败永远不进入 Task Memory。
4. `executed` 永远不等于 task effect verified。
5. 当前 Canvas 优先于 Memory 中的历史命令预期。
6. Live References 与 Task Memory 生命周期完全分离。
7. renderer 失败终止实验，不静默回退。
8. 所有完整诊断留在 trace，不通过错误文本污染下一轮推理。
