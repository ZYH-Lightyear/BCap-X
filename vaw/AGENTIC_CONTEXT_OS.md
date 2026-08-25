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
  "action_proposal":{"action_id":"a2","intent":"approach can for grasp","state":"planned"},
  "valid_region_ids":["region3"],
  "valid_point_ids":[],
  "valid_seed_ids":["s6","s7"]
}
```

seed 和 action 是 revision-local：物理动作刷新 observation 后失效，因为其缓存计划的出发状态已不存在。
region 和 point 则跨物理动作持续：每次刷新后由确定性的证据复验对照新画面裁决——

- `verified`：存档窗口（FK 机器人剪影之外的可见部分）深度与像素都与当前画面一致，条目续期，
  Live References 中不带标签；
- `occluded`：机器人身体或更近的表面挡住了检查，条目保留并带 `"status":"occluded"`，
  之后仍对照原始存档复验，视野恢复即回到 verified；
- 变化被证实（存档表面后方的背景暴露，或原位外观被替换）：条目删除，且只在删除时报告。

“无法证明存在”永不当作“证明不存在”：删除需要正面证据，遮挡只是推迟检查。此外，commit 的
grasp 目标 region 由动作本身宣告失效，不等像素投票。物体未被触碰时，Agent 应直接复用仍列出的
引用，不重新 detection/locate；Task Memory 不失效。同一 revision 内若 Imagination 已被调用，
Live References 还会带一个最小失败摘要：

```json
{"imagination_attempts":{"total":3,"failed":3,"last_reason":"turn_limit","failed_source_refs":["s2"]}}
```

只含计数、最近策略级 `reason` 和已失败 source 的 ID；不含 instruction 原文或 solver telemetry。

### 2.4 Current Function Event

Current Event 是最近一次 Function result 的 handler-owned 最小投影，只保留状态和新引用：

```json
{"function":"select","status":"ok","references":{"action_id":"a2"}}
```

失败只保留一条可行动的通用信息。pre-dispatch 拒绝必须写明世界未变：

```json
{"function":"commit","status":"failed","message":"rejected before dispatch; the real world is unchanged: active action 'a1' has no executable cached plan"}
```

已 dispatch 的物理失败才暗示效果不确定。`call_imagination` 失败携带策略级 `reason`：

```json
{"function":"call_imagination","status":"failed","references":{"action_id":"a3","reason":"turn_limit"},"message":"local imagination did not deliver a refined action; reason=turn_limit"}
```

物理动作成功时，Current Event 附带一行确定性的 world change check，报告复验结论；重复闭合
夹爪还会附带一条 advisory（只提示，不设门控）：

```json
{"function":"close_gripper","status":"ok","message":"world change check — changed and dropped: region1(can); verified unchanged: region3, p1; this is close_gripper attempt #2 this episode near 'can'; check the gripper opening and Contact views before releasing again"}
```

下一次 Function transaction 会完整覆盖 Current Event。CuRobo/PyRoki 状态、异常文本、IK 数值、
position error 和 trace 路径不得进入这里。

## 3. Function result 的唯一归档规则

| Function 结果 | Task Memory | Live References | Current Event | Trace |
|---|---:|---:|---:|---:|
| perception 成功 | 否 | 新 region/point | 最小引用 | 完整 |
| proposal 成功 | 否 | 新 seed/action | 最小引用 | 完整 |
| perception/planner 未 dispatch 失败 | 否 | 不变 | 一轮失败 | 完整 |
| physical dispatch 成功 | `executed` | seed/action 清空；region/point 复验 | ok + world change check | 完整 |
| physical dispatch 后失败 | `effect_unknown` | seed/action 清空；region/point 复验 | 一轮失败 | 完整 |
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

Imagination 是 `call_imagination(action_id, instruction)` 内同步调用的局部 SubAgent。`action_id` 必须
显式存在，禁止从当前 TCP 隐式创建动作。它只接收 instruction、当前 Focused Canvas、累计 edit
summary、最近两次 edit 和剩余预算；不接收 Task Memory 或 Main transcript。

每次空间 edit 是一个事务：只有规划返回可执行计划时才整体替换 working proposal 与 cached plan；
规划失败时二者都不变，返回 `{"preview": "unchanged", "reason": "plan_unavailable"}`，失败 target 与
planner telemetry 只写 trace。失败因此是事件而不是状态——Imagination 下一轮会看到该原因，但
Canvas、下一次编辑的基准和可交付的 Action 始终停在最后一个可执行目标上。

`failed`/turn limit/provider error 回滚到进入 Imagination 前的 ActionProposal 与完整 artifacts
（seeds 与当前 revision 的感知证据保持有效）。策略级失败类别只有
`geometry_unresolved`、`plan_unavailable`、`turn_limit`、`subagent_error`，进入 Function result
与该 session 的 `meta.json`。`ready` 只把 Action 标记为 refined 并交回 Main 审核，不自动执行。
planned/refined 只记录来源，不区分 commit 资格：只要存在与当前 target 对齐的可执行
cached plan，Main 审核 Preview 后即可 commit；无可执行计划的 commit 是 pre-dispatch 拒绝，不改变
世界。Imagination 因此是可选的空间推理工具，不是 commit 的前置条件。

空间规划采用自适应双层路由：`select/propose_pose` 的 planned 目标由 CuRobo 规划；Imagination
每次编辑后根据“当前真实 TCP → 完整 target”重新选择 backend。仅当平移不超过 `6 cm`且姿态
差不超过 `20°` 时使用 PyRoki；远处目标即使只编辑了 `1 cm`，仍使用 CuRobo。Main 的立即
`delta_move` 因为始终从真实 TCP 产生单次厘米级 target，固定使用 PyRoki。两者经过 adapter
共享同一个 policy-visible TCP 定义。`commit` 必须使用生成当前 cached plan 的同一 backend；失败时
不进行静默 fallback。PyRoki 的 endpoint IK 不提供路径或碰撞保证，这些边界不得写成视觉真值。

## 5. Canvas 路由

Canvas schema 固定为：

```text
context:  vaw-context-v46-oblique-contact
web:      46
renderer: context-web-v46-oblique-contact
viewport: 2048×1280, DPR=1
```

携带载荷时 CONTACT SIDE 抬升为斜俯视（默认 55°），标题标出 `OBLIQUE <角度>° DOWN`。
两个近水平面板只能就高度互相印证；横向对齐以带 OBLIQUE 标记的那一幅为准。

下层按当前决策证据路由：

1. imagination session active → Focused Preview；
2. 当前产生 seeds → 候选比较；
3. action proposal → 空间 Preview；
4. 最近执行物理动作 → 最新真实 Contact Views；
5. 之后调用 detection/locate → Contact Views 保持，grounding 只作为小 inset；
6. 普通 grounding → 当前 region/point evidence；
7. 无相关 evidence → 扩大当前真实世界，不放低密度报告。

不存在 `post_commit` 页面、before/current 旧图对比或 error 报告页。

Focused Contact Camera 在一次 Imagination session 内锁定：一条水平视轴始终与夹爪闭合方向平行，另一条
始终与其正交；presenter 只在每条轴的正/反观察侧之间选择，使两张图都尽量看清目标。场景内不绘制
位移箭头，右下角只保留目标深度处的 `5 cm` 标尺；最近一次数值 edit 仍以紧凑文字条显示。
Main Canvas 上层只显示 Agentview、Opposite View 和紧凑 `GRIP` 数值；TCP 与 joints 留在结构化状态/trace，
不再占据独立视觉卡片。下层删除重复全局图，Contact Front 与 Contact Side 获得严格相等的面积。
Focused Imagination Canvas 同样只显示两张等高 Contact raster。两种 projection 均按最终槽位原生渲染，
不通过 letterbox 浪费像素。

Contact View 用青色半透明 mask 标出当前 FK 的两根真实手指；琥珀色只标出当前 revision 内由感知工具得到的
操作对象表面。对象证据过期时不沿用旧 mask，也不以仿真 segmentation 或 carried proxy 冒充真实观测。

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
