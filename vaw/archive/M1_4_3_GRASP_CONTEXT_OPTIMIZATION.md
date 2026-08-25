# ARCHIVED — VAW M1.4.3 20-Turn Grasp Context Optimization

> **历史实验计划，不是当前规范。** 本文的 Context v7、20-turn 基线、Imagination
> `open/close` 和当时的可视化缺陷保留为失败证据；后续实现已跨多个 M1.5 版本演进。
> 当前代码契约见 [`CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md)，当前验收路线见
> [`M1_5_AGENTIC_SYSTEM_COMPLETION.md`](../M1_5_AGENTIC_SYSTEM_COMPLETION.md)。

## 目标

在固定真实评测 `libero_object_swap:0, seed=1` 中，VAW Agent 必须在 **20 个总 Function
turn** 内把 `alphabet soup can` 从原支撑面真实抓起。仅完成 detection、生成轨迹、到达目标
TCP、闭合夹爪或由 Agent 声称成功，均不算通过。

通过证据必须同时包含：

1. commit 后的当前真实 RGB / RGB-D 近场显示目标离开原支撑；
2. 后续至少一次不小于 3 cm 的上移动作中，目标与夹爪共同运动；
3. Agent-visible Context 不包含 reward、task success、privileged object pose 或 evaluator 结论；
4. `steps.jsonl` 中满足上述状态时的总 turn 不超过 20。

环境真值或对象位姿只允许由独立 evaluator 用于验收，不进入 Main / Imagination Context。

## 当前基线：M1.4.2 / Context v7

当前架构为 history-free Main / Imagination 双 Agent：

```text
Main：任务理解、detection_and_sam、ActionSeed、最终 review、commit
Imagination：delta_move、rotate、open/close preview、finish_imagination
```

已有优势：

- CURRENT OBSERVED 与 IMAGINATION 分离；
- 只有 commit 改变物理世界；
- gripper-local 使用当前 RGB-D 近场；
- Function transcript 不再污染策略 Context；
- region / point / seed 保持 revision-local。

基线失败证据 `dual_agent_review_v7_t0_s1`：

- Imagination 在 Z 方向反复抵消并以 `budget_exhausted` 交回；
- Main commit 的 `a2` 只执行 arm，gripper 未改变；
- commit 后动作意图与执行阶段被清空，下一轮重新调用旧 `inspect`；
- detection 随后把前景红绿罐误认为 alphabet soup，抓取链偏离目标；
- 因而没有在 20 turn 内形成已验证抓持。

## Context Builder 审计

当前 Main 请求包含：

```text
System Prompt
User Task + minimal policy state + optional handoff
CURRENT Context Canvas
optional raw PREVIOUS OBSERVED image after commit
```

当前 Imagination 请求包含：

```text
System Prompt
User Task + refinement goal + current ActionTarget
CURRENT Context Canvas
```

主要缺陷：

1. revision 刷新正确清除了旧 evidence，但同时清除了刚执行动作的目的、实际阶段和 handoff；
2. previous RGB 没有说明执行的是 arm、gripper 还是两者，不能承担因果记忆；
3. post-commit 下层退化为空白 idle，模型容易把当前状态理解为新的任务起点；
4. Main 的 refinement goal 只存在于 Imagination session，无法跨一次 commit；
5. commit result 只进入 trace，下一轮 Main 不知道 gripper 是否实际改变；
6. 感知 Function 名称 `inspect` 容易被理解成“重新看图”，而它实际执行 detection + SAM。

## V1：Post-Commit Causal Context

下一版本只增加一个固定大小、覆盖更新的因果事件，不恢复 Function history：

```text
PostCommitContext
├── intent: str
├── executed_stages: arm | gripper | arm+gripper
└── outcome: completed | arm_failed | gripper_failed
```

约束：

- 不包含 receipt ID、旧 revision、过期 action ID 或 reasoning transcript；
- 每次 commit 覆盖上一条，不形成列表；
- detection_and_sam 或其他非物理调用不会清除它；
- reset 清除；
- Canvas 的当前 robot state 继续提供实际 TCP、joint 和 gripper opening，不在文本重复数值。

post-commit Canvas 下层改为：

```text
LAST PHYSICAL CHANGE
intent · executed stages · outcome

BEFORE COMMIT              CURRENT OBSERVED
```

前后图由 compiler 组合进同一张 Context Canvas；停止向 provider 额外发送格式不同的 raw previous
image。该页面只呈现观测变化，不判断 grasp success。

Prompt 明确：当前 Canvas 已经是最新 observation；`detection_and_sam` 只创建检测/分割 region，
不刷新图像。

## 迭代与提交规则

每个版本必须按以下顺序完成：

1. 在本文档记录假设、接口和预期改善；
2. 实现代码与离线回归测试；
3. 生成固定 fixture Canvas，检查信息泄漏和视觉可读性；
4. 创建独立 git commit；
5. 运行一次真实 `libero_object_swap:0, seed=1`；
6. 在下表记录 trace、20-turn gate 和下一失败原因。

| 版本 | 核心假设 | Commit | 真实 Trace | ≤20 turn 抓起 | 结论 |
|---|---|---|---|---|---|
| M1.4.2 baseline | 双 Agent 与无 transcript history 能降低决策噪声 | 待冻结 | `dual_agent_review_v7_t0_s1` | 否 | post-commit 因果状态丢失；detection 偏离目标 |
| M1.4.3 V1 | 一条持续的物理事件足以避免 arm-only commit 后重启抓取链 | 待实现 | 待运行 | 待验证 | 待验证 |

## 后续候选假设

只有 V1 真实 trace 仍失败时，才按证据选择下一项，避免同时修改过多变量：

- Imagination 收敛：显式显示累计变换与最近一次反向编辑，减少来回抵消；
- seed 质量：为目标 region 增加语义一致性复核，而非硬编码 top-down 规则；
- close 几何：在 JAW PLANE 中强化闭合通道与目标点云的相交关系；
- 抓持验收：提供纯观测的 before/after lift comparison，不向 Agent 注入真值；
- Main review：区分 `completed` 与 `budget_exhausted`，但不把 LIMIT 变成自动批准或硬 gate。

## 非目标

- 不加入 pick/place phase machine；
- 不硬编码 alphabet soup、basket 或固定动作序列；
- 不恢复 K-history；
- 不把 evaluator 真值写入 Canvas 或 Prompt；
- 不以 IK returned、trajectory checked、TCP error 或 closed opening 代替真实抓持证据。
